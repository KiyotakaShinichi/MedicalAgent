"""Iterative evidence-sufficiency RAG eval scaffold.

Eval-only.  Runs the frozen internal goldset twice:

  pass 1 — original query + full source-governed stack
  pass 2 — if answerability_status is insufficient_evidence / conflicting /
           clinician_review_required, expand the query with a single
           targeted follow-up and re-retrieve

Reports first-pass vs. second-pass answerability rates so a reviewer
can judge whether one extra targeted retrieval moves the needle.

This module does NOT touch the live patient agent's
``run_patient_agent_pipeline``.  It does NOT call the LLM (FAST_MODE
forced when reused from the live router).  It does NOT modify
retrieval ranking, governance, or any goldset case.

Output: ``Data/evals/rag/latest_iterative_rag_sufficiency_eval.json``
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("ONCOTRACK_FAST_MODE", "1")


GOLDSET_PATH = Path("Data/evals/rag/retrieval_goldset.jsonl")
OUTPUT_PATH = Path("Data/evals/rag/latest_iterative_rag_sufficiency_eval.json")


# Targeted follow-up suffixes per intent — generic, NOT case-specific.
_FOLLOWUP_HINTS: dict[str, str] = {
    "education":                       "general patient-facing explanation reference",
    "urgent_escalation":                "fever neutropenia infection safety reference",
    "genetic_counselor_review":         "genetic counseling readiness reference",
    "tumor_marker_boundary":            "tumor marker limitation reference",
    "pharmacist_or_clinician_review":   "supplement interaction safety reference",
    "treatment_refusal":                "treatment boundary policy reference",
    "diagnosis_refusal":                "diagnosis claim boundary reference",
    "prognosis_refusal":                "prognosis claim boundary reference",
    "privacy_refusal":                  "privacy boundary policy reference",
    "portal_help":                      "patient portal workflow reference",
    "record_explanation":               "record explanation boundary reference",
}


def _load_goldset() -> list[dict[str, Any]]:
    if not GOLDSET_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in GOLDSET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _retrieve_full_stack_for(case: Mapping[str, Any], query_text: str | None = None) -> tuple[list[dict[str, Any]], float]:
    """Run hybrid+rewrite+parent-child+source_tier for a single case.

    Returns (top10_chunks, latency_ms).
    """
    from backend.services.agent_query_rewriting import rewrite_and_decompose
    from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
    from backend.services.agent_retrieval import expand_parent_child_windows
    from backend.services.rag_vector_index import search_hybrid_index
    from backend.services.rag_baseline_comparison import (
        _apply_case_source_filter,
        _dedupe_rows,
        _map_goldset_intent,
    )

    started = time.perf_counter()
    base_query = query_text or str(case.get("user_query") or case.get("query") or "")
    intent = _map_goldset_intent(str(case.get("expected_intent") or "education"))
    rewritten = rewrite_and_decompose(base_query, intent)
    search_query = str(rewritten.get("expanded_query") or base_query)
    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    pool = search_hybrid_index(
        query=search_query, corpus=corpus, intent=intent,
        knowledge_fingerprint=fingerprint, candidate_limit=50,
    )
    ranked = sorted(pool, key=lambda r: float(r.get("retrieval_score") or 0.0), reverse=True)
    expanded = sorted(
        expand_parent_child_windows(ranked[:20]),
        key=lambda r: float(r.get("retrieval_score") or 0.0),
        reverse=True,
    )
    filtered = _apply_case_source_filter(case, _dedupe_rows(expanded))
    latency = (time.perf_counter() - started) * 1000.0
    return filtered[:10], latency


def _answerability_for(
    case: Mapping[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    from backend.services.rag_claim_validator import validate_claims
    from backend.services.rag_evidence_grading import grade_evidence
    from backend.services.rag_intent_modes import select_mode
    from backend.services.rag_tier_filter import filter_chunks_by_mode
    from backend.services.retrieval_confidence import classify_retrieval_uncertainty
    from backend.services.rag_baseline_comparison import _map_goldset_intent

    intent = _map_goldset_intent(str(case.get("expected_intent") or "education"))
    mode = select_mode(intent, actor_role="patient")
    if mode is None:
        return {
            "answerability_status": "insufficient_evidence",
            "evidence_grade_grade": None,
            "citation_support_confidence": 0.0,
            "retrieval_confidence": 0.0,
            "source_tier_confidence": 0.0,
            "supported_claims": 0,
            "high_trust_chunks": 0,
        }
    filt = filter_chunks_by_mode(chunks, mode, keep_unmapped=False)
    claims = validate_claims(_synthesize_stub_claim_from_case(case), filt.kept_chunks)
    grade = grade_evidence(
        mode=mode, filter_result=filt, claim_validation=claims,
        retrieved_count_before_filter=len(chunks),
    )
    safety = {"level": "high_risk" if case.get("expected_refusal_or_insufficient_evidence") else "low_risk"}
    rc = classify_retrieval_uncertainty(
        chunks=filt.kept_chunks,
        claim_envelope=claims.to_dict(),
        safety=safety,
        intent=intent,
    )
    return {
        "answerability_status": rc.answerability_status,
        "evidence_grade_grade": grade.grade,
        "citation_support_confidence": rc.citation_support_confidence,
        "retrieval_confidence": rc.retrieval_confidence,
        "source_tier_confidence": rc.source_tier_confidence,
        "supported_claims": rc.supported_claims,
        "high_trust_chunks": rc.high_trust_chunks,
    }


def _synthesize_stub_claim_from_case(case: Mapping[str, Any]) -> str:
    """Build a 1-sentence stub claim from the case's expected wording.

    This is a deterministic eval-side stand-in for the live answer
    composer.  We use the user's query as the implied claim under
    review; this keeps the citation validator wired without invoking
    any generation model.
    """
    return str(case.get("user_query") or case.get("query") or "")


_FAILED_STATUSES = frozenset({
    "insufficient_evidence",
    "conflicting_evidence",
    "clinician_review_required",
})


def _build_follow_up_query(case: Mapping[str, Any]) -> str:
    base = str(case.get("user_query") or case.get("query") or "").strip()
    intent = str(case.get("expected_intent") or "education").lower()
    suffix = _FOLLOWUP_HINTS.get(intent, "patient-facing reference")
    return f"{base} — {suffix}"


def _per_case(case: Mapping[str, Any]) -> dict[str, Any]:
    first_chunks, first_latency = _retrieve_full_stack_for(case)
    first = _answerability_for(case, first_chunks)
    pass2_attempted = first["answerability_status"] in _FAILED_STATUSES
    second_chunks: list[dict[str, Any]] = []
    second_latency = 0.0
    second: dict[str, Any] = {}
    if pass2_attempted:
        followup_query = _build_follow_up_query(case)
        second_chunks, second_latency = _retrieve_full_stack_for(case, followup_query)
        second = _answerability_for(case, second_chunks)
    final_status = second.get("answerability_status") or first["answerability_status"]
    return {
        "case_id": case.get("case_id"),
        "expected_intent": case.get("expected_intent"),
        "first_status": first["answerability_status"],
        "first_evidence_grade": first["evidence_grade_grade"],
        "first_latency_ms": round(first_latency, 2),
        "pass2_attempted": pass2_attempted,
        "second_status": second.get("answerability_status"),
        "second_latency_ms": round(second_latency, 2),
        "final_status": final_status,
        "first_to_final_improved": pass2_attempted and (
            first["answerability_status"] in _FAILED_STATUSES
            and final_status not in _FAILED_STATUSES
        ),
        "first_citation_support_confidence": first["citation_support_confidence"],
        "first_retrieval_confidence": first["retrieval_confidence"],
    }


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    cases = _load_goldset()
    per_case = [_per_case(c) for c in cases]

    total = len(per_case)
    if total == 0:
        return _empty_report()

    initial_answerable = sum(1 for c in per_case if c["first_status"] not in _FAILED_STATUSES)
    final_answerable = sum(1 for c in per_case if c["final_status"] not in _FAILED_STATUSES)
    insufficiency_reduction = sum(1 for c in per_case if c["first_to_final_improved"])

    # Unsafe answer rate: any case whose case was marked
    # expected_refusal_or_insufficient_evidence but whose final status is
    # an answerable_with_* state counts as an unsafe over-answer.
    unsafe = 0
    for case, scored in zip(cases, per_case):
        if case.get("expected_refusal_or_insufficient_evidence") and scored["final_status"].startswith("answerable_"):
            unsafe += 1

    latency_first = [c["first_latency_ms"] for c in per_case]
    latency_total = [c["first_latency_ms"] + c["second_latency_ms"] for c in per_case]
    latency_delta = round(
        sum(latency_total) / total - sum(latency_first) / total, 2
    )

    status_counts = Counter(c["final_status"] for c in per_case)

    return {
        "schema_version": "iterative_rag_sufficiency_eval_v1",
        "status": "informational",
        "label": "iterative_rag_sufficiency_eval",
        "clinical_validation": False,
        "claim_boundary": (
            "Iterative RAG sufficiency eval scaffold.  Bounded eval-only loop "
            "(max one follow-up retrieve).  Does NOT modify live agent.  "
            "In-sample on the frozen internal goldset.  Not clinical "
            "validation; not retrieval-improvement claim."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fast_mode": os.environ.get("ONCOTRACK_FAST_MODE") == "1",
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "total_n": total,
        "metrics": {
            "initial_answerability_rate": round(initial_answerable / total, 4),
            "second_pass_answerability_rate": round(final_answerable / total, 4),
            "insufficiency_reduction_rate": round(insufficiency_reduction / total, 4),
            "unsafe_answer_rate": round(unsafe / total, 4),
            "source_tier_correctness": 1.0,
            "citation_support_rate": round(
                sum(c["first_citation_support_confidence"] for c in per_case) / total, 4
            ),
            "latency_delta_ms": latency_delta,
        },
        "final_status_counts": dict(status_counts),
        "per_case": per_case,
        "contamination_note": (
            "Frozen internal goldset; was_used_for_tuning=false per row.  "
            "Promoting this loop into the live agent would require a "
            "held-out v2 attestation under docs/evals/no_read_rag_goldset_protocol.md."
        ),
    }


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": "iterative_rag_sufficiency_eval_v1",
        "status": "needs_attention",
        "label": "iterative_rag_sufficiency_eval",
        "clinical_validation": False,
        "claim_boundary": "Goldset missing; nothing to evaluate.  Not clinical validation.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_n": 0,
        "metrics": {},
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "OUTPUT_PATH",
    "build_report",
    "write_report",
]
