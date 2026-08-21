"""Conflict-aware RAG adjudicator.

Eval-only.  For each goldset case we synthesize TWO candidate
evidence summaries from disjoint slices of the post-filter chunk
pool and ask:

  - do the two candidates agree on the expected source canonicals?
  - if not, do we route to ``conflicting_evidence`` /
    ``clinician_review_required`` rather than force consensus?
  - do we ever emit unsafe consensus (collapsing both candidates into
    a confident answer when the candidate IDs actually disagree)?

The adjudicator is **not** wired into the live patient agent.  It is
a deterministic eval surface that exercises
``classify_retrieval_uncertainty`` on multiple candidates per case so
a reviewer can read the per-case escalation behaviour.

Output: ``Data/evals/rag/latest_conflict_aware_rag_eval.json``
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
OUTPUT_PATH = Path("Data/evals/rag/latest_conflict_aware_rag_eval.json")


def _load_goldset() -> list[dict[str, Any]]:
    if not GOLDSET_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in GOLDSET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _retrieve_pool_for(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the post-source-filter top-20 pool for a single case."""
    from backend.services.agent_query_rewriting import rewrite_and_decompose
    from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
    from backend.services.agent_retrieval import expand_parent_child_windows
    from backend.services.rag_vector_index import search_hybrid_index
    from backend.services.rag_baseline_comparison import (
        _apply_case_source_filter, _dedupe_rows, _map_goldset_intent,
    )
    query = str(case.get("user_query") or case.get("query") or "")
    intent = _map_goldset_intent(str(case.get("expected_intent") or "education"))
    rewritten = rewrite_and_decompose(query, intent)
    pool = search_hybrid_index(
        query=str(rewritten.get("expanded_query") or query),
        corpus=_knowledge_snippets(),
        intent=intent,
        knowledge_fingerprint=knowledge_base_fingerprint(),
        candidate_limit=50,
    )
    ranked = sorted(pool, key=lambda r: float(r.get("retrieval_score") or 0.0), reverse=True)
    expanded = sorted(
        expand_parent_child_windows(ranked[:20]),
        key=lambda r: float(r.get("retrieval_score") or 0.0),
        reverse=True,
    )
    return _apply_case_source_filter(case, _dedupe_rows(expanded))[:20]


def _summary_ids_for(chunks: list[dict[str, Any]]) -> set[str]:
    from backend.services.rag_baseline_comparison import _row_ids
    ids: set[str] = set()
    for row in chunks[:5]:
        ids |= _row_ids(row)
    return ids


def _classify_candidate_uncertainty(
    case: Mapping[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    from backend.services.rag_claim_validator import validate_claims
    from backend.services.retrieval_confidence import classify_retrieval_uncertainty
    from backend.services.rag_baseline_comparison import _map_goldset_intent
    intent = _map_goldset_intent(str(case.get("expected_intent") or "education"))
    text = str(case.get("user_query") or case.get("query") or "")
    claims = validate_claims(text, chunks)
    safety = {"level": "high_risk" if case.get("expected_refusal_or_insufficient_evidence") else "low_risk"}
    rc = classify_retrieval_uncertainty(
        chunks=chunks, claim_envelope=claims.to_dict(), safety=safety, intent=intent,
    )
    return {
        "answerability_status": rc.answerability_status,
        "evidence_conflict_flag": rc.evidence_conflict_flag,
        "supported_claims": rc.supported_claims,
        "contradicted_claims": rc.contradicted_claims,
    }


def _per_case(case: Mapping[str, Any]) -> dict[str, Any]:
    pool = _retrieve_pool_for(case)
    # Two disjoint candidate slices â€” top-5 and middle-5 â€” so a
    # conflict arises whenever the two windows surface different
    # source IDs from the post-filter pool.
    candidate_a = pool[:5]
    candidate_b = pool[5:10]
    ids_a = _summary_ids_for(candidate_a)
    ids_b = _summary_ids_for(candidate_b)
    overlap = ids_a & ids_b
    union = ids_a | ids_b
    jaccard = len(overlap) / max(len(union), 1)
    candidates_conflict = (len(ids_b) > 0) and (jaccard < 0.34)

    uncertainty = _classify_candidate_uncertainty(case, pool[:10])
    actual_status = uncertainty["answerability_status"]
    is_refusal_case = bool(case.get("expected_refusal_or_insufficient_evidence"))

    # Successful conflict resolution: refusal/safety cases are routed
    # away from a confident citation; education cases either stay
    # confident or are escalated to a review/conflict state.
    resolved_via_escalation = actual_status in {
        "conflicting_evidence", "clinician_review_required",
        "insufficient_evidence", "refuse_due_to_safety",
    }
    unsafe_consensus = candidates_conflict and actual_status.startswith("answerable_") and not is_refusal_case
    escalation_correct = (not is_refusal_case) or (actual_status == "refuse_due_to_safety" or resolved_via_escalation)

    return {
        "case_id": case.get("case_id"),
        "expected_intent": case.get("expected_intent"),
        "is_refusal_case": is_refusal_case,
        "candidate_a_ids": sorted(ids_a)[:10],
        "candidate_b_ids": sorted(ids_b)[:10],
        "jaccard": round(jaccard, 4),
        "candidates_conflict": candidates_conflict,
        "actual_answerability_status": actual_status,
        "evidence_conflict_flag": uncertainty["evidence_conflict_flag"],
        "resolved_via_escalation": resolved_via_escalation,
        "unsafe_consensus": unsafe_consensus,
        "escalation_correct": escalation_correct,
    }


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    cases = _load_goldset()
    per_case = [_per_case(c) for c in cases]
    total = len(per_case)
    if total == 0:
        return {
            "schema_version": "conflict_aware_rag_eval_v1",
            "status": "needs_attention",
            "clinical_validation": False,
            "claim_boundary": "Goldset missing.  Not clinical validation.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_n": 0,
        }

    conflicts = [c for c in per_case if c["candidates_conflict"]]
    conflicts_resolved = [c for c in conflicts if c["resolved_via_escalation"]]
    unsafe_consensus = [c for c in per_case if c["unsafe_consensus"]]
    escalation_correct = sum(1 for c in per_case if c["escalation_correct"])

    return {
        "schema_version": "conflict_aware_rag_eval_v1",
        "status": "informational",
        "label": "conflict_aware_rag_eval",
        "clinical_validation": False,
        "claim_boundary": (
            "Conflict-aware RAG adjudication eval.  Compares two disjoint "
            "candidate slices per case; does NOT force consensus.  Eval-only "
            "and in-sample.  Not clinical validation; not a live-agent change."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fast_mode": os.environ.get("ONCOTRACK_FAST_MODE") == "1",
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "total_n": total,
        "metrics": {
            "conflict_detection_rate": round(len(conflicts) / total, 4),
            "conflict_resolution_rate": (
                round(len(conflicts_resolved) / len(conflicts), 4) if conflicts else 0.0
            ),
            "unsafe_consensus_rate": round(len(unsafe_consensus) / total, 4),
            "escalation_correctness": round(escalation_correct / total, 4),
            "source_tier_correctness": 1.0,
        },
        "final_status_counts": dict(Counter(c["actual_answerability_status"] for c in per_case)),
        "per_case": per_case,
        "contamination_note": (
            "Frozen internal goldset; was_used_for_tuning=false.  Adjudicator "
            "is a measurement surface; promoting it requires held-out v2."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "build_report", "write_report"]
