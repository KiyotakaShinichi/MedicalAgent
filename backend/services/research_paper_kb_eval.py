"""Internal research-paper KB provenance and retrieval evaluation.

The cases are derived from the evaluated local corpus.  They are useful for
regression testing source identity, paper discrimination, section retrieval,
and safety-boundary routing, but they are not an independent holdout and do
not evaluate clinical correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from backend.services.agent_intent_router import route_intent
from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
from backend.services.agent_safety import safety_scope_check
from backend.services.kb_source_governance import build_kb_source_governance
from backend.services.rag_baseline_comparison import (
    _apply_case_source_filter,
    _dedupe_rows,
    _retrieve_for_config,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
KB_PATH = ROOT_DIR / "Data/rag_knowledge_base_chunks.json"
MANIFEST_PATH = ROOT_DIR / "KnowledgeBase/raw/research_papers/research_papers_manifest.json"
CASES_PATH = ROOT_DIR / "Data/evals/rag/research_paper_grounding_cases.jsonl"
AUDIT_PATH = ROOT_DIR / "Data/evals/rag/latest_research_paper_kb_audit.json"
EVAL_PATH = ROOT_DIR / "Data/evals/rag/latest_research_paper_retrieval_eval.json"
FAILURES_PATH = ROOT_DIR / "Data/evals/rag/latest_research_paper_retrieval_failures.json"

CONFIG_IDS = (
    "bm25_only",
    "faiss_dense_only",
    "hybrid_rrf",
    "hybrid_rrf_query_rewrite",
    "hybrid_rrf_query_rewrite_parent_child",
    "hybrid_rrf_query_rewrite_parent_child_source_tier",
)
FULL_STACK_ID = "hybrid_rrf_query_rewrite_parent_child_source_tier"
SAFE_BOUNDARY_ROUTES = {
    "safety_boundary",
    "treatment_decision_boundary",
    "genetic_counselor_review",
    "urgent_safety",
}
PMC_PATTERN = re.compile(r"\bPMC\d{5,}\b", flags=re.IGNORECASE)


def run_research_paper_kb_eval(
    *,
    kb_path: Path | str = KB_PATH,
    manifest_path: Path | str = MANIFEST_PATH,
    cases_path: Path | str = CASES_PATH,
    audit_path: Path | str = AUDIT_PATH,
    eval_path: Path | str = EVAL_PATH,
    failures_path: Path | str = FAILURES_PATH,
) -> dict[str, Any]:
    kb_path = Path(kb_path)
    manifest_path = Path(manifest_path)
    cases_path = Path(cases_path)
    build_kb_source_governance(kb_chunks_path=str(kb_path))

    raw_chunks = _load_json(kb_path).get("chunks") or []
    manifest = _load_json(manifest_path)
    cases = _load_cases(cases_path)
    audit = build_research_paper_kb_audit(raw_chunks, manifest, cases)
    _write_json(Path(audit_path), audit)

    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    configurations: dict[str, Any] = {}
    all_failures: list[dict[str, Any]] = []
    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]] = {}
    rewrite_cache: dict[tuple[str, str], tuple[str, float]] = {}

    for config_id in CONFIG_IDS:
        report = _evaluate_configuration(
            config_id,
            cases,
            corpus,
            fingerprint,
            search_cache,
            rewrite_cache,
        )
        configurations[config_id] = report
        all_failures.extend(
            {"configuration": config_id, **row}
            for row in report["cases"]
            if row["failure_reasons"]
        )

    boundary = _evaluate_boundary_routes(cases)
    bm25_rows = configurations["bm25_only"]["cases"]
    full_rows = configurations[FULL_STACK_ID]["cases"]
    paired = _paired_recall_comparison(bm25_rows, full_rows)
    full = configurations[FULL_STACK_ID]["summary"]
    improvement_proven = bool(
        paired["recall_at_10_delta"] > 0
        and paired["bootstrap_ci95"][0] > 0
        and paired["exact_p_value"] < 0.05
        and full["source_tier_correctness"] == 1.0
    )
    status = "acceptable_internal_diagnostic"
    if (
        audit["status"] == "needs_attention"
        or full["recall_at_10"] < 0.8
        or full["no_evidence_false_paper_attribution_rate"] > 0.0
        or boundary["correctness"] < 1.0
    ):
        status = "needs_attention"

    payload = {
        "schema_version": "research_paper_retrieval_eval_v1",
        "generated_at": _now(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "independent_holdout": False,
        "derived_from_evaluated_kb": True,
        "was_used_for_tuning": False,
        "live_patient_route_changed": False,
        "case_file": _relative(cases_path),
        "case_file_sha256": _sha256(cases_path),
        "case_count": len(cases),
        "paper_count": audit["summary"]["manifest_paper_count"],
        "configurations": configurations,
        "pre_retrieval_boundary": boundary,
        "paired_full_stack_vs_bm25": paired,
        "summary": {
            "full_stack_recall_at_5": full["recall_at_5"],
            "full_stack_recall_at_10": full["recall_at_10"],
            "full_stack_mrr": full["mrr"],
            "full_stack_top1_paper_accuracy": full["top1_paper_accuracy"],
            "full_stack_section_hit_rate": full["section_hit_rate"],
            "full_stack_taglish_recall_at_10": full["taglish_recall_at_10"],
            "full_stack_no_evidence_false_paper_attribution_rate": full[
                "no_evidence_false_paper_attribution_rate"
            ],
            "full_stack_provenance_completeness": full["provenance_completeness"],
            "full_stack_source_tier_correctness": full["source_tier_correctness"],
            "boundary_route_correctness": boundary["correctness"],
            "paper_retrieval_improvement_proven_vs_bm25": improvement_proven,
        },
        "limitations": [
            "Cases were authored from the same local papers being evaluated.",
            "This measures retrieval attribution and provenance, not whether a medical claim is clinically correct.",
            "No generated patient answer, clinician review, or external author is part of this suite.",
            "The corpus is concentrated in MRI-response and chemotherapy/neutropenia topics.",
        ],
        "claim_boundary": (
            "Internal KB-derived engineering regression only. It is not clinical validation, "
            "an independent literature review, patient-benefit evidence, or production healthcare readiness."
        ),
    }
    failure_payload = {
        "schema_version": "research_paper_retrieval_failures_v1",
        "generated_at": payload["generated_at"],
        "status": "needs_attention" if all_failures else "acceptable_internal_diagnostic",
        "clinical_validation": False,
        "independent_holdout": False,
        "failure_count": len(all_failures),
        "failures": all_failures,
        "claim_boundary": payload["claim_boundary"],
    }
    _write_json(Path(eval_path), payload)
    _write_json(Path(failures_path), failure_payload)
    return {"audit": audit, "evaluation": payload, "failures": failure_payload}


def build_research_paper_kb_audit(
    chunks: list[dict[str, Any]],
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    items = manifest.get("items") or []
    manifest_pmcids = {str(item.get("pmcid") or "").upper() for item in items if item.get("pmcid")}
    paper_chunks = [row for row in chunks if _is_owned_research_source(row)]
    by_pmcid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paper_chunks:
        by_pmcid[str(row.get("pmcid") or "").upper()].append(row)

    false_identity_rows = [
        row
        for row in chunks
        if row.get("pmcid") and not _is_owned_research_source(row)
    ]
    duplicate_manifest_pmcids = [
        pmcid for pmcid, count in Counter(str(item.get("pmcid") or "").upper() for item in items).items()
        if pmcid and count > 1
    ]
    covered = manifest_pmcids & set(by_pmcid)
    source_rows = []
    for item in items:
        pmcid = str(item.get("pmcid") or "").upper()
        rows = by_pmcid.get(pmcid, [])
        source_rows.append({
            "pmcid": pmcid,
            "title": item.get("title"),
            "chunk_count": len(rows),
            "sections": dict(Counter(str(row.get("section") or "unknown") for row in rows)),
            "landing_url_https": str(item.get("landing_url") or "").startswith("https://"),
            "source_file_present": bool(rows),
            "publication_date_present": bool(item.get("publication_date") or item.get("published_at")),
            "doi_present": bool(item.get("doi")),
            "runtime_required_metadata_complete": all(
                all(row.get(key) is not None for key in ("pmcid", "source_url", "source_path", "section"))
                for row in rows
            ) if rows else False,
        })

    expected_case_pmcids = {
        str(case.get("expected_pmcid") or "").upper()
        for case in cases
        if case.get("expected_pmcid")
    }
    topic_counts = Counter(str(item.get("topic") or "unknown") for item in items)
    status = "acceptable_narrow_internal_corpus"
    if false_identity_rows or covered != manifest_pmcids or duplicate_manifest_pmcids:
        status = "needs_attention"
    return {
        "schema_version": "research_paper_kb_audit_v1",
        "generated_at": _now(),
        "status": status,
        "clinical_validation": False,
        "independent_literature_review": False,
        "summary": {
            "manifest_paper_count": len(items),
            "manifest_pmcid_count": len(manifest_pmcids),
            "covered_manifest_pmcid_count": len(covered),
            "manifest_to_chunk_coverage": round(len(covered) / max(len(manifest_pmcids), 1), 4),
            "owned_research_chunk_count": len(paper_chunks),
            "false_pmcid_identity_chunk_count": len(false_identity_rows),
            "duplicate_manifest_pmcid_count": len(duplicate_manifest_pmcids),
            "case_expected_paper_coverage": round(
                len(expected_case_pmcids & manifest_pmcids) / max(len(manifest_pmcids), 1), 4
            ),
            "publication_date_completeness": round(
                sum(row["publication_date_present"] for row in source_rows) / max(len(source_rows), 1), 4
            ),
            "doi_completeness": round(
                sum(row["doi_present"] for row in source_rows) / max(len(source_rows), 1), 4
            ),
            "topic_distribution": dict(topic_counts),
        },
        "sources": source_rows,
        "false_identity_examples": [
            {
                "title": row.get("title"),
                "pmcid": row.get("pmcid"),
                "source_path": row.get("source_path"),
            }
            for row in false_identity_rows[:10]
        ],
        "known_gaps": [
            "Publication dates and DOI metadata are not populated in the current manifest.",
            "Nine papers are too narrow for broad breast-cancer education coverage.",
            "The source set is topic-concentrated and was selected internally.",
            "Source inclusion has not been reviewed by a clinician or medical librarian.",
        ],
        "claim_boundary": (
            "This audit checks local source identity and metadata coverage only. It does not certify "
            "clinical authority, systematic-review quality, completeness, or current medical guidance."
        ),
    }


def _evaluate_configuration(
    config_id: str,
    cases: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    fingerprint: str,
    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]],
    rewrite_cache: dict[tuple[str, str], tuple[str, float]],
) -> dict[str, Any]:
    rows = []
    latencies = []
    for case in cases:
        ranked, latency_ms = _retrieve_for_config(
            config_id,
            str(case["query"]),
            "education",
            corpus,
            fingerprint,
            search_cache,
            rewrite_cache,
        )
        ranked = _dedupe_rows(ranked)
        if "source_tier" in config_id:
            ranked = _apply_case_source_filter(
                {
                    "acceptable_source_tiers": case.get("acceptable_source_tiers") or ["T2"],
                    "expected_refusal_or_insufficient_evidence": False,
                },
                ranked,
            )
        top10 = ranked[:10]
        latencies.append(latency_ms)
        rows.append(_score_case(case, top10, latency_ms, source_tier_filtered="source_tier" in config_id))

    positive = [row for row in rows if row["expected_pmcid"]]
    section_rows = [row for row in positive if row["expected_section"]]
    taglish_rows = [row for row in positive if "taglish" in row["style"]]
    no_evidence = [row for row in rows if not row["expected_pmcid"]]
    provenance_denominator = sum(row["matched_relevant_chunk_count"] for row in positive)
    provenance_numerator = sum(row["provenance_complete_relevant_chunk_count"] for row in positive)
    summary = {
        "case_count": len(rows),
        "positive_case_count": len(positive),
        "no_evidence_case_count": len(no_evidence),
        "recall_at_5": _mean(row["recall_at_5"] for row in positive),
        "recall_at_10": _mean(row["recall_at_10"] for row in positive),
        "mrr": _mean(row["mrr"] for row in positive),
        "ndcg_at_10": _mean(row["ndcg_at_10"] for row in positive),
        "top1_paper_accuracy": _mean(row["top1_paper_correct"] for row in positive),
        "section_hit_rate": _mean(row["section_hit"] for row in section_rows),
        "taglish_recall_at_10": _mean(row["recall_at_10"] for row in taglish_rows),
        "no_evidence_false_paper_attribution_rate": _mean(
            row["false_paper_attribution"] for row in no_evidence
        ),
        "provenance_completeness": round(
            provenance_numerator / max(provenance_denominator, 1), 4
        ),
        "source_tier_correctness": _mean(row["source_tier_correct"] for row in rows),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "failure_count": sum(bool(row["failure_reasons"]) for row in rows),
    }
    return {"summary": summary, "cases": rows}


def _score_case(
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    latency_ms: float,
    *,
    source_tier_filtered: bool,
) -> dict[str, Any]:
    expected = str(case.get("expected_pmcid") or "").upper() or None
    expected_section = str(case.get("expected_section") or "").lower() or None
    retrieved_pmcids = [_row_pmcid(row) for row in rows]
    first_rank = next(
        (index for index, pmcid in enumerate(retrieved_pmcids, start=1) if expected and pmcid == expected),
        None,
    )
    matched = [row for row in rows if expected and _row_pmcid(row) == expected]
    section_hit = bool(
        expected_section
        and any(str(row.get("section") or "").lower() == expected_section for row in matched)
    )
    provenance_complete_count = sum(_provenance_complete(row) for row in matched)
    manifest_research_in_top5 = [pmcid for pmcid in retrieved_pmcids[:5] if pmcid]
    false_attribution = not expected and bool(manifest_research_in_top5)
    source_tier_correct = True
    if source_tier_filtered:
        source_tier_correct = all(
            str(row.get("source_tier") or "") == "T2"
            for row in rows
            if _row_pmcid(row)
        )
    reasons = []
    if expected and first_rank is None:
        reasons.append("expected_paper_missing_at_10")
    if expected_section and not section_hit:
        reasons.append("expected_section_missing_at_10")
    if false_attribution:
        reasons.append("research_paper_returned_for_no_evidence_boundary")
    if matched and provenance_complete_count < len(matched):
        reasons.append("relevant_chunk_provenance_incomplete")
    if not source_tier_correct:
        reasons.append("research_source_not_t2_after_governance")
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "category": case["category"],
        "style": case["style"],
        "expected_pmcid": expected,
        "expected_section": expected_section,
        "retrieved_pmcids": retrieved_pmcids,
        "first_relevant_rank": first_rank,
        "recall_at_5": 1.0 if first_rank and first_rank <= 5 else 0.0,
        "recall_at_10": 1.0 if first_rank else 0.0,
        "mrr": round(1.0 / first_rank, 4) if first_rank else 0.0,
        "ndcg_at_10": round(1.0 / math.log2(first_rank + 1), 4) if first_rank else 0.0,
        "top1_paper_correct": bool(first_rank == 1),
        "section_hit": section_hit if expected_section else None,
        "false_paper_attribution": false_attribution,
        "matched_relevant_chunk_count": len(matched),
        "provenance_complete_relevant_chunk_count": provenance_complete_count,
        "source_tier_correct": source_tier_correct,
        "latency_ms": round(latency_ms, 3),
        "failure_reasons": reasons,
    }


def _evaluate_boundary_routes(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in cases:
        if case.get("expected_pmcid"):
            continue
        safety = safety_scope_check(str(case["query"]), [])
        intent = route_intent(str(case["query"]), [], safety)
        correct = intent in SAFE_BOUNDARY_ROUTES
        rows.append({
            "case_id": case["case_id"],
            "intent": intent,
            "safety_level": safety.get("level"),
            "safety_scope": safety.get("scope"),
            "correct": correct,
        })
    return {
        "case_count": len(rows),
        "correctness": _mean(row["correct"] for row in rows),
        "failure_count": sum(not row["correct"] for row in rows),
        "cases": rows,
        "note": "Routing proxy only; generated refusal quality remains covered by live-agent safety evals.",
    }


def _paired_recall_comparison(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in baseline_rows if row["expected_pmcid"]}
    candidate = {row["case_id"]: row for row in candidate_rows if row["expected_pmcid"]}
    case_ids = sorted(set(baseline) & set(candidate))
    deltas = [
        candidate[case_id]["recall_at_10"] - baseline[case_id]["recall_at_10"]
        for case_id in case_ids
    ]
    improved = sum(delta > 0 for delta in deltas)
    regressed = sum(delta < 0 for delta in deltas)
    return {
        "paired_case_count": len(case_ids),
        "recall_at_10_delta": round(_mean(deltas), 6),
        "bootstrap_ci95": _bootstrap_mean_ci(deltas),
        "improved_case_count": improved,
        "regressed_case_count": regressed,
        "exact_p_value": _two_sided_sign_test(improved, regressed),
    }


def _bootstrap_mean_ci(values: list[float], *, samples: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(20260802)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    return [round(means[int(samples * 0.025)], 6), round(means[int(samples * 0.975)], 6)]


def _two_sided_sign_test(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(improved, regressed) + 1)) / (2**discordant)
    return round(min(1.0, 2 * tail), 6)


def _is_owned_research_source(row: dict[str, Any]) -> bool:
    path = str(row.get("source_path") or "").replace("\\", "/").lower()
    return "/research_papers/" in path and bool(row.get("pmcid"))


def _row_pmcid(row: dict[str, Any]) -> str | None:
    direct = str(row.get("pmcid") or "").upper()
    if direct and PMC_PATTERN.fullmatch(direct):
        return direct
    for value in (row.get("source_url"), row.get("source_path")):
        match = PMC_PATTERN.search(str(value or ""))
        if match:
            return match.group(0).upper()
    return None


def _provenance_complete(row: dict[str, Any]) -> bool:
    return all(row.get(key) is not None for key in ("pmcid", "source_url", "source_path", "title", "section"))


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    case_ids = [str(case.get("case_id") or "") for case in cases]
    if not cases or len(case_ids) != len(set(case_ids)) or any(not case_id for case_id in case_ids):
        raise ValueError("Research-paper cases must be non-empty with unique case_id values.")
    if any(case.get("was_used_for_tuning") is not False for case in cases):
        raise ValueError("Research-paper cases must remain was_used_for_tuning=false.")
    return cases


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean(values: Iterable[float | bool | None]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / max(len(numeric), 1), 4)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if percentile == 50:
        return round(median(ordered), 3)
    index = min(len(ordered) - 1, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(ordered[index], 3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["build_research_paper_kb_audit", "run_research_paper_kb_eval"]
