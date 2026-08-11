"""Consolidated, machine-readable attribution for current RAG failures.

This diagnostic consumes existing case-level artifacts. It does not rerun the
agent or change retrieval. Its purpose is to turn several opaque failure lists
into one engineering backlog with explicit stage ownership.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data/evals/rag/latest_rag_failure_attribution_v_next.json"
INPUTS = (
    ROOT / "Data/evals/agentic_tool_use/latest_mixed_query_scale_failures.json",
    ROOT / "Data/evals/rag/latest_rag_baseline_failures.json",
    ROOT / "Data/evals/rag/latest_section_aware_retrieval_failures.json",
    ROOT / "Data/evals/rag/latest_research_paper_retrieval_failures.json",
)
CLAIM_BOUNDARY = (
    "Internal failure attribution over existing engineering evaluations. Counts can overlap because one case "
    "may fail at multiple stages. This is not independent evaluation, clinical validation, or proof of medical correctness."
)


REASON_TO_STAGE = {
    "expected_section_missing_at_10": "section_mismatch",
    "low_citation_precision": "citation_alignment",
    "low_claim_support": "generation_grounding",
    "unsupported_context": "context_assembly",
    "missing_expected_source": "retrieval_miss",
    "low_recall_at_10": "retrieval_miss",
    "source_tier_incorrect": "source_tier_filtering",
    "wrong_refusal": "inappropriate_answer",
    "refusal_incorrect": "inappropriate_answer",
    "false_attribution": "answerability",
    "no_evidence_false_attribution": "answerability",
    "citation_mismatch": "citation_alignment",
    "expected_source_missing_at_10": "retrieval_miss",
    "expected_paper_missing_at_10": "retrieval_miss",
    "expected_paper_missing_at_5": "ranking_failure",
    "expected_section_missing": "section_mismatch",
}


def build_rag_failure_attribution(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    input_paths: Iterable[str | Path] = INPUTS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_inputs: list[str] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if not path.exists():
            missing_inputs.append(_relative(path))
            continue
        payload = _read(path)
        for failure in payload.get("failures") or []:
            if not isinstance(failure, Mapping):
                continue
            reasons = _failure_reasons(failure)
            stages = sorted({_stage_for_reason(reason, failure) for reason in reasons})
            rows.append(_row(path, failure, reasons, stages))

    raw_stage_counts = Counter(stage for row in rows for stage in row["failure_stages"])
    unique_case_stages = {
        (str(row.get("case_id") or f"row:{index}"), stage)
        for index, row in enumerate(rows)
        for stage in row["failure_stages"]
    }
    stage_counts = Counter(stage for _, stage in unique_case_stages)
    reason_counts = Counter(reason for row in rows for reason in row["failure_reasons"])
    total_stage_assignments = sum(stage_counts.values())
    payload = {
        "schema_version": "rag_failure_attribution_v_next_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_internal_diagnostic" if rows else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "independent_holdout": False,
        "source_artifacts": [_relative(Path(path)) for path in input_paths if Path(path).exists()],
        "missing_optional_inputs": missing_inputs,
        "failure_row_count": len(rows),
        "stage_assignment_count": total_stage_assignments,
        "raw_stage_assignment_count": sum(raw_stage_counts.values()),
        "aggregate_by_stage": {
            stage: {
                "count": count,
                "share_of_stage_assignments": round(count / max(total_stage_assignments, 1), 6),
            }
            for stage, count in stage_counts.most_common()
        },
        "aggregate_by_reason": dict(reason_counts.most_common()),
        "raw_aggregate_by_stage": dict(raw_stage_counts.most_common()),
        "failures": rows,
        "engineering_decision": _decision(stage_counts),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _failure_reasons(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("failure_reasons") or row.get("reasons") or []
    if isinstance(raw, str):
        reasons = [raw]
    elif isinstance(raw, list):
        reasons = [str(item) for item in raw if item]
    else:
        reasons = []
    if not reasons:
        if row.get("expected_section") and not row.get("candidate_section_hit_at_10", True):
            reasons.append("expected_section_missing_at_10")
        elif row.get("recall_at_10") == 0 or row.get("paper_hit_at_10") is False:
            reasons.append("expected_source_missing_at_10")
        else:
            reasons.append("unknown_unclassified")
    return sorted(set(reasons))


def _stage_for_reason(reason: str, row: Mapping[str, Any]) -> str:
    normalized = reason.strip().lower().replace(" ", "_")
    if normalized in REASON_TO_STAGE:
        return REASON_TO_STAGE[normalized]
    if "section" in normalized:
        return "section_mismatch"
    if "citation" in normalized:
        return "citation_alignment"
    if "tier" in normalized or "allowed_use" in normalized:
        return "source_tier_filtering"
    if "context" in normalized:
        return "context_assembly"
    if "answerability" in normalized or "false_attribution" in normalized:
        return "answerability"
    if "rank" in normalized or row.get("first_relevant_rank"):
        return "ranking_failure"
    if "retriev" in normalized or "recall" in normalized or "source" in normalized:
        return "retrieval_miss"
    if "support" in normalized or "ground" in normalized:
        return "generation_grounding"
    if "refusal" in normalized:
        return "inappropriate_refusal"
    return "unknown_unclassified"


def _row(path: Path, failure: Mapping[str, Any], reasons: list[str], stages: list[str]) -> dict[str, Any]:
    return {
        "source_artifact": _relative(path),
        "case_id": failure.get("case_id"),
        "query": failure.get("query"),
        "intent": failure.get("expected_intent") or failure.get("intent") or failure.get("category"),
        "expected_source": failure.get("expected_source_ids") or failure.get("expected_pmcid"),
        "retrieved_sources": failure.get("retrieved_source_ids") or failure.get("retrieved_pmcids"),
        "retrieval_scores": failure.get("retrieval_scores"),
        "route": failure.get("route") or failure.get("observed_route"),
        "answerability_state": failure.get("answerability_status"),
        "failure_stages": stages,
        "failure_reasons": reasons,
        "support_score": failure.get("claim_support_rate") or failure.get("support_score"),
        "citation_result": failure.get("citation_precision"),
        "safety_result": failure.get("safety_result"),
        "latency_ms": failure.get("latency_ms"),
        "model_version": failure.get("model_version"),
        "index_version": failure.get("index_version"),
        "dataset_version": failure.get("dataset_version"),
    }


def _decision(counts: Counter[str]) -> dict[str, Any]:
    highest = counts.most_common(1)
    stage = highest[0][0] if highest else "none"
    return {
        "largest_observed_bucket": stage,
        "recommended_owner": {
            "section_mismatch": "ingestion_and_indexing",
            "retrieval_miss": "candidate_generation",
            "ranking_failure": "ranking",
            "citation_alignment": "citation_assembly",
            "context_assembly": "context_selection",
            "source_tier_filtering": "goldset_adjudication_and_governance",
            "answerability": "answerability_classifier",
            "generation_grounding": "generation_and_verification",
        }.get(stage, "evaluation_triage"),
        "promotion_authorized": False,
        "reason": "Attribution identifies engineering ownership; it does not itself prove a fix or authorize promotion.",
    }


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


__all__ = ["build_rag_failure_attribution"]
