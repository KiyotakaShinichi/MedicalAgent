"""Canonical RAG effectiveness-versus-governance trade-off report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
HOLDOUT_PATH = Path("Data/evals/rag/latest_rag_holdout_baseline_comparison.json")
OUTPUT_PATH = Path("Data/evals/rag/latest_rag_governance_tradeoff.json")


def build_rag_governance_tradeoff() -> dict[str, Any]:
    comparison = _read(BASELINE_PATH)
    holdout = _read(HOLDOUT_PATH)
    configurations = comparison.get("configurations") or {}
    bm25 = ((configurations.get("bm25_only") or {}).get("summary") or {})
    full = ((configurations.get("hybrid_rrf_query_rewrite_parent_child_source_tier") or {}).get("summary") or {})
    best = ((configurations.get("hybrid_rrf_query_rewrite") or {}).get("summary") or {})

    recall_delta = _delta(full.get("recall_at_10"), bm25.get("recall_at_10"))
    tier_delta = _delta(full.get("source_tier_correctness"), bm25.get("source_tier_correctness"))
    latency_ratio = _ratio(full.get("latency_p95_ms"), bm25.get("latency_p95_ms"))
    external_completed = holdout.get("completed") is True and holdout.get("external_author_eval_completed") is True
    policy_constraints = {
        "source_tier_correctness": _constraint(full.get("source_tier_correctness"), minimum=1.0),
        "refusal_correctness": _constraint(full.get("refusal_correctness"), minimum=1.0),
        "unsupported_context_rate": _constraint(
            full.get("unsupported_context_rate"),
            maximum=0.20,
        ),
    }
    policy_constraints_pass = all(
        item["passed"] for item in policy_constraints.values()
    )

    return {
        "schema_version": "rag_governance_tradeoff_v2_2026_07",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if policy_constraints_pass else "needs_attention",
        "status_basis": (
            "acceptable_for_internal_policy_constrained_use"
            if policy_constraints_pass
            else "patient_facing_governance_constraints_not_met"
        ),
        "internal_goldset_n": comparison.get("total_n"),
        "internal_goldset_only": True,
        "bm25": _core(bm25),
        "best_ungoverned_hybrid": _core(best),
        "full_source_governed_stack": _core(full),
        "policy_constrained_selection": {
            "selected_configuration": "hybrid_rrf_query_rewrite_parent_child_source_tier",
            "selected_for_raw_retrieval_superiority": False,
            "selected_for_patient_facing_source_governance": True,
            "constraints": policy_constraints,
            "all_constraints_pass": policy_constraints_pass,
            "scope": "internal engineering prototype only",
        },
        "tradeoffs": {
            "full_minus_bm25_recall_at_10": recall_delta,
            "full_minus_bm25_source_tier_correctness": tier_delta,
            "full_vs_bm25_latency_p95_ratio": latency_ratio,
            "raw_retrieval_improvement_proven": bool(recall_delta is not None and recall_delta > 0),
            "governance_improvement_observed_internally": bool(tier_delta is not None and tier_delta > 0),
        },
        "raw_retrieval_superiority_status": "not_proven",
        "governance_control_status": (
            "passed_internal_policy_checks"
            if policy_constraints_pass
            else "needs_attention"
        ),
        "residual_limitations": [
            "BM25 has higher Recall@10 on the internal goldset.",
            "The source-governed stack has higher p95 latency than BM25.",
            "Citation precision and unsupported-context rate still need improvement.",
            "No completed external no-read retrieval holdout exists.",
        ],
        "external_holdout": {
            "completed": external_completed,
            "status": holdout.get("status", "missing"),
            "reason": holdout.get("reason"),
        },
        "decision": (
            "Retain the source-governed configuration for its allowed-use and source-tier controls, "
            "while reporting that raw retrieval superiority over BM25 is not proven. Do not promote "
            "the experimental citation pruner."
        ),
        "improvement_proven_vs_bm25": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "This is internal engineering evidence on a frozen, internally authored goldset. It is not "
            "clinical validation, an external generalisation result, or proof of patient benefit."
        ),
    }


def write_rag_governance_tradeoff(path: str | Path = OUTPUT_PATH) -> dict[str, Any]:
    payload = build_rag_governance_tradeoff()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _core(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary.get(key)
        for key in (
            "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "citation_precision",
            "claim_support_rate", "unsupported_context_rate", "refusal_correctness",
            "source_tier_correctness", "latency_p50_ms", "latency_p95_ms",
        )
    }


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _ratio(left: Any, right: Any) -> float | None:
    if left is None or right in {None, 0}:
        return None
    return round(float(left) / float(right), 4)


def _constraint(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    numeric = float(value) if value is not None else None
    passed = numeric is not None
    if passed and minimum is not None:
        passed = numeric >= minimum
    if passed and maximum is not None:
        passed = numeric <= maximum
    return {
        "value": numeric,
        "minimum": minimum,
        "maximum": maximum,
        "passed": bool(passed),
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


__all__ = ["build_rag_governance_tradeoff", "write_rag_governance_tradeoff"]
