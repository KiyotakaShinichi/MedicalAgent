"""Post-hoc route-aware retrieval-policy diagnostic.

This module composes existing per-case baseline results.  It does not rerun
retrieval, alter the live patient route, or claim that the policy was
pre-registered.  The policy is intentionally conservative: safety-sensitive
and medical-boundary routes use the full source-governed stack, while ordinary
education and portal-help routes use the simpler BM25 baseline.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INPUT_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
OUTPUT_PATH = Path("Data/evals/rag/latest_route_aware_rag_policy_eval.json")

BM25 = "bm25_only"
FULL_GOVERNED = "hybrid_rrf_query_rewrite_parent_child_source_tier"
SIMPLE_ROUTE_INTENTS = {"education", "portal_help"}

METRIC_FIELDS = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "citation_precision",
    "refusal_correct",
    "source_tier_correct",
)


def configuration_for_intent(intent: str) -> str:
    return BM25 if str(intent or "").strip().lower() in SIMPLE_ROUTE_INTENTS else FULL_GOVERNED


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool))]
    return round(statistics.fmean(values), 4) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 2)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    return {
        "case_count": len(rows),
        **{field: _mean(rows, field) for field in METRIC_FIELDS},
        "claim_support_rate": _mean(rows, "claim_supported"),
        "unsupported_context_rate": _mean(rows, "unsupported_context"),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "failure_count": sum(1 for row in rows if row.get("failure_reasons")),
    }


def _baseline_summary(payload: dict[str, Any], configuration: str) -> dict[str, Any]:
    cases = list(payload["configurations"][configuration]["cases"])
    return _summary(cases)


def build_report(input_path: Path = INPUT_PATH) -> dict[str, Any]:
    if not input_path.exists():
        return {
            "schema_version": "route_aware_rag_policy_eval_v1",
            "status": "needs_attention",
            "clinical_validation": False,
            "reason": f"missing input artifact: {input_path.as_posix()}",
        }

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    configurations = payload.get("configurations") or {}
    if BM25 not in configurations or FULL_GOVERNED not in configurations:
        raise ValueError("Required BM25 and full source-governed configurations are missing")

    by_configuration = {
        config_id: {row["case_id"]: row for row in config.get("cases") or []}
        for config_id, config in configurations.items()
    }
    case_ids = list(by_configuration[BM25])
    selected_rows: list[dict[str, Any]] = []
    route_counts: dict[str, int] = {BM25: 0, FULL_GOVERNED: 0}
    for case_id in case_ids:
        reference = by_configuration[BM25][case_id]
        selected_configuration = configuration_for_intent(reference.get("expected_intent") or "")
        selected = dict(by_configuration[selected_configuration][case_id])
        selected["selected_configuration"] = selected_configuration
        selected["selection_reason"] = (
            "simple low-risk route" if selected_configuration == BM25
            else "safety-sensitive or medical-boundary route"
        )
        route_counts[selected_configuration] += 1
        selected_rows.append(selected)

    route_summary = _summary(selected_rows)
    bm25_summary = _baseline_summary(payload, BM25)
    full_summary = _baseline_summary(payload, FULL_GOVERNED)
    failures = [
        {
            "case_id": row["case_id"],
            "intent": row.get("expected_intent"),
            "selected_configuration": row["selected_configuration"],
            "failure_reasons": row.get("failure_reasons") or [],
            "retrieved_source_ids": row.get("retrieved_source_ids") or [],
        }
        for row in selected_rows
        if row.get("failure_reasons")
    ]

    return {
        "schema_version": "route_aware_rag_policy_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "informational",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "external_author_eval_completed": False,
        "internal_vs_external_authored": "internal_post_hoc",
        "was_used_for_tuning": True,
        "policy_pre_registered_before_goldset_inspection": False,
        "live_patient_route_changed": False,
        "input_artifact": input_path.as_posix(),
        "input_goldset_path": payload.get("goldset_path"),
        "input_goldset_total_n": payload.get("total_n"),
        "policy": {
            "education_and_portal_help": BM25,
            "safety_sensitive_and_medical_boundary": FULL_GOVERNED,
            "route_counts": route_counts,
        },
        "route_aware_summary": route_summary,
        "bm25_summary": bm25_summary,
        "full_source_governed_summary": full_summary,
        "deltas": {
            "recall_at_10_vs_bm25": round(route_summary["recall_at_10"] - bm25_summary["recall_at_10"], 4),
            "recall_at_10_vs_full_governed": round(route_summary["recall_at_10"] - full_summary["recall_at_10"], 4),
            "source_tier_correctness_vs_bm25": round(
                route_summary["source_tier_correct"] - bm25_summary["source_tier_correct"], 4
            ),
            "latency_p95_ms_vs_full_governed": round(
                route_summary["latency_p95_ms"] - full_summary["latency_p95_ms"], 2
            ),
        },
        "failure_examples": failures[:12],
        "promotion_decision": "hold_for_external_holdout",
        "claim_boundary": (
            "Post-hoc internal engineering diagnostic only. The policy was created after the internal "
            "goldset and baseline results were available, so it is tuning-used and cannot prove "
            "generalisation, retrieval superiority, clinical validity, or production healthcare readiness."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "BM25",
    "FULL_GOVERNED",
    "OUTPUT_PATH",
    "SIMPLE_ROUTE_INTENTS",
    "build_report",
    "configuration_for_intent",
    "write_report",
]
