"""Non-compensatory Accuracy-Latency-Unit-Cost governance for NLCare."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "config/ai_trinity_policy.json"
DEFAULT_OUTPUT_PATH = ROOT / "Data/evals/governance/latest_ai_trinity_tradeoff.json"
DEFAULT_RAG_PATH = ROOT / "Data/evals/rag/latest_rag_baseline_comparison.json"
DEFAULT_COST_PATH = ROOT / "Data/evals/ops/latest_cost_latency_report.json"
DEFAULT_PROVIDER_PATH = ROOT / "Data/evals/ops/latest_provider_usage_reconciliation.json"
DEFAULT_LATENCY_PATH = ROOT / "Data/evals/ops/latest_route_latency_budget.json"

CURRENT_OPERATING_CONFIGURATION = (
    "hybrid_rrf_query_rewrite_parent_child_source_tier"
)
CLAIM_BOUNDARY = (
    "The AI Trinity is internal engineering governance over Accuracy, Latency, "
    "and Unit Cost. Missing provider usage is not treated as zero cost. Passing "
    "does not establish clinical validation, patient benefit, audited billing, "
    "a production SLO, or production healthcare readiness."
)


def build_ai_trinity_tradeoff(
    *,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    rag_path: str | Path = DEFAULT_RAG_PATH,
    cost_path: str | Path = DEFAULT_COST_PATH,
    provider_path: str | Path = DEFAULT_PROVIDER_PATH,
    latency_path: str | Path = DEFAULT_LATENCY_PATH,
) -> dict[str, Any]:
    policy = _read_json(policy_path)
    rag = _read_json(rag_path)
    cost = _read_json(cost_path)
    provider = _read_json(provider_path)
    latency = _read_json(latency_path)

    configurations = rag.get("configurations") or {}
    summaries = {
        name: value.get("summary") or {}
        for name, value in configurations.items()
        if isinstance(value, dict)
    }
    candidates = _retrieval_candidates(summaries, policy)
    _assign_dominance(candidates)

    current = next(
        (
            row
            for row in candidates
            if row["configuration"] == CURRENT_OPERATING_CONFIGURATION
        ),
        None,
    )
    provider_summary = _provider_evidence(provider, cost, policy)
    latency_axis = _latency_axis(current, latency, policy)
    accuracy_axis = _accuracy_axis(current)
    unit_cost_axis = _unit_cost_axis(provider_summary, cost, policy)
    axes = {
        "accuracy": accuracy_axis,
        "latency": latency_axis,
        "unit_cost": unit_cost_axis,
    }
    decision = _decision(accuracy_axis, latency_axis, unit_cost_axis)
    planning_scenarios = _planning_scenarios(cost, policy)

    return {
        "schema_version": "ai_trinity_tradeoff_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_internal_tradeoff" if decision == "RETAIN" else "needs_attention",
        "decision": decision,
        "promotion_allowed": decision == "PROMOTE",
        "current_operating_configuration": CURRENT_OPERATING_CONFIGURATION,
        "decision_order": policy.get("decision_order") or [],
        "policy_path": _relative(policy_path),
        "source_artifacts": {
            "rag_baseline": _relative(rag_path),
            "cost_latency": _relative(cost_path),
            "provider_reconciliation": _relative(provider_path),
            "route_latency": _relative(latency_path),
        },
        "axes": axes,
        "summary": {
            "accuracy_status": accuracy_axis["status"],
            "latency_status": latency_axis["status"],
            "unit_cost_status": unit_cost_axis["status"],
            "provider_usage_coverage_rate": provider_summary["coverage_rate"],
            "provider_paired_request_count": provider_summary["paired_request_count"],
            "current_accuracy_grounding_score": (
                current.get("accuracy_grounding_score") if current else None
            ),
            "current_retrieval_p95_ms": (
                current.get("latency_p95_ms") if current else None
            ),
            "current_cost_per_safe_supported_answer_usd": unit_cost_axis.get(
                "cost_per_safe_supported_answer_usd"
            ),
            "retrieval_candidate_count": len(candidates),
            "fully_promotion_eligible_candidate_count": sum(
                row["promotion_eligible"] for row in candidates
            ),
            "improvement_proven_vs_bm25": bool(
                (rag.get("summary") or {}).get("improvement_proven_vs_bm25", False)
            ),
        },
        "scenarios": candidates,
        "planning_only_route_cost_scenarios": planning_scenarios,
        "current_policy": {
            "decision": "retain_source_governed_stack_for_governance",
            "reason": (
                "The current source-governed stack preserves source-tier and refusal "
                "correctness, but it is not promoted as an accuracy, latency, or cost winner."
            ),
            "dense_or_complex_retrieval_promoted": False,
            "cost_optimization_promoted": False,
        },
        "next_actions": [
            "Raise citation precision and reduce unsupported context on development cases without weakening source governance.",
            "Capture at least 30 paired provider-usage requests with at least 80% coverage on non-patient synthetic traffic.",
            "Recompute cost per safe, source-governed, claim-supported answer from provider-token-derived pricing rather than treating missing telemetry as zero.",
            "Re-run the same frozen candidate matrix after quality changes and reject any option that buys speed or cost by crossing a safety floor.",
        ],
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "audited_billing": False,
        "production_slo": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_ai_trinity_tradeoff(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_ai_trinity_tradeoff(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _retrieval_candidates(
    summaries: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    weights = policy.get("retrieval_quality_weights") or {}
    hard = policy.get("hard_floors") or {}
    quality = policy.get("accuracy_floors") or {}
    latency = policy.get("latency_budgets_ms") or {}
    p50_values = [
        _number(summary.get("latency_p50_ms"))
        for summary in summaries.values()
        if _number(summary.get("latency_p50_ms")) is not None
    ]
    fastest = min(p50_values) if p50_values else None
    rows = []
    for name, summary in summaries.items():
        score = round(
            sum(
                (_number(summary.get(metric)) or 0.0) * float(weight)
                for metric, weight in weights.items()
            ),
            6,
        )
        refusal = _number(summary.get("refusal_correctness")) or 0.0
        source_tier = _number(summary.get("source_tier_correctness")) or 0.0
        unsupported = _number(summary.get("unsupported_context_rate"))
        p50 = _number(summary.get("latency_p50_ms"))
        p95 = _number(summary.get("latency_p95_ms"))
        governance_pass = (
            refusal >= float(hard.get("refusal_correctness_min", 1.0))
            and source_tier >= float(hard.get("source_tier_correctness_min", 1.0))
        )
        quality_pass = (
            (_number(summary.get("recall_at_10")) or 0.0)
            >= float(quality.get("recall_at_10_min", 0.8))
            and (_number(summary.get("citation_precision")) or 0.0)
            >= float(quality.get("citation_precision_min", 0.6))
            and (_number(summary.get("claim_support_rate")) or 0.0)
            >= float(quality.get("claim_support_rate_min", 0.9))
            and unsupported is not None
            and unsupported <= float(quality.get("unsupported_context_rate_max", 0.1))
        )
        latency_pass = (
            p95 is not None
            and p95 <= float(latency.get("retrieval_p95_max", 500.0))
        )
        rows.append(
            {
                "configuration": name,
                "accuracy_grounding_score": score,
                "recall_at_10": _number(summary.get("recall_at_10")),
                "mrr": _number(summary.get("mrr")),
                "ndcg_at_10": _number(summary.get("ndcg_at_10")),
                "citation_precision": _number(summary.get("citation_precision")),
                "claim_support_rate": _number(summary.get("claim_support_rate")),
                "unsupported_context_rate": unsupported,
                "refusal_correctness": refusal,
                "source_tier_correctness": source_tier,
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "relative_local_compute_index": (
                    round(p50 / fastest, 4)
                    if p50 is not None and fastest not in {None, 0}
                    else None
                ),
                "governance_floor_pass": governance_pass,
                "accuracy_floor_pass": quality_pass,
                "latency_budget_pass": latency_pass,
                "unit_cost_evidence_complete": False,
                "unit_cost_usd": None,
                "promotion_eligible": False,
                "promotion_blockers": _candidate_blockers(
                    governance_pass, quality_pass, latency_pass
                ),
                "dominated_by": [],
                "two_axis_pareto_frontier": False,
                "governance_eligible_pareto_frontier": False,
            }
        )
    return rows


def _candidate_blockers(
    governance_pass: bool, quality_pass: bool, latency_pass: bool
) -> list[str]:
    blockers = []
    if not governance_pass:
        blockers.append("safety_or_source_governance_floor")
    if not quality_pass:
        blockers.append("accuracy_or_grounding_floor")
    if not latency_pass:
        blockers.append("latency_budget")
    blockers.append("provider_unit_cost_evidence")
    return blockers


def _assign_dominance(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        dominators = []
        for other in rows:
            if other is row:
                continue
            if _dominates(other, row):
                dominators.append(other["configuration"])
        row["dominated_by"] = dominators
        row["two_axis_pareto_frontier"] = not dominators

        governed_dominators = [
            other["configuration"]
            for other in rows
            if other is not row
            and other["governance_floor_pass"]
            and row["governance_floor_pass"]
            and _dominates(other, row)
        ]
        row["governance_eligible_pareto_frontier"] = (
            row["governance_floor_pass"] and not governed_dominators
        )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_quality = _number(left.get("accuracy_grounding_score"))
    right_quality = _number(right.get("accuracy_grounding_score"))
    left_latency = _number(left.get("latency_p95_ms"))
    right_latency = _number(right.get("latency_p95_ms"))
    left_compute = _number(left.get("relative_local_compute_index"))
    right_compute = _number(right.get("relative_local_compute_index"))
    values = (
        left_quality,
        right_quality,
        left_latency,
        right_latency,
        left_compute,
        right_compute,
    )
    if any(value is None for value in values):
        return False
    no_worse = (
        left_quality >= right_quality
        and left_latency <= right_latency
        and left_compute <= right_compute
    )
    strictly_better = (
        left_quality > right_quality
        or left_latency < right_latency
        or left_compute < right_compute
    )
    return no_worse and strictly_better


def _provider_evidence(
    provider: dict[str, Any], cost: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    requirements = policy.get("unit_cost_evidence") or {}
    coverage = _number(provider.get("actual_usage_coverage_rate")) or 0.0
    paired = int(provider.get("paired_request_count") or 0)
    complete = (
        provider.get("completed") is True
        and coverage >= float(requirements.get("provider_usage_coverage_min", 0.8))
        and paired >= int(requirements.get("paired_request_count_min", 30))
    )
    return {
        "status": provider.get("status") or "missing",
        "coverage_rate": coverage,
        "paired_request_count": paired,
        "evidence_complete": complete,
        "estimated_tokens_are_billing_truth": False,
        "cost_report_request_count": int((cost.get("summary") or {}).get("request_count") or 0),
    }


def _accuracy_axis(current: dict[str, Any] | None) -> dict[str, Any]:
    if not current:
        return {"status": "missing", "reason": "Current retrieval configuration is absent."}
    passed = current["governance_floor_pass"] and current["accuracy_floor_pass"]
    return {
        "status": "pass" if passed else "needs_attention",
        "accuracy_grounding_score": current["accuracy_grounding_score"],
        "governance_floor_pass": current["governance_floor_pass"],
        "accuracy_floor_pass": current["accuracy_floor_pass"],
        "recall_at_10": current["recall_at_10"],
        "citation_precision": current["citation_precision"],
        "claim_support_rate": current["claim_support_rate"],
        "unsupported_context_rate": current["unsupported_context_rate"],
        "source_tier_correctness": current["source_tier_correctness"],
        "refusal_correctness": current["refusal_correctness"],
    }


def _latency_axis(
    current: dict[str, Any] | None,
    route_latency: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    budgets = policy.get("latency_budgets_ms") or {}
    retrieval_p95 = current.get("latency_p95_ms") if current else None
    route_summary = route_latency.get("summary") or {}
    normal_p95 = _number(route_summary.get("highest_observed_p95_ms"))
    retrieval_pass = (
        retrieval_p95 is not None
        and retrieval_p95 <= float(budgets.get("retrieval_p95_max", 500.0))
    )
    normal_pass = (
        normal_p95 is not None
        and normal_p95 <= float(budgets.get("normal_route_p95_max", 8000.0))
    )
    return {
        "status": "pass" if retrieval_pass and normal_pass else "needs_attention",
        "retrieval_p95_ms": retrieval_p95,
        "retrieval_budget_ms": budgets.get("retrieval_p95_max"),
        "normal_route_highest_p95_ms": normal_p95,
        "normal_route_budget_ms": budgets.get("normal_route_p95_max"),
        "retrieval_budget_pass": retrieval_pass,
        "normal_route_budget_pass": normal_pass,
        "production_slo": False,
    }


def _unit_cost_axis(
    provider: dict[str, Any], cost: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    requirements = policy.get("unit_cost_evidence") or {}
    requests = [row for row in cost.get("requests") or [] if isinstance(row, dict)]
    eligible = [
        row
        for row in requests
        if row.get("provider_reported_total_tokens") is not None
        and row.get("claim_validation_passed") is True
        and row.get("source_tier_correct") is True
        and row.get("unsafe_leakage") is False
    ]
    unit_cost = None
    if provider["evidence_complete"] and eligible:
        unit_cost = round(
            sum(float(row.get("estimated_cost_usd") or 0.0) for row in eligible)
            / len(eligible),
            8,
        )
    budget = float(
        requirements.get("planning_cost_per_safe_supported_answer_usd_max", 0.002)
    )
    return {
        "status": (
            "pass"
            if unit_cost is not None and unit_cost <= budget
            else "needs_attention"
            if unit_cost is not None
            else "blocked_evidence"
        ),
        "provider_usage_coverage_rate": provider["coverage_rate"],
        "paired_request_count": provider["paired_request_count"],
        "evidence_complete": provider["evidence_complete"],
        "safe_supported_request_count": len(eligible),
        "cost_per_safe_supported_answer_usd": unit_cost,
        "planning_budget_usd": budget,
        "missing_cost_is_zero": False,
        "audited_billing": False,
    }


def _planning_scenarios(
    cost: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    budget = float(
        (policy.get("unit_cost_evidence") or {}).get(
            "planning_cost_per_safe_supported_answer_usd_max", 0.002
        )
    )
    rows = []
    for scenario in cost.get("route_cost_comparison") or []:
        if not isinstance(scenario, dict):
            continue
        scenario_cost = _number(scenario.get("cost_per_successful_safe_answer_usd"))
        rows.append(
            {
                "route": scenario.get("route"),
                "p50_latency_ms_estimate": scenario.get("p50_latency_ms_estimate"),
                "planning_cost_per_safe_answer_usd": scenario_cost,
                "within_planning_budget": scenario_cost is not None and scenario_cost <= budget,
                "accuracy_measured_on_same_requests": False,
                "provider_billing_observed": False,
                "promotion_eligible": False,
                "basis": scenario.get("basis"),
            }
        )
    return rows


def _decision(
    accuracy: dict[str, Any], latency: dict[str, Any], unit_cost: dict[str, Any]
) -> str:
    if accuracy.get("status") != "pass":
        return "HOLD_ACCURACY_GROUNDING"
    if latency.get("status") != "pass":
        return "HOLD_LATENCY"
    if unit_cost.get("status") == "blocked_evidence":
        return "HOLD_UNIT_COST_EVIDENCE"
    if unit_cost.get("status") != "pass":
        return "HOLD_UNIT_COST_BUDGET"
    return "PROMOTE"


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _relative(path: str | Path) -> str:
    target = Path(path)
    try:
        return str(target.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(target).replace("\\", "/")


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["build_ai_trinity_tradeoff", "write_ai_trinity_tradeoff"]
