"""Phase-3 latency transparency and optimization plan artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_latency_phase3_plan.json")
LATENCY_PROFILE = Path("Data/evals/ops/latest_latency_profile_phase2.json")
ROUTE_BUDGET = Path("Data/evals/ops/latest_route_latency_budget.json")
COST_REPORT = Path("Data/evals/ops/latest_cost_latency_report.json")

CLAIM_BOUNDARY = (
    "Phase-3 latency planning is local engineering observability. It is not a "
    "production SLO, hospital-readiness claim, or real-world performance proof."
)


def build_latency_phase3_plan(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    profile = _read(LATENCY_PROFILE)
    budget = _read(ROUTE_BUDGET)
    cost = _read(COST_REPORT)
    routes = _route_rows(profile, budget)
    bottlenecks = _bottlenecks(routes)
    status = "acceptable" if not any(row["phase3_status"] == "needs_attention" for row in routes if row["sample_count"]) else "needs_attention"
    payload = {
        "schema_version": "latency_phase3_plan_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "headline_metric": _headline(routes),
        "total_n": len(routes),
        "pass_count": sum(1 for row in routes if row["phase3_status"] in {"ideal", "acceptable", "not_sampled"}),
        "fail_count": sum(1 for row in routes if row["phase3_status"] == "needs_attention"),
        "skipped_count": 0,
        "routes": routes,
        "bottleneck_summary": bottlenecks,
        "safe_optimizations_enabled_or_preserved": [
            "skip generation on deterministic refusal",
            "keep cross-encoder reranker disabled unless retrieval improvement is proven",
            "separate warm-up/cold-start timing from steady route timing",
            "cache only low-risk educational answers tied to KB fingerprint",
            "short-circuit safe-default routes before generation when evidence is insufficient",
        ],
        "next_optimization_backlog": [
            {
                "item": "parallelize dense/sparse retrieval when dense backend is enabled",
                "risk": "low",
                "proof": "before/after retrieval p95 without source-tier regression",
            },
            {
                "item": "persist reusable low-risk retrieval contexts with KB fingerprint",
                "risk": "medium",
                "proof": "cached educational p95 plus citation-preservation test",
            },
            {
                "item": "batch semantic claim checks when multiple claims cite the same snippet",
                "risk": "low",
                "proof": "validator latency p95 reduction with same hard-failure count",
            },
        ],
        "cost_summary": cost.get("summary", {}),
        "production_ready": False,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _route_rows(profile: dict[str, Any], budget: dict[str, Any]) -> list[dict[str, Any]]:
    profile_routes = profile.get("routes") if isinstance(profile.get("routes"), list) else []
    budget_routes = {row.get("route"): row for row in budget.get("routes", []) if isinstance(row, dict)}
    rows = []
    for row in profile_routes:
        route = row.get("route")
        p95 = row.get("current_p95_ms")
        b = budget_routes.get(route, {})
        acceptable = b.get("acceptable_p95_ms") or row.get("acceptable_p95_ms")
        needs_attention = b.get("needs_attention_p95_ms") or row.get("needs_attention_p95_ms")
        phase3_status = _status(p95, acceptable, needs_attention)
        rows.append({
            "route": route,
            "sample_count": row.get("sample_count", 0),
            "current_p50_ms": row.get("current_p50_ms"),
            "current_p95_ms": p95,
            "current_p99_ms": row.get("current_p99_ms"),
            "acceptable_p95_ms": acceptable,
            "needs_attention_p95_ms": needs_attention,
            "bottleneck_stage": row.get("bottleneck_stage"),
            "phase3_status": phase3_status,
            "production_ready": False,
        })
    return rows


def _status(p95: float | None, acceptable: float | None, needs_attention: float | None) -> str:
    if p95 is None:
        return "not_sampled"
    if acceptable is not None and p95 <= acceptable:
        return "acceptable"
    if needs_attention is not None and p95 > needs_attention:
        return "needs_attention"
    return "warning"


def _bottlenecks(routes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in routes:
        stage = row.get("bottleneck_stage") or "unknown"
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _headline(routes: list[dict[str, Any]]) -> str:
    measured = [row for row in routes if row.get("current_p95_ms") is not None]
    if not measured:
        return "no measured routes"
    worst = max(measured, key=lambda row: float(row.get("current_p95_ms") or 0.0))
    return f"worst measured p95 {worst['current_p95_ms']} ms on {worst['route']}"


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


__all__ = ["build_latency_phase3_plan"]
