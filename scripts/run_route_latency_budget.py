from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data/evals/ops/latest_route_latency_budget.json"
COST_REPORT = ROOT / "Data/evals/ops/latest_cost_latency_report.json"
LOAD_REPORT = ROOT / "Data/evals/ops/latest_load_test_report.json"
RERANKER_REPORT = ROOT / "Data/evals/rag/latest_reranker_ablation.json"
LATENCY_PROFILE = ROOT / "Data/evals/ops/latest_latency_profile.json"


BUDGETS = {
    "deterministic_safety_refusal": {"ideal_p95_ms": 750, "acceptable_p95_ms": 1500, "needs_attention_p95_ms": 3000},
    "cached_educational_answer": {"ideal_p95_ms": 500, "acceptable_p95_ms": 1200, "needs_attention_p95_ms": 2500},
    "dense_sparse_rag": {"ideal_p95_ms": 2500, "acceptable_p95_ms": 6000, "needs_attention_p95_ms": 12000},
    "rag_plus_reranker": {"ideal_p95_ms": 4000, "acceptable_p95_ms": 9000, "needs_attention_p95_ms": 15000},
    "low_confidence_safe_default": {"ideal_p95_ms": 1000, "acceptable_p95_ms": 2500, "needs_attention_p95_ms": 5000},
    "hybrid_prediction": {"ideal_p95_ms": 500, "acceptable_p95_ms": 1500, "needs_attention_p95_ms": 3000},
}


def main() -> int:
    cost = _read(COST_REPORT)
    load = _read(LOAD_REPORT)
    reranker = _read(RERANKER_REPORT)
    profile = _read(LATENCY_PROFILE)
    profile_routes = _profile_routes(profile)
    if profile_routes:
        routes = profile_routes
        status = profile.get("status") or ("needs_attention" if any(row.get("latency_status") == "needs_attention" for row in routes) else "acceptable")
    else:
        route_observed = _observed_routes(cost, load, reranker)
        routes = []
        for route, budget in BUDGETS.items():
            observed = route_observed.get(route)
            status_row = _status(observed, budget)
            routes.append({
                "route": route,
                **budget,
                "observed_p95_ms": observed,
                "status": status_row,
            })
        status = "needs_attention" if any(row["status"] == "needs_attention" for row in routes) else "acceptable"
    payload = {
        "schema_version": "route_latency_budget_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "route_count": len(routes),
            "needs_attention_count": sum(1 for row in routes if (row.get("status") or row.get("latency_status")) == "needs_attention"),
            "insufficient_sample_count": sum(
                1
                for row in routes
                if (row.get("status") or row.get("latency_status")) == "insufficient_samples"
            ),
            "highest_observed_p95_ms": max((row.get("observed_p95_ms") or row.get("current_p95_ms") or 0 for row in routes), default=0),
            "production_ready": False,
        },
        "routes": routes,
        "claim_boundary": (
            "Route latency budgets are prototype engineering targets, not production SLOs. "
            "A p95 is not treated as credible until the route has at least 30 local samples."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _patch_cost_report(payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _observed_routes(cost: dict, load: dict, reranker: dict) -> dict[str, float | None]:
    overall_p95 = _dig(cost, ["summary", "overall_latency_ms", "p95"])
    load_p95 = _dig(load, ["summary", "latency_ms", "p95"])
    reranker_p95 = _dig(reranker, ["summary", "p95_retrieval_latency_ms"])
    route_comparison = _dig(cost, ["route_comparison"]) or _dig(cost, ["routes"]) or []
    observed = {
        "deterministic_safety_refusal": None,
        "cached_educational_answer": None,
        "dense_sparse_rag": float(overall_p95 or load_p95 or 0) or None,
        "rag_plus_reranker": float(reranker_p95 or overall_p95 or 0) or None,
        "low_confidence_safe_default": None,
        "hybrid_prediction": None,
    }
    if isinstance(route_comparison, list):
        for row in route_comparison:
            name = str(row.get("route") or row.get("route_name") or "").lower()
            p95 = row.get("p95_latency_ms") or row.get("latency_p95_ms")
            if p95 is None:
                continue
            if "cache" in name:
                observed["cached_educational_answer"] = float(p95)
            elif "refusal" in name or "deterministic" in name:
                observed["deterministic_safety_refusal"] = float(p95)
            elif "prediction" in name:
                observed["hybrid_prediction"] = float(p95)
    return observed


def _profile_routes(profile: dict) -> list[dict] | None:
    rows = profile.get("routes")
    if not isinstance(rows, list) or not rows:
        return None
    converted = []
    for row in rows:
        route = str(row.get("route") or "")
        converted.append({
            **row,
            "observed_p95_ms": row.get("current_p95_ms"),
            "status": row.get("latency_status"),
        })
    return converted


def _status(observed: float | None, budget: dict[str, int]) -> str:
    if observed is None:
        return "not_measured"
    if observed > budget["needs_attention_p95_ms"]:
        return "needs_attention"
    if observed > budget["acceptable_p95_ms"]:
        return "warning"
    return "acceptable"


def _patch_cost_report(budget_payload: dict) -> None:
    cost = _read(COST_REPORT)
    if not cost:
        return
    cost["route_latency_budget"] = {
        "artifact_path": "Data/evals/ops/latest_route_latency_budget.json",
        "status": budget_payload["status"],
        "summary": budget_payload["summary"],
    }
    COST_REPORT.write_text(json.dumps(cost, indent=2), encoding="utf-8")


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dig(payload, path):
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
