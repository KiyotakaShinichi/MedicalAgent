from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.agent_latency_probe import run_latency_probe  # noqa: E402


OUTPUT_PATH = ROOT / "Data/evals/ops/latest_latency_profile.json"


ROUTE_BUDGETS = {
    "deterministic_safety_refusal": {"ideal_p95_ms": 600, "acceptable_p95_ms": 1500, "needs_attention_p95_ms": 3000},
    "cached_educational_answer": {"ideal_p95_ms": 800, "acceptable_p95_ms": 2000, "needs_attention_p95_ms": 4000},
    "normal_rag": {"ideal_p95_ms": 2500, "acceptable_p95_ms": 8000, "needs_attention_p95_ms": 15000},
    "rag_plus_reranker": {"ideal_p95_ms": 4000, "acceptable_p95_ms": 12000, "needs_attention_p95_ms": 20000},
    "low_confidence_safe_default": {"ideal_p95_ms": 1200, "acceptable_p95_ms": 3000, "needs_attention_p95_ms": 6000},
    "hybrid_prediction": {"ideal_p95_ms": 500, "acceptable_p95_ms": 1500, "needs_attention_p95_ms": 3000},
    "emotional_distress_support": {"ideal_p95_ms": 1500, "acceptable_p95_ms": 5000, "needs_attention_p95_ms": 10000},
}


def main() -> int:
    probe = run_latency_probe(output_path=ROOT / "Data/evals/models/latest_agent_latency_probe.json")
    per_query = probe.get("per_query") or []
    routes = _route_rows(per_query)
    payload = {
        "schema_version": "latency_profile_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status(routes),
        "routes": routes,
        "stage_summary": probe.get("summary") or {},
        "optimization_policy": {
            "skip_reranker_when_retrieval_confidence_high": True,
            "skip_generation_on_deterministic_refusal": True,
            "skip_generation_on_insufficient_evidence_safe_default": True,
            "cache_low_risk_educational_retrieval_only": True,
            "local_load_smoke_forces_sparse_backend": True,
        },
        "claim_boundary": (
            "Latency profile is local engineering observability only. It is not a production SLO, "
            "hospital-readiness claim, or real-world performance guarantee."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "routes": routes}, indent=2))
    return 0


def _route_rows(per_query: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in ROUTE_BUDGETS}
    for row in per_query:
        route = _route_name(row)
        buckets.setdefault(route, []).append(row)
    output = []
    for route, budget in ROUTE_BUDGETS.items():
        rows = buckets.get(route) or []
        values = sorted(float(row.get("total_ms") or 0.0) for row in rows)
        p50 = _pct(values, 50)
        p95 = _pct(values, 95)
        p99 = _pct(values, 99)
        output.append({
            "route": route,
            **budget,
            "sample_count": len(rows),
            "current_p50_ms": p50,
            "current_p95_ms": p95,
            "current_p99_ms": p99,
            "latency_status": _latency_status(p95, budget),
            "bottleneck_stage": _bottleneck(rows),
            "production_ready": False,
        })
    return output


def _route_name(row: dict[str, Any]) -> str:
    terminal = str(row.get("terminal_step") or "").lower()
    query = str(row.get("query") or "").lower()
    if "cache" in terminal:
        return "cached_educational_answer"
    if any(term in query for term in ("stop chemo", "do i have cancer", "ignore previous", "another patient")):
        return "deterministic_safety_refusal"
    if any(term in query for term in ("scared", "natatakot", "panic")):
        return "emotional_distress_support"
    if "turmeric" in query:
        return "low_confidence_safe_default"
    if "generated" in terminal:
        return "normal_rag"
    return "normal_rag"


def _pct(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    idx = round((pct / 100) * (len(values) - 1))
    return round(values[max(0, min(idx, len(values) - 1))], 3)


def _latency_status(p95: float | None, budget: dict[str, float]) -> str:
    if p95 is None:
        return "not_sampled"
    if p95 <= budget["ideal_p95_ms"]:
        return "ideal"
    if p95 <= budget["acceptable_p95_ms"]:
        return "acceptable"
    return "needs_attention"


def _bottleneck(rows: list[dict[str, Any]]) -> str | None:
    totals: dict[str, float] = {}
    for row in rows:
        for stage, value in (row.get("stage_ms") or {}).items():
            totals[stage] = totals.get(stage, 0.0) + float(value or 0.0)
    if not totals:
        return "total_or_external_wait" if rows else None
    return max(totals, key=totals.get)


def _status(routes: list[dict[str, Any]]) -> str:
    if any(row["latency_status"] == "needs_attention" for row in routes if row["sample_count"]):
        return "needs_attention"
    if any(row["latency_status"] == "acceptable" for row in routes if row["sample_count"]):
        return "acceptable"
    return "strong"


if __name__ == "__main__":
    raise SystemExit(main())
