"""Compare the RAG runtime-cache change with a fixed local probe baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_PATH = Path(
    "Data/evals/ops/baselines/retrieval_runtime_cache_baseline.json"
)
DEFAULT_CURRENT_PATH = Path("Data/evals/models/latest_agent_latency_probe.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_retrieval_runtime_cache_eval.json")

CLAIM_BOUNDARY = (
    "This comparison measures a repeated local forced-sparse regression workload. "
    "It does not measure dense unique-query latency, provider/network latency, "
    "cloud load, production SLOs, clinical validation, or patient benefit."
)


def build_retrieval_runtime_cache_eval(
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
    current_path: str | Path = DEFAULT_CURRENT_PATH,
) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    current = _read_json(current_path)
    summary = (current.get("summary_by_route") or {}).get("normal_rag") or {}
    current_metrics = {
        "sample_count": _value(summary, "total_ms", "samples"),
        "retrieval_p50_ms": _value(summary, "retrieval_ms", "median_ms"),
        "retrieval_p95_ms": _value(summary, "retrieval_ms", "p95_ms"),
        "total_p50_ms": _value(summary, "total_ms", "median_ms"),
        "total_p95_ms": _value(summary, "total_ms", "p95_ms"),
    }
    comparison = {
        metric: _delta(float(baseline[metric]), float(current_metrics[metric]))
        for metric in (
            "retrieval_p50_ms",
            "retrieval_p95_ms",
            "total_p50_ms",
            "total_p95_ms",
        )
    }
    enough_samples = int(current_metrics["sample_count"] or 0) >= 30
    improved = (
        enough_samples
        and comparison["retrieval_p95_ms"]["delta_ms"] < 0
        and comparison["total_p95_ms"]["delta_ms"] < 0
    )
    return {
        "schema_version": "retrieval_runtime_cache_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "improved_local_regression" if improved else "needs_attention",
        "optimization": {
            "index_file_cache": True,
            "bm25_runtime_cache": True,
            "faiss_runtime_cache": True,
            "bounded_exact_query_cache": True,
            "index_change_invalidation": "mtime_and_size",
        },
        "baseline": baseline,
        "current": {
            **current_metrics,
            "source_generated_at": current.get("generated_at"),
            "workload": baseline.get("workload"),
        },
        "comparison": comparison,
        "sample_contract_met": enough_samples,
        "local_regression_improvement_observed": improved,
        "dense_unique_query_latency_measured": False,
        "provider_network_latency_measured": False,
        "production_ready": False,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_retrieval_runtime_cache_eval(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
    current_path: str | Path = DEFAULT_CURRENT_PATH,
) -> dict[str, Any]:
    payload = build_retrieval_runtime_cache_eval(baseline_path, current_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _value(summary: dict[str, Any], stage: str, metric: str) -> float | int:
    return ((summary.get(stage) or {}).get(metric)) or 0


def _delta(before: float, after: float) -> dict[str, float]:
    relative = (after - before) / before if before else 0.0
    return {
        "baseline_ms": round(before, 2),
        "current_ms": round(after, 2),
        "delta_ms": round(after - before, 2),
        "relative_change": round(relative, 4),
    }


__all__ = [
    "build_retrieval_runtime_cache_eval",
    "write_retrieval_runtime_cache_eval",
]
