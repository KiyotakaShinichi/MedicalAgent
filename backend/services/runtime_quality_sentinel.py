"""Runtime quality sentinel for RAG/AI engineering observability.

The sentinel aggregates existing evaluation/ops artifacts into one trendable
snapshot. It is not a clinical safety monitor; it is a PoC engineering guard
that makes regressions visible instead of burying them across many JSON files.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT_DIR / "Data/evals/ops/latest_runtime_quality_sentinel.json"


DEFAULT_THRESHOLDS = {
    "unsupported_claim_rate": 0.05,
    "unsafe_answer_rate": 0.0,
    "over_refusal_rate": 0.05,
    "post_generation_validator_trigger_rate": 0.25,
    "source_governance_rejection_rate": 0.30,
    "latency_p95_ms": 5000.0,
    "drift_ood_alert_count": 0,
}


def build_runtime_quality_sentinel(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    live_rag = _read_json("Data/evals/rag/latest_live_rag_eval.json")
    claim_eval = _read_json("Data/evals/rag/latest_claim_level_citation_eval.json")
    uncertainty = _read_json("Data/evals/rag/latest_uncertainty_aware_retrieval_eval.json")
    over_refusal = _read_json("Data/evals/rag/latest_over_refusal_eval.json")
    cost_latency = _read_json("Data/evals/ops/latest_cost_latency_report.json")
    ood = _read_json("Data/evals/ops/latest_realtime_ood_eval.json")
    trace = _read_json("Data/evals/ops/latest_trace_diagnostics_coverage.json")

    metrics = {
        "unsupported_claim_rate": _first_number(
            _dig(claim_eval, ["summary", "unsupported_claim_rate"]),
            _dig(live_rag, ["summary", "unsupported_answer_rate"]),
            0.0,
        ),
        "unsafe_answer_rate": _first_number(
            _dig(live_rag, ["summary", "unsafe_answer_rate"]),
            _dig(cost_latency, ["summary", "unsafe_leakage_rate"]),
            0.0,
        ),
        "retrieval_confidence_distribution": _confidence_distribution(uncertainty),
        "insufficient_evidence_rate": _first_number(
            _dig(uncertainty, ["summary", "insufficient_evidence_rate"]),
            _dig(uncertainty, ["insufficient_evidence_rate"]),
            0.0,
        ),
        "over_refusal_rate": _first_number(
            _dig(over_refusal, ["summary", "inappropriate_refusal_rate"]),
            0.0,
        ),
        "post_generation_validator_trigger_rate": _first_number(
            _dig(live_rag, ["summary", "post_gen_validator_trigger_rate"]),
            _dig(trace, ["summary", "post_gen_validator_trigger_rate"]),
            0.0,
        ),
        "source_governance_rejection_rate": _first_number(
            _dig(live_rag, ["summary", "source_governance_rejection_rate"]),
            0.0,
        ),
        "cache_hit_rate": _first_number(
            _dig(cost_latency, ["summary", "cache_hit_rate"]),
            0.0,
        ),
        "latency_ms": {
            "p50": _first_number(_dig(cost_latency, ["summary", "overall_latency_ms", "p50"]), _dig(live_rag, ["summary", "latency_p50_ms"]), 0.0),
            "p95": _first_number(_dig(cost_latency, ["summary", "overall_latency_ms", "p95"]), _dig(live_rag, ["summary", "latency_p95_ms"]), 0.0),
            "p99": _first_number(_dig(cost_latency, ["summary", "overall_latency_ms", "p99"]), 0.0),
        },
        "estimated_cost_usd": {
            "total": _first_number(_dig(cost_latency, ["summary", "estimated_total_cost_usd"]), 0.0),
            "per_successful_safe_answer": _first_number(_dig(cost_latency, ["summary", "cost_per_successful_safe_answer"]), 0.0),
        },
        "drift_ood_alerts": {
            "count": int(_first_number(_dig(ood, ["summary", "modality_drift_alert_count"]), 0.0)),
            "severe_ood_abstention_rate": _first_number(_dig(ood, ["summary", "severe_ood_abstention_rate"]), 0.0),
        },
    }

    alerts = _alerts(metrics, thresholds)
    payload = {
        "schema_version": "runtime_quality_sentinel_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_id": "2026-05-ai-swe-hardening",
        "commit_hash": _commit_hash(),
        "status": "needs_attention" if alerts else "strong",
        "thresholds": thresholds,
        "metrics": metrics,
        "summary": {
            "alert_count": len(alerts),
            "unsupported_claim_rate": metrics["unsupported_claim_rate"],
            "unsafe_answer_rate": metrics["unsafe_answer_rate"],
            "insufficient_evidence_rate": metrics["insufficient_evidence_rate"],
            "over_refusal_rate": metrics["over_refusal_rate"],
            "latency_p95_ms": metrics["latency_ms"]["p95"],
            "cache_hit_rate": metrics["cache_hit_rate"],
            "drift_ood_alert_count": metrics["drift_ood_alerts"]["count"],
        },
        "alerts": alerts,
        "source_artifacts": {
            "live_rag": "Data/evals/rag/latest_live_rag_eval.json",
            "claim_level_citation": "Data/evals/rag/latest_claim_level_citation_eval.json",
            "uncertainty_aware_retrieval": "Data/evals/rag/latest_uncertainty_aware_retrieval_eval.json",
            "over_refusal": "Data/evals/rag/latest_over_refusal_eval.json",
            "cost_latency": "Data/evals/ops/latest_cost_latency_report.json",
            "realtime_ood": "Data/evals/ops/latest_realtime_ood_eval.json",
        },
        "claim_boundary": (
            "Runtime quality sentinel is an engineering observability snapshot. "
            "It is not clinical monitoring, patient safety proof, or production SRE."
        ),
    }
    _write_json(Path(output_path), payload)
    return payload


def _alerts(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    checks = [
        ("unsupported_claim_rate", metrics["unsupported_claim_rate"], "<=", thresholds["unsupported_claim_rate"]),
        ("unsafe_answer_rate", metrics["unsafe_answer_rate"], "<=", thresholds["unsafe_answer_rate"]),
        ("over_refusal_rate", metrics["over_refusal_rate"], "<=", thresholds["over_refusal_rate"]),
        ("post_generation_validator_trigger_rate", metrics["post_generation_validator_trigger_rate"], "<=", thresholds["post_generation_validator_trigger_rate"]),
        ("source_governance_rejection_rate", metrics["source_governance_rejection_rate"], "<=", thresholds["source_governance_rejection_rate"]),
        ("latency_p95_ms", metrics["latency_ms"]["p95"], "<=", thresholds["latency_p95_ms"]),
        ("drift_ood_alert_count", metrics["drift_ood_alerts"]["count"], "<=", thresholds["drift_ood_alert_count"]),
    ]
    alerts: list[dict[str, Any]] = []
    for metric, value, op, threshold in checks:
        if value is None:
            continue
        if float(value) > float(threshold):
            alerts.append({
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "comparison": op,
                "severity": "warning" if metric != "unsafe_answer_rate" else "critical",
            })
    return alerts


def _confidence_distribution(artifact: dict[str, Any]) -> dict[str, Any]:
    value = _dig(artifact, ["summary", "retrieval_confidence_distribution"])
    if isinstance(value, dict):
        return value
    cases = artifact.get("cases") if isinstance(artifact, dict) else None
    bins = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    if isinstance(cases, list):
        for case in cases:
            score = _first_number(case.get("retrieval_confidence"), None)
            if score is None:
                bins["unknown"] += 1
            elif score < 0.35:
                bins["low"] += 1
            elif score < 0.70:
                bins["medium"] += 1
            else:
                bins["high"] += 1
    return bins


def _read_json(path: str) -> dict[str, Any]:
    full_path = ROOT_DIR / path
    if not full_path.exists():
        return {}
    try:
        return json.loads(full_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dig(payload: Any, path: list[Any]) -> Any:
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["build_runtime_quality_sentinel"]
