"""Evidence-backed operational snapshot for the engineering prototype.

This aggregates existing evaluation artifacts. It deliberately reports missing
or weak measurements instead of filling an SRE dashboard with optimistic nulls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_service_health_snapshot.json"
DEFAULT_ARTIFACTS = {
    "benchmark_registry": "Data/evals/benchmark/latest_benchmark_summary.json",
    "release_surface": "Data/evals/governance/latest_release_decision_surface.json",
    "live_rag": "Data/evals/rag/latest_live_rag_eval.json",
    "route_latency": "Data/evals/ops/latest_route_latency_budget.json",
    "evidence_abstention": "Data/evals/models/latest_evidence_abstention_eval.json",
    "adversarial": "Data/evals/safety/latest_adversarial_safety_regression.json",
    "automation": "Data/evals/ops/latest_durable_automation_worker_eval.json",
    "data_pipeline": "Data/lakehouse/manifests/latest_pipeline_run.json",
    "cloud": "Data/evals/ops/latest_cloud_infrastructure_readiness.json",
    "deployment": "Data/evals/ops/latest_deployment_profile_validation.json",
}


def build_service_health_snapshot(
    *,
    artifacts: Mapping[str, str | Path] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    paths = dict(DEFAULT_ARTIFACTS if artifacts is None else artifacts)
    loaded = {name: _read_json(_resolve(path)) for name, path in paths.items()}

    registry = loaded.get("benchmark_registry", {})
    benchmarks = registry.get("benchmarks") or []
    stale = [row for row in benchmarks if row.get("freshness") == "stale"]
    failed = [
        row
        for row in benchmarks
        if row.get("status") in {"failed", "missing", "error", "unavailable"}
    ]

    live_rag = loaded.get("live_rag", {})
    live_summary = live_rag.get("summary") or {}
    route_latency = loaded.get("route_latency", {})
    route_summary = route_latency.get("summary") or {}
    normal_rag = next(
        (row for row in route_latency.get("routes") or [] if row.get("route") == "normal_rag"),
        {},
    )
    abstention_summary = (loaded.get("evidence_abstention", {}).get("summary") or {})
    adversarial = loaded.get("adversarial", {})
    adversarial_metrics = adversarial.get("metrics") or {}
    automation = loaded.get("automation", {})
    data_pipeline = loaded.get("data_pipeline", {})
    cloud = loaded.get("cloud", {})
    deployment = loaded.get("deployment", {})
    release_surface = loaded.get("release_surface", {})

    metrics = {
        "live_rag_case_count": live_summary.get("case_count"),
        "live_rag_latency_p50_ms": live_summary.get("latency_p50_ms"),
        "normal_rag_latency_p95_ms": normal_rag.get("current_p95_ms"),
        "normal_rag_percentile_credible": normal_rag.get("percentile_credible"),
        "routes_with_insufficient_samples": route_summary.get("insufficient_sample_count"),
        "retrieval_case_failure_rate": _complement(live_summary.get("pass_rate")),
        "citation_support_failure_rate": _complement(live_summary.get("claim_support_rate")),
        "citation_precision": live_summary.get("citation_precision"),
        "post_generation_validator_trigger_rate": live_summary.get("post_gen_validator_trigger_rate"),
        "full_data_abstention_rate": (
            abstention_summary.get("abstention_rates_by_scenario") or {}
        ).get("full_data"),
        "no_imaging_abstention_rate": (
            abstention_summary.get("abstention_rates_by_scenario") or {}
        ).get("no_imaging"),
        "internal_adversarial_unsafe_leakage_rate": adversarial_metrics.get("unsafe_leakage_rate"),
        "internal_attack_block_rate": adversarial.get("overall_attack_block_rate"),
        "automation_control_pass_rate": automation.get("control_pass_rate"),
        "automation_live_delivery_test_completed": automation.get("live_delivery_test_completed"),
        "data_quality_hard_failures": (data_pipeline.get("quality") or {}).get("hard_failures"),
        "data_patient_records_processed": data_pipeline.get("patient_data_processed"),
        "cloud_compile_passed": cloud.get("bicep_compile_completed"),
        "cloud_what_if_completed": cloud.get("what_if_completed"),
        "cloud_deployment_completed": cloud.get("cloud_deployment_completed"),
        "strict_deployment_profile": deployment.get("strict_profile"),
        "deployment_profile_status": deployment.get("status"),
        "release_decision": release_surface.get("engineering_release_decision"),
        "release_gate_status": registry.get("status"),
        "stale_artifact_count": len(stale),
        "failed_benchmark_count": len(failed),
    }
    attention = _attention_items(metrics)
    measured = sum(value is not None for value in metrics.values())
    payload = {
        "schema_version": "service_health_snapshot_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if attention else "acceptable_internal_snapshot",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "production_slo_claimed": False,
        "metric_count": len(metrics),
        "measured_metric_count": measured,
        "measurement_coverage": round(measured / max(1, len(metrics)), 4),
        "metrics": metrics,
        "attention_items": attention,
        "stale_artifacts": [row.get("id") for row in stale[:50]],
        "failed_artifacts": [row.get("id") for row in failed[:50]],
        "source_artifacts": {name: str(path).replace("\\", "/") for name, path in paths.items()},
        "claim_boundary": (
            "Internal engineering observability assembled from local evaluation artifacts. "
            "These values are not production SLOs, clinical monitoring, emergency coverage, "
            "real-world safety evidence, or healthcare production readiness."
        ),
    }
    _write_json(_resolve(output_path), payload)
    return payload


def _attention_items(metrics: Mapping[str, Any]) -> list[str]:
    items: list[str] = []
    if (metrics.get("routes_with_insufficient_samples") or 0) > 0:
        items.append("route_latency_percentiles_have_insufficient_samples")
    if (metrics.get("internal_adversarial_unsafe_leakage_rate") or 0) > 0:
        items.append("internal_adversarial_unsafe_leakage_is_nonzero")
    if metrics.get("automation_live_delivery_test_completed") is not True:
        items.append("live_automation_delivery_not_tested")
    if metrics.get("cloud_what_if_completed") is not True:
        items.append("authenticated_cloud_what_if_not_completed")
    if metrics.get("cloud_deployment_completed") is not True:
        items.append("cloud_deployment_not_completed")
    if metrics.get("strict_deployment_profile") is not True:
        items.append("strict_deployment_profile_not_active")
    if (metrics.get("failed_benchmark_count") or 0) > 0:
        items.append("benchmark_registry_contains_failed_or_missing_entries")
    return items


def _complement(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(max(0.0, min(1.0, 1.0 - float(value))), 4)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = ["DEFAULT_ARTIFACTS", "DEFAULT_OUTPUT_PATH", "build_service_health_snapshot"]
