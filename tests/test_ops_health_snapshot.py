import json

from backend.services.ops_health_snapshot import build_service_health_snapshot


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_snapshot_aggregates_real_metrics_without_production_claims(tmp_path):
    artifacts = {
        "benchmark_registry": _write(tmp_path / "registry.json", {
            "status": "needs_attention",
            "benchmarks": [{"id": "old", "freshness": "stale", "status": "acceptable"}],
        }),
        "release_surface": _write(tmp_path / "surface.json", {
            "engineering_release_decision": "PROCEED_WITH_WARNINGS",
        }),
        "live_rag": _write(tmp_path / "rag.json", {"summary": {
            "case_count": 20,
            "pass_rate": 0.9,
            "claim_support_rate": 0.8,
            "citation_precision": 0.75,
            "post_gen_validator_trigger_rate": 0.1,
            "latency_p50_ms": 400,
        }}),
        "route_latency": _write(tmp_path / "latency.json", {
            "summary": {"insufficient_sample_count": 1},
            "routes": [{"route": "normal_rag", "current_p95_ms": 1200, "percentile_credible": False}],
        }),
        "evidence_abstention": _write(tmp_path / "abstention.json", {"summary": {
            "abstention_rates_by_scenario": {"full_data": 0.1, "no_imaging": 1.0},
        }}),
        "adversarial": _write(tmp_path / "adversarial.json", {
            "overall_attack_block_rate": 0.95,
            "metrics": {"unsafe_leakage_rate": 0.05},
        }),
        "automation": _write(tmp_path / "automation.json", {
            "control_pass_rate": 1.0,
            "live_delivery_test_completed": False,
        }),
        "data_pipeline": _write(tmp_path / "data.json", {
            "patient_data_processed": False,
            "quality": {"hard_failures": 0},
        }),
        "cloud": _write(tmp_path / "cloud.json", {
            "bicep_compile_completed": True,
            "what_if_completed": False,
            "cloud_deployment_completed": False,
        }),
        "deployment": _write(tmp_path / "deployment.json", {
            "strict_profile": False,
            "status": "development_profile",
        }),
    }

    result = build_service_health_snapshot(
        artifacts=artifacts,
        output_path=tmp_path / "health.json",
    )

    assert result["schema_version"] == "service_health_snapshot_v2"
    assert result["metrics"]["retrieval_case_failure_rate"] == 0.1
    assert result["metrics"]["citation_support_failure_rate"] == 0.2
    assert result["metrics"]["data_patient_records_processed"] is False
    assert result["measurement_coverage"] > 0.8
    assert result["status"] == "needs_attention"
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False
    assert result["production_slo_claimed"] is False


def test_snapshot_preserves_missing_measurements_as_none(tmp_path):
    paths = {}
    for name in (
        "benchmark_registry",
        "release_surface",
        "live_rag",
        "route_latency",
        "evidence_abstention",
        "adversarial",
        "automation",
        "data_pipeline",
        "cloud",
        "deployment",
    ):
        paths[name] = tmp_path / f"{name}.json"

    result = build_service_health_snapshot(
        artifacts=paths,
        output_path=tmp_path / "health.json",
    )

    assert result["metrics"]["citation_precision"] is None
    assert result["measurement_coverage"] < 0.5
    assert result["healthcare_production_ready"] is False
