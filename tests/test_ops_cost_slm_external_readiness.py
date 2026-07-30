from __future__ import annotations

from backend.services.cost_latency_observability import build_cost_latency_report
from backend.services.external_stress_test_readiness import build_external_stress_test_readiness
from backend.services.local_slm_readiness import (
    BLOCKED_LOCAL_SLM_SOLO_TASKS,
    ALLOWED_LOCAL_SLM_TASKS,
    build_local_slm_readiness_manifest,
    is_local_slm_task_allowed,
)


def test_cost_latency_report_has_route_comparison_and_boundaries(tmp_path):
    report = build_cost_latency_report(output_path=tmp_path / "cost_latency.json")

    assert report["schema_version"] == "cost_latency_observability_v3"
    assert report["status"] in {"strong", "needs_attention"}
    assert "clinical validation" in report["claim_boundary"].lower()
    assert report["route_cost_comparison"]
    assert "provider_reported_usage" in report["summary"]
    assert "local_probe_stage_latency" in report
    assert report["local_probe_stage_latency"]["source_artifact"].endswith(
        "latest_agent_latency_probe.json"
    )
    assert "estimated_pipeline_usage" in report["summary"]
    assert report["summary"]["overall_latency_ms"]["percentile_credibility"] in {
        "stable_internal_sample",
        "directional_internal_sample",
        "insufficient_n_for_tail_claim",
        "not_measured",
    }
    routes = {row["route"] for row in report["route_cost_comparison"]}
    assert {
        "full_api_path",
        "cached_path",
        "local_slm_routing_plus_api_answer",
        "local_slm_query_rewrite_plus_api_answer",
        "deterministic_only_refusal_path",
    }.issubset(routes)
    for row in report["requests"]:
        assert "route" in row
        assert "latency_ms" in row
        assert "estimated_cost_usd" in row
        assert "cache_status" in row
        assert row["token_usage_basis"] in {
            "provider_reported",
            "per_call_estimate",
            "pipeline_estimate_only",
        }


def test_local_slm_readiness_allows_only_low_risk_tasks(tmp_path):
    manifest = build_local_slm_readiness_manifest(output_path=tmp_path / "slm.json")

    assert manifest["status"] == "strong"
    assert set(ALLOWED_LOCAL_SLM_TASKS).isdisjoint(BLOCKED_LOCAL_SLM_SOLO_TASKS)
    assert is_local_slm_task_allowed("query rewriting")
    assert is_local_slm_task_allowed("claim-extraction")
    assert not is_local_slm_task_allowed("treatment advice")
    assert not is_local_slm_task_allowed("genetic-risk interpretation")
    assert "post_generation_safety_validator" in manifest["required_gates_after_local_slm"]


def test_external_stress_readiness_blocks_promotion_and_clinical_claims(tmp_path):
    report = build_external_stress_test_readiness(output_path=tmp_path / "external.json")

    assert report["schema_version"] == "external_stress_test_readiness_v1"
    assert report["summary"]["clinical_validation"] is False
    assert report["summary"]["promotion_allowed"] is False
    assert "not clinical validation" in report["claim_boundary"].lower()
    dataset_ids = {row["dataset_id"] for row in report["datasets"]}
    assert {"tcga_brca", "metabric", "breastdcedl_spy1", "duke_tcia_mri"}.issubset(dataset_ids)
    for row in report["datasets"]:
        assert row["promotion_allowed"] is False
        assert "mapped_fields" in row
        assert "missing_fields" in row
        assert "prediction_stress" in row
