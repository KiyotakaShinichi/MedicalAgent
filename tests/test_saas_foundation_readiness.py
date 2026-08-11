from backend.services.saas_foundation_readiness import (
    build_saas_foundation_readiness,
    write_saas_foundation_readiness,
)


def test_saas_foundation_readiness_is_complete_but_strictly_nonclinical():
    report = build_saas_foundation_readiness()
    assert report["status"] == "ready_for_restricted_synthetic_saas_alpha"
    assert report["passed_control_count"] == report["control_count"]
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["real_patient_data_allowed"] is False
    assert report["billing_enabled"] is False
    assert report["managed_cloud_deployment_completed"] is False
    assert report["live_oidc_provider_verified"] is False
    assert report["row_level_tenancy_applied_to_legacy_patient_demo"] is False
    assert "not clinical validation" in report["claim_boundary"]


def test_saas_foundation_artifact_is_written(tmp_path):
    target = tmp_path / "saas-readiness.json"
    report = write_saas_foundation_readiness(output_path=target)
    assert target.exists()
    assert report["control_count"] >= 15
    assert {item["id"] for item in report["controls"]} >= {
        "tenant_schema_defined",
        "leased_job_worker_defined",
        "leased_outbox_defined",
        "shared_rate_limit_defined",
    }
