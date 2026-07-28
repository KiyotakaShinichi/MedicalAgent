from __future__ import annotations

from pathlib import Path

from backend.services.cloud_infrastructure_readiness import (
    REQUIRED_RESOURCE_MARKERS,
    build_cloud_infrastructure_readiness,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_azure_reference_foundation_covers_required_services():
    report = build_cloud_infrastructure_readiness(root_dir=ROOT_DIR)
    assert report["status"] == "reference_architecture_only"
    assert report["cloud_deployment_completed"] is False
    assert report["healthcare_production_ready"] is False
    assert report["patient_data_allowed"] is False
    assert report["failed"] == 0
    assert report["n_checks"] >= len(REQUIRED_RESOURCE_MARKERS)


def test_reference_foundation_defaults_cost_bearing_resources_off():
    text = (ROOT_DIR / "infra" / "azure" / "main.bicep").read_text(encoding="utf-8")
    assert "param deployManagedSearch bool = false" in text
    assert "param deployMessaging bool = false" in text
    assert "param deployPostgres bool = false" in text
    assert "param allowPublicNetworkAccess bool = false" in text


def test_reference_foundation_has_no_committed_password():
    text = (ROOT_DIR / "infra" / "azure" / "main.bicep").read_text(encoding="utf-8")
    assert "@secure()" in text
    assert "param postgresAdminPassword string = ''" in text
    assert "replace_with" not in text.lower()
