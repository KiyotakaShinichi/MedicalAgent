from __future__ import annotations

from pathlib import Path

from backend.services.cloud_infrastructure_readiness import (
    REQUIRED_RESOURCE_MARKERS,
    build_cloud_infrastructure_readiness,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_azure_reference_foundation_covers_required_services():
    report = build_cloud_infrastructure_readiness(root_dir=ROOT_DIR)
    assert report["status"] == "compiled_reference_architecture"
    assert report["cloud_deployment_completed"] is False
    assert report["healthcare_production_ready"] is False
    assert report["patient_data_allowed"] is False
    assert report["bicep_compile_completed"] is True
    assert report["what_if_completed"] is False
    assert report["failed"] == 0
    assert report["n_checks"] >= len(REQUIRED_RESOURCE_MARKERS)


def test_reference_foundation_defaults_cost_bearing_resources_off():
    text = (ROOT_DIR / "infra" / "azure" / "main.bicep").read_text(encoding="utf-8")
    assert "param deployManagedSearch bool = false" in text
    assert "param deployMessaging bool = false" in text
    assert "param deployPostgres bool = false" in text
    assert "param deployComputeEnvironment bool = false" in text
    assert "param deployApplication bool = false" in text
    assert "param deployPrivateNetworking bool = false" in text
    assert "param deployOperationalAlerts bool = false" in text
    assert "param deployCostControls bool = false" in text
    assert "param allowPublicNetworkAccess bool = false" in text


def test_reference_foundation_has_no_committed_password():
    text = (ROOT_DIR / "infra" / "azure" / "main.bicep").read_text(encoding="utf-8")
    assert "@secure()" in text
    assert "param postgresAdminPassword string = ''" in text
    assert "replace_with" not in text.lower()


def test_reference_foundation_declares_private_network_rbac_budget_and_recovery_controls():
    text = (ROOT_DIR / "infra" / "azure" / "main.bicep").read_text(encoding="utf-8")
    assert "Microsoft.Network/privateEndpoints@" in text
    assert "Microsoft.Network/privateDnsZones@" in text
    assert "Microsoft.ManagedIdentity/userAssignedIdentities@" in text
    assert "Microsoft.Authorization/roleAssignments@" in text
    assert "Microsoft.Consumption/budgets@" in text
    assert "Microsoft.Insights/activityLogAlerts@" in text
    assert "isVersioningEnabled: true" in text
    assert "param postgresBackupRetentionDays int = 14" in text
    assert "keyVaultUrl: databaseUrlSecretUri" in text
    assert "identity: workloadIdentity.id" in text
    assert "secretRef: 'database-url'" in text


def test_ship_ci_installs_checksum_pinned_bicep_cli():
    workflow = (ROOT_DIR / ".github" / "workflows" / "ship.yml").read_text(
        encoding="utf-8"
    )
    assert "v0.45.15/bicep-linux-x64" in workflow
    assert (
        "ff5b194b042c220df4a50d6768ed1d6c39a32894bfdc4ff83d62b115d966a7ce"
        in workflow
    )
    assert "sha256sum --check" in workflow
