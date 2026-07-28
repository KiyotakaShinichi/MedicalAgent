"""Compile and inspect the Azure reference foundation without claiming deployment."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = Path("infra/azure/main.bicep")
DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_cloud_infrastructure_readiness.json")

REQUIRED_RESOURCE_MARKERS = {
    "container_apps_environment": "Microsoft.App/managedEnvironments@",
    "container_app_revision": "Microsoft.App/containerApps@",
    "adls_gen2": "Microsoft.Storage/storageAccounts@",
    "key_vault": "Microsoft.KeyVault/vaults@",
    "azure_ai_search": "Microsoft.Search/searchServices@",
    "service_bus": "Microsoft.ServiceBus/namespaces@",
    "postgres_flexible": "Microsoft.DBforPostgreSQL/flexibleServers@",
    "log_analytics": "Microsoft.OperationalInsights/workspaces@",
    "workload_identity": "Microsoft.ManagedIdentity/userAssignedIdentities@",
    "private_endpoints": "Microsoft.Network/privateEndpoints@",
    "private_dns": "Microsoft.Network/privateDnsZones@",
    "role_assignments": "Microsoft.Authorization/roleAssignments@",
    "action_groups": "Microsoft.Insights/actionGroups@",
    "cost_budget": "Microsoft.Consumption/budgets@",
}


def build_cloud_infrastructure_readiness(
    *,
    root_dir: str | Path = ROOT_DIR,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    root = Path(root_dir)
    template = _resolve(root, template_path)
    text = template.read_text(encoding="utf-8") if template.exists() else ""

    checks = [
        _check(
            f"resource_{name}",
            marker in text,
            f"Reference foundation declares {name.replace('_', ' ')}.",
        )
        for name, marker in REQUIRED_RESOURCE_MARKERS.items()
    ]
    checks.extend(
        [
            _check(
                "cost_bearing_services_default_off",
                all(
                    marker in text
                    for marker in (
                        "param deployComputeEnvironment bool = false",
                        "param deployApplication bool = false",
                        "param deployPrivateNetworking bool = false",
                        "param deployManagedSearch bool = false",
                        "param deployMessaging bool = false",
                        "param deployPostgres bool = false",
                        "param deployOperationalAlerts bool = false",
                        "param deployCostControls bool = false",
                    )
                ),
                "Cost-bearing and network resources remain opt-in.",
            ),
            _check(
                "public_network_default_off",
                "param allowPublicNetworkAccess bool = false" in text,
                "Public network access is disabled by default.",
            ),
            _check(
                "container_app_identity_secret_reference",
                "keyVaultUrl: databaseUrlSecretUri" in text
                and "identity: workloadIdentity.id" in text
                and "secretRef: 'database-url'" in text
                and "path: '/health'" in text
                and "path: '/ready'" in text,
                "Optional internal app revision uses workload identity, Key Vault reference, and probes.",
            ),
            _check(
                "private_network_contract",
                all(
                    marker in text
                    for marker in (
                        "privateEndpointNetworkPolicies: 'Disabled'",
                        "privatelink.search.windows.net",
                        "privatelink.vaultcore.azure.net",
                        "Microsoft.DBforPostgreSQL/flexibleServers",
                        "privateDnsZoneArmResourceId",
                    )
                ),
                "Private endpoint, DNS, and delegated PostgreSQL network contracts are declared.",
            ),
            _check(
                "storage_security_and_recovery_baseline",
                all(
                    marker in text
                    for marker in (
                        "allowBlobPublicAccess: false",
                        "allowSharedKeyAccess: false",
                        "defaultToOAuthAuthentication: true",
                        "isHnsEnabled: true",
                        "isVersioningEnabled: true",
                        "days: 14",
                    )
                ),
                "ADLS Gen2 uses OAuth-first access, versioning, and soft-delete retention.",
            ),
            _check(
                "managed_identity_rbac",
                all(
                    marker in text
                    for marker in (
                        "storageBlobDataContributorRoleId",
                        "searchIndexDataContributorRoleId",
                        "keyVaultSecretsUserRoleId",
                        "principalType: 'ServicePrincipal'",
                    )
                ),
                "A user-assigned workload identity receives scoped data-plane roles.",
            ),
            _check(
                "service_bus_idempotency",
                "requiresDuplicateDetection: true" in text
                and "deadLetteringOnMessageExpiration: true" in text,
                "Engineering event queue enables duplicate detection and dead lettering.",
            ),
            _check(
                "cost_and_operational_alert_contract",
                "Actual80Percent" in text
                and "Forecast100Percent" in text
                and "deploymentFailureAlert" in text
                and "!empty(operationsContactEmail)" in text,
                "Budget and operational alerts require an explicit engineering contact.",
            ),
            _check(
                "postgres_backup_contract",
                "param postgresBackupRetentionDays int = 14" in text
                and "postgresGeoRedundantBackup" in text
                and "autoGrow: 'Enabled'" in text,
                "PostgreSQL declares retention, optional geo-backup, and storage autogrow.",
            ),
            _check(
                "secrets_not_committed",
                "param postgresAdminPassword string = ''" in text
                and "@secure()" in text
                and "replace_with" not in text.lower(),
                "PostgreSQL password is a secure deployment parameter, not a committed value.",
            ),
            _check(
                "anti_overclaim_outputs",
                "output clinicalValidation bool = false" in text
                and "output healthcareProductionReady bool = false" in text
                and "output patientDataAllowed bool = false" in text,
                "Template outputs preserve the project claim boundary.",
            ),
        ]
    )

    bicep_tool = _find_bicep(root)
    compile_result = _compile_bicep(bicep_tool, template)
    checks.append(
        _check(
            "bicep_compiles",
            compile_result["completed"] and compile_result["exit_code"] == 0,
            "The checked-in template compiles with the available Bicep CLI.",
        )
    )
    az_tool = shutil.which("az")
    azure_auth = _azure_auth_status(az_tool)
    passed = sum(check["passed"] for check in checks)
    status = (
        "compiled_reference_architecture"
        if passed == len(checks)
        else "needs_attention"
    )
    what_if_blocker = (
        "azure_cli_unavailable"
        if not az_tool
        else "azure_authentication_unavailable"
        if not azure_auth["authenticated"]
        else "not_requested_to_avoid_unreviewed_cost_scope"
    )
    payload = {
        "schema_version": "nlcare_cloud_infrastructure_readiness_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "cloud_deployment_completed": False,
        "patient_data_allowed": False,
        "provider": "azure_reference_profile",
        "deployment_environment": "disposable_non_patient_dev_only",
        "template_path": Path(template_path).as_posix(),
        "template_exists": template.exists(),
        "bicep_cli_available": bool(bicep_tool),
        "bicep_cli_path": _relative_or_name(root, bicep_tool),
        "azure_cli_available": bool(az_tool),
        "azure_authenticated": azure_auth["authenticated"],
        "azure_subscription_id_present": azure_auth["subscription_id_present"],
        "bicep_compile_completed": compile_result["completed"]
        and compile_result["exit_code"] == 0,
        "bicep_compile_exit_code": compile_result["exit_code"],
        "bicep_compile_diagnostics": compile_result["diagnostics"],
        "what_if_completed": False,
        "what_if_blocker": what_if_blocker,
        "n_checks": len(checks),
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
        "service_decisions": {
            "azure_container_apps": "optional internal API revision with managed identity, Key Vault reference, and health probes",
            "adls_gen2": "curated non-patient bronze/silver/gold/quarantine target with versioning and soft delete",
            "azure_ai_search": "optional Entra/RBAC, private-endpoint hybrid-search shadow candidate",
            "service_bus": "optional engineering events with duplicate detection and dead lettering",
            "postgres_flexible_server": "optional private delegated-subnet database with 14-day default backup retention",
            "key_vault": "RBAC secret boundary with private endpoint contract and purge protection",
            "log_analytics": "central engineering logs; public ingestion/query disabled by default",
            "managed_identity": "user-assigned workload identity with scoped storage, vault, search, and messaging roles",
            "cost_management": "optional 80% actual and 100% forecast budget notifications",
            "alerts": "optional failed control-plane operation alert to an engineering address",
            "data_factory_or_databricks": "not justified at current volume; local incremental pipeline remains canonical",
            "azure_managed_redis": "not provisioned; existing cache abstraction remains replaceable",
        },
        "known_blockers": [
            "No authenticated Azure CLI or subscription what-if is available on this machine.",
            "No Azure resource has been deployed and no cost-bearing service has been enabled.",
            "Private connectivity has not been exercised from a deployed workload.",
            "Container application and Key Vault references are defined but not deployed.",
            "Private Log Analytics ingestion requires an Azure Monitor Private Link design before a private deployment.",
            "PostgreSQL point-in-time restore and regional recovery remain unproven.",
            "Azure AI Search index provisioning and frozen shadow retrieval comparison remain incomplete.",
            "No measured cloud load, reliability, security, or cost evidence exists.",
        ],
        "official_references": [
            "https://learn.microsoft.com/azure/container-apps/managed-identity",
            "https://learn.microsoft.com/azure/search/search-security-rbac",
            "https://learn.microsoft.com/azure/search/service-create-private-endpoint",
            "https://learn.microsoft.com/azure/storage/common/storage-private-endpoints",
            "https://learn.microsoft.com/azure/postgresql/flexible-server/concepts-networking-private",
            "https://learn.microsoft.com/azure/cost-management-billing/costs/quick-create-budget-bicep",
        ],
        "claim_boundary": (
            "This artifact compiles and inspects a local Azure reference template only. It is not evidence "
            "of a successful cloud deployment, security certification, HIPAA compliance, clinical "
            "validation, or production healthcare readiness."
        ),
    }
    destination = _resolve(root, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _find_bicep(root: Path) -> str | None:
    local = root / "tools" / "bin" / "bicep.exe"
    if local.exists():
        return str(local)
    return shutil.which("bicep")


def _compile_bicep(tool: str | None, template: Path) -> dict[str, Any]:
    if not tool or not template.exists():
        return {
            "completed": False,
            "exit_code": None,
            "diagnostics": ["bicep_cli_or_template_unavailable"],
        }
    result = subprocess.run(
        [tool, "build", str(template), "--stdout"],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    diagnostics = [
        line.strip()
        for line in (result.stderr or "").splitlines()
        if line.strip()
    ]
    return {
        "completed": True,
        "exit_code": result.returncode,
        "diagnostics": diagnostics[:20],
    }


def _azure_auth_status(tool: str | None) -> dict[str, bool]:
    if not tool:
        return {"authenticated": False, "subscription_id_present": False}
    try:
        result = subprocess.run(
            [tool, "account", "show", "--output", "json"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        payload = {}
    return {
        "authenticated": bool(payload.get("tenantId") and payload.get("id")),
        "subscription_id_present": bool(payload.get("id")),
    }


def _relative_or_name(root: Path, tool: str | None) -> str | None:
    if not tool:
        return None
    path = Path(tool)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _check(check_id: str, passed: bool, description: str) -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed), "description": description}


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TEMPLATE_PATH",
    "REQUIRED_RESOURCE_MARKERS",
    "ROOT_DIR",
    "build_cloud_infrastructure_readiness",
]
