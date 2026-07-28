"""Validate the Azure reference foundation without claiming deployment."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = Path("infra/azure/main.bicep")
DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_cloud_infrastructure_readiness.json")

REQUIRED_RESOURCE_MARKERS = {
    "container_apps_environment": "Microsoft.App/managedEnvironments@",
    "adls_gen2": "Microsoft.Storage/storageAccounts@",
    "key_vault": "Microsoft.KeyVault/vaults@",
    "azure_ai_search": "Microsoft.Search/searchServices@",
    "service_bus": "Microsoft.ServiceBus/namespaces@",
    "postgres_flexible": "Microsoft.DBforPostgreSQL/flexibleServers@",
    "log_analytics": "Microsoft.OperationalInsights/workspaces@",
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
                "managed_services_default_off",
                "param deployManagedSearch bool = false" in text
                and "param deployMessaging bool = false" in text
                and "param deployPostgres bool = false" in text,
                "Cost-bearing managed services remain opt-in.",
            ),
            _check(
                "public_network_default_off",
                "param allowPublicNetworkAccess bool = false" in text,
                "Public network access is disabled by default.",
            ),
            _check(
                "storage_security_baseline",
                "allowBlobPublicAccess: false" in text
                and "allowSharedKeyAccess: false" in text
                and "defaultToOAuthAuthentication: true" in text
                and "isHnsEnabled: true" in text,
                "ADLS Gen2 uses OAuth-first access and blocks public blobs/shared keys.",
            ),
            _check(
                "service_bus_idempotency",
                "requiresDuplicateDetection: true" in text
                and "deadLetteringOnMessageExpiration: true" in text,
                "Engineering event queue enables duplicate detection and dead lettering.",
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

    bicep_tool = shutil.which("bicep")
    az_tool = shutil.which("az")
    passed = sum(check["passed"] for check in checks)
    status = "reference_architecture_only" if passed == len(checks) else "needs_attention"
    payload = {
        "schema_version": "nlcare_cloud_infrastructure_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "cloud_deployment_completed": False,
        "patient_data_allowed": False,
        "provider": "azure_reference_profile",
        "template_path": str(Path(template_path).as_posix()),
        "template_exists": template.exists(),
        "bicep_cli_available": bool(bicep_tool),
        "azure_cli_available": bool(az_tool),
        "bicep_compile_completed": False,
        "what_if_completed": False,
        "n_checks": len(checks),
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
        "service_decisions": {
            "azure_container_apps": "reference compute environment; application revisions not deployed",
            "adls_gen2": "bronze/silver/gold/quarantine target for curated non-patient data products",
            "azure_ai_search": "optional managed hybrid-search shadow candidate",
            "service_bus": "optional engineering events; stable message IDs and idempotent consumers still required",
            "postgres_flexible_server": "optional durable application database; private access and Entra auth remain future work",
            "key_vault": "RBAC-oriented secret boundary; workload role assignments remain future work",
            "log_analytics": "central log target; dashboards, alerts, sampling, and retention review remain future work",
            "data_factory_or_databricks": "not justified at current data volume; local incremental pipeline is the default",
            "azure_managed_redis": "not provisioned; current Redis boundary remains replaceable",
        },
        "known_blockers": [
            "No Azure subscription deployment or what-if run has been completed.",
            "Private endpoints, VNet integration, private DNS, WAF, and egress policy are not implemented.",
            "Container application revisions and managed-identity RBAC assignments are not provisioned.",
            "PostgreSQL Entra-only authentication, backup restore drill, and regional recovery are not proven.",
            "Azure AI Search index creation and frozen shadow retrieval comparison are incomplete.",
            "No measured cloud load, reliability, security, or cost evidence exists.",
        ],
        "official_references": [
            "https://learn.microsoft.com/azure/container-apps/managed-identity",
            "https://learn.microsoft.com/azure/container-apps/manage-secrets",
            "https://learn.microsoft.com/azure/search/search-get-started-vector",
            "https://learn.microsoft.com/rest/api/searchservice/search-service-api-versions",
            "https://learn.microsoft.com/azure/service-bus-messaging/service-bus-message-loss-and-duplicates",
            "https://learn.microsoft.com/azure/architecture/data-guide/scenarios/data-lake",
        ],
        "claim_boundary": (
            "This artifact validates a local Azure reference template only. It is not evidence of a "
            "successful cloud deployment, security certification, HIPAA compliance, clinical validation, "
            "or production healthcare readiness."
        ),
    }
    destination = _resolve(root, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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
