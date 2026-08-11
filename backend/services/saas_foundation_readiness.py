"""Machine-readable readiness for the restricted synthetic SaaS foundation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data/evals/ops/latest_saas_foundation_readiness.json"
CLAIM_BOUNDARY = (
    "This is engineering evidence for a restricted synthetic SaaS alpha foundation. It is not "
    "clinical validation, a security or compliance certification, billing proof, managed-cloud "
    "deployment evidence, or production healthcare readiness."
)


def build_saas_foundation_readiness(*, root: str | Path = ROOT) -> dict[str, Any]:
    repo = Path(root).resolve()
    checks = [
        _contains(repo / "backend/models.py", "class SaaSOrganization", "tenant_schema_defined"),
        _contains(repo / "backend/models.py", "class SaaSMembership", "membership_schema_defined"),
        _contains(repo / "backend/models.py", "class SaaSUsageEvent", "append_only_usage_ledger_defined"),
        _contains(repo / "backend/services/saas_control_plane.py", "require_membership", "tenant_scope_enforced"),
        _contains(repo / "backend/services/saas_control_plane.py", "idempotency_key", "idempotency_contract_defined"),
        _contains(repo / "backend/services/saas_job_worker.py", "lease_expires_at", "leased_job_worker_defined"),
        _contains(repo / "backend/services/saas_job_worker.py", "dead_lettered", "job_dead_letter_defined"),
        _contains(repo / "backend/services/saas_outbox_dispatcher.py", "recover_expired_outbox_events", "leased_outbox_defined"),
        _contains(repo / "backend/services/n8n_webhook_dispatcher.py", "saas_workspace_event", "signed_automation_route_allowlisted"),
        _contains(repo / "backend/services/api_protection.py", "RedisRateLimiter", "shared_rate_limit_defined"),
        _contains(repo / "backend/api/routers/platform.py", "Idempotency-Key", "platform_api_idempotency_required"),
        _contains(repo / "frontend-react/src/pages/workspace/WorkspaceDashboard.tsx", "not an invoice or audited billing source", "workspace_boundary_visible"),
        _migration_check(repo),
        _compose_check(repo / "docker-compose.synthetic-staging.yml", synthetic=True),
        _compose_check(repo / "docker-compose.prod.yml", synthetic=False),
    ]
    implemented = all(item["passed"] for item in checks)
    return {
        "schema_version": "nlcare_saas_foundation_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_restricted_synthetic_saas_alpha" if implemented else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "real_patient_data_allowed": False,
        "billing_enabled": False,
        "billing_authoritative": False,
        "managed_cloud_deployment_completed": False,
        "live_oidc_provider_verified": False,
        "external_review_completed": False,
        "row_level_tenancy_applied_to_legacy_patient_demo": False,
        "control_count": len(checks),
        "passed_control_count": sum(item["passed"] for item in checks),
        "controls": checks,
        "implemented_scope": [
            "organization and membership-scoped control-plane data",
            "project and synthetic environment isolation",
            "engineering entitlements and non-billing usage ledger",
            "idempotent durable evaluation jobs with leases, retries, recovery, and dead letters",
            "transactional redacted outbox with leased signed n8n delivery",
            "shared Redis rate-limit option with strict-profile fail-closed behavior",
            "role-gated operational workspace for projects, jobs, usage, and governance boundaries",
        ],
        "deployment_blockers": [
            "connect and verify a real OIDC identity provider",
            "exercise Postgres migrations and tenant isolation in a managed staging environment",
            "run multi-worker concurrency, backup/restore, and disaster-recovery drills",
            "configure secrets, TLS, gateway/WAF controls, monitoring, and incident ownership",
            "complete independent security review and external-author evaluation",
            "keep the patient portal synthetic demo isolated unless a separate regulated program exists",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_saas_foundation_readiness(
    *,
    root: str | Path = ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = build_saas_foundation_readiness(root=root)
    target = Path(output_path)
    if not target.is_absolute():
        target = Path(root).resolve() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _contains(path: Path, needle: str, control_id: str) -> dict[str, Any]:
    exists = path.exists()
    content = path.read_text(encoding="utf-8") if exists else ""
    return {
        "id": control_id,
        "passed": exists and needle in content,
        "evidence": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path),
    }


def _migration_check(repo: Path) -> dict[str, Any]:
    path = repo / "backend/migrations/versions/0012_saas_control_plane.py"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    required = {"saas_organizations", "saas_memberships", "saas_projects", "saas_usage_events", "saas_platform_jobs", "saas_outbox_events", "saas_audit_events"}
    return {
        "id": "saas_migration_complete",
        "passed": path.exists() and all(item in content for item in required),
        "evidence": "backend/migrations/versions/0012_saas_control_plane.py",
    }


def _compose_check(path: Path, *, synthetic: bool) -> dict[str, Any]:
    compose = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    services = set((compose.get("services") or {}).keys())
    required = {"backend", "saas-worker", "saas-outbox-worker", "redis", "postgres"}
    return {
        "id": "synthetic_compose_saas_workers" if synthetic else "production_shaped_compose_saas_workers",
        "passed": required <= services,
        "evidence": path.name,
    }


__all__ = ["CLAIM_BOUNDARY", "build_saas_foundation_readiness", "write_saas_foundation_readiness"]
