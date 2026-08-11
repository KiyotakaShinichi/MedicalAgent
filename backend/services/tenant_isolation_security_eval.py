"""Controlled three-tenant authorization and namespace-isolation matrix."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.services.saas_control_plane import (
    SaaSAccessError,
    SaaSActor,
    cancel_platform_job,
    create_organization,
    create_project,
    enqueue_platform_job,
    list_platform_jobs,
    list_projects,
    record_usage_event,
    require_membership,
    usage_summary,
    workspace_overview,
)
from backend.services.tenant_scoping import tenant_cache_key, tenant_vector_namespace


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data/evals/security/latest_tenant_isolation_security_eval.json"
CLAIM_BOUNDARY = (
    "Controlled synthetic control-plane security regression only. This is not a penetration test, "
    "compliance certification, clinical validation, or proof that the legacy patient demo is multi-tenant."
)


def build_tenant_isolation_security_eval(output_path: Path | str = DEFAULT_OUTPUT) -> dict[str, Any]:
    db = _database()
    tenants: list[dict[str, Any]] = []
    for index in range(3):
        actor = SaaSActor(
            subject=f"oidc|tenant-{index + 1}-owner",
            application_role="admin",
            auth_source="oidc",
        )
        organization = create_organization(db, actor=actor, name=f"Synthetic Tenant {index + 1}")
        project = create_project(
            db,
            organization_id=organization.id,
            actor=actor,
            name=f"Assurance Project {index + 1}",
        )
        job, _ = enqueue_platform_job(
            db,
            organization_id=organization.id,
            project_id=project.id,
            actor=actor,
            job_type="release_gate",
            idempotency_key=f"tenant-seed-{index + 1}",
            payload={"dry_run": True},
        )
        tenants.append({"actor": actor, "organization": organization, "project": project, "job": job})
    db.commit()

    cases: list[dict[str, Any]] = []
    operations: tuple[tuple[str, Callable[..., Any]], ...] = (
        ("membership", lambda attacker, victim, variant: require_membership(
            db, organization_id=victim["organization"].id, actor=attacker["actor"]
        )),
        ("project_list", lambda attacker, victim, variant: list_projects(
            db, organization_id=victim["organization"].id, actor=attacker["actor"]
        )),
        ("job_list", lambda attacker, victim, variant: list_platform_jobs(
            db, organization_id=victim["organization"].id, actor=attacker["actor"]
        )),
        ("workspace_overview", lambda attacker, victim, variant: workspace_overview(
            db, organization_id=victim["organization"].id, actor=attacker["actor"]
        )),
        ("usage_summary", lambda attacker, victim, variant: usage_summary(
            db, organization_id=victim["organization"].id, actor=attacker["actor"]
        )),
        ("foreign_project_enqueue", lambda attacker, victim, variant: enqueue_platform_job(
            db,
            organization_id=attacker["organization"].id,
            project_id=victim["project"].id,
            actor=attacker["actor"],
            job_type="release_gate",
            idempotency_key=f"foreign-{variant}-{attacker['organization'].id}",
            payload={"dry_run": True},
        )),
        ("foreign_job_cancel", lambda attacker, victim, variant: cancel_platform_job(
            db,
            organization_id=attacker["organization"].id,
            job_id=victim["job"].id,
            actor=attacker["actor"],
        )),
    )

    # Twenty request mutations for each ordered tenant pair gives 120 concrete
    # authorization attempts while keeping the protected operations explicit.
    for attacker in tenants:
        for victim in tenants:
            if attacker is victim:
                continue
            for variant in range(20):
                operation_name, operation = operations[variant % len(operations)]
                case_id = f"tenant_{len(cases) + 1:03d}"
                try:
                    value = operation(attacker, victim, variant)
                except SaaSAccessError as exc:
                    cases.append(_case(case_id, operation_name, variant, True, "rejected", str(exc)))
                except Exception as exc:
                    cases.append(_case(case_id, operation_name, variant, False, "unexpected_error", type(exc).__name__))
                else:
                    leaked = _contains_identifier(value, victim["organization"].id, victim["project"].id, victim["job"].id)
                    cases.append(_case(
                        case_id,
                        operation_name,
                        variant,
                        not leaked,
                        "no_protected_data_returned" if not leaked else "unauthorized_data_returned",
                        None,
                    ))

    relation_cases: list[dict[str, Any]] = []
    for index, tenant in enumerate(tenants):
        victim = tenants[(index + 1) % len(tenants)]
        try:
            record_usage_event(
                db,
                organization_id=tenant["organization"].id,
                project_id=victim["project"].id,
                metric_key="provider_tokens",
                quantity=1,
                unit="tokens",
                source="tenant_isolation_eval",
                idempotency_key=f"foreign-relation-{index}",
            )
        except SaaSAccessError as exc:
            relation_cases.append({"case_id": f"relation_{index + 1}", "passed": True, "outcome": "rejected", "detail": str(exc)})
        else:
            relation_cases.append({"case_id": f"relation_{index + 1}", "passed": False, "outcome": "accepted"})

    namespace_cases: list[dict[str, Any]] = []
    for left_index, left in enumerate(tenants):
        for right_index, right in enumerate(tenants):
            if left_index >= right_index:
                continue
            left_cache = tenant_cache_key(left["organization"].id, left["project"].id, "rag", "query")
            right_cache = tenant_cache_key(right["organization"].id, right["project"].id, "rag", "query")
            left_vector = tenant_vector_namespace(left["organization"].id, left["project"].id)
            right_vector = tenant_vector_namespace(right["organization"].id, right["project"].id)
            namespace_cases.append({
                "case_id": f"namespace_{left_index + 1}_{right_index + 1}",
                "passed": left_cache != right_cache and left_vector != right_vector,
                "cache_collision": left_cache == right_cache,
                "vector_namespace_collision": left_vector == right_vector,
            })

    all_cases = [*cases, *relation_cases, *namespace_cases]
    failures = [row for row in all_cases if not row["passed"]]
    payload = {
        "schema_version": "tenant_isolation_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "tenant_count": len(tenants),
        "authorization_attack_count": len(cases),
        "cross_tenant_relation_case_count": len(relation_cases),
        "namespace_case_count": len(namespace_cases),
        "total_case_count": len(all_cases),
        "passed_count": len(all_cases) - len(failures),
        "failed_count": len(failures),
        "unauthorized_success_count": sum(row.get("outcome") == "unauthorized_data_returned" for row in cases),
        "cross_tenant_leakage_count": len(failures),
        "cases": all_cases,
        "limitations": [
            "The matrix targets the synthetic SaaS control-plane service boundary, not the legacy patient-demo tables.",
            "Repeated mutation variants exercise the same bounded operations with distinct request identities; this is not an external penetration test.",
            "Infrastructure-provider IAM, network policy, and managed vector-database enforcement are not exercised here.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _database():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _case(case_id: str, operation: str, variant: int, passed: bool, outcome: str, detail: str | None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "operation": operation,
        "mutation_variant": variant,
        "passed": bool(passed),
        "outcome": outcome,
        "detail": detail,
    }


def _contains_identifier(value: Any, *identifiers: str) -> bool:
    rendered = json.dumps(value, default=str, sort_keys=True)
    return any(identifier and identifier in rendered for identifier in identifiers)


__all__ = ["build_tenant_isolation_security_eval"]
