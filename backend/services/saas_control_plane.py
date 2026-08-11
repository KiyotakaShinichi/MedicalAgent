"""Tenant-scoped control plane for NLCare's synthetic AI assurance workspace.

This module owns organization, project, environment, entitlement, usage,
durable-job, outbox, and audit boundaries. It does not make the legacy patient
demo multi-tenant and it does not authorize real patient data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from sqlalchemy import func

from backend.models import (
    SaaSAuditEvent,
    SaaSEntitlement,
    SaaSEnvironment,
    SaaSMembership,
    SaaSOrganization,
    SaaSOutboxEvent,
    SaaSPlatformJob,
    SaaSProject,
    SaaSUsageEvent,
)
from backend.services.request_context import get_request_id


MEMBERSHIP_ROLES = {"owner", "admin", "evaluator", "viewer"}
WRITE_ROLES = {"owner", "admin"}
RUN_ROLES = {"owner", "admin", "evaluator"}
ALLOWED_JOB_TYPES = {
    "rag_baseline_comparison",
    "adversarial_safety_eval",
    "agent_workflow_eval",
    "release_gate",
    "evidence_packet_export",
}
DEFAULT_ENTITLEMENTS = {
    "project_count": (10.0, 8.0, "projects"),
    "evaluation_runs": (1_000.0, 800.0, "runs"),
    "evaluation_cases": (50_000.0, 40_000.0, "cases"),
    "provider_tokens": (1_000_000.0, 800_000.0, "tokens"),
    "automation_runs": (500.0, 400.0, "runs"),
    "storage_bytes": (1_073_741_824.0, 858_993_459.0, "bytes"),
    "vector_count": (100_000.0, 80_000.0, "vectors"),
}
FORBIDDEN_PAYLOAD_KEY_PARTS = {
    "patient",
    "diagnosis",
    "message",
    "prompt",
    "email",
    "phone",
    "name",
    "address",
    "dob",
    "birth",
    "medical_record",
    "raw_text",
    "content",
}
CLAIM_BOUNDARY = (
    "This control plane supports synthetic AI engineering and evaluation only. "
    "It is not a clinical service, a billing system, a compliance certification, "
    "or evidence of healthcare production readiness."
)


class SaaSAccessError(PermissionError):
    pass


class SaaSValidationError(ValueError):
    pass


class SaaSQuotaExceeded(SaaSValidationError):
    pass


@dataclass(frozen=True)
class SaaSActor:
    subject: str
    application_role: str
    auth_source: str


def actor_from_access_context(context: Any) -> SaaSActor:
    subject = str(getattr(context, "subject", "") or "").strip()
    if not subject:
        role = str(getattr(context, "role", "unknown"))
        patient_id = str(getattr(context, "patient_id", "") or "global")
        subject = f"demo:{role}:{patient_id}"
    return SaaSActor(
        subject=subject,
        application_role=str(getattr(context, "role", "unknown")),
        auth_source=str(getattr(context, "auth_source", "unknown")),
    )


def bootstrap_demo_workspace(db: Any, actor: SaaSActor) -> SaaSOrganization | None:
    """Create one deterministic synthetic workspace for local demo identities."""
    if actor.auth_source != "demo_session" or actor.application_role not in {"admin", "clinician"}:
        return None
    organization = (
        db.query(SaaSOrganization)
        .filter(SaaSOrganization.slug == "nlcare-synthetic-assurance-lab")
        .first()
    )
    if organization is None:
        organization = SaaSOrganization(
            id=_id("org"),
            slug="nlcare-synthetic-assurance-lab",
            name="NLCare Synthetic Assurance Lab",
            status="active",
            plan_code="engineering_preview",
            data_class="synthetic_only",
            created_by_subject=actor.subject,
        )
        db.add(organization)
        db.flush()
        _seed_entitlements(db, organization.id)
    membership = _membership(db, organization.id, actor.subject)
    if membership is None:
        membership = SaaSMembership(
            id=_id("mem"),
            organization_id=organization.id,
            subject=actor.subject,
            role="owner" if actor.application_role == "admin" else "evaluator",
            status="active",
        )
        db.add(membership)
        db.flush()
    project = (
        db.query(SaaSProject)
        .filter(
            SaaSProject.organization_id == organization.id,
            SaaSProject.slug == "breast-monitoring-demo",
        )
        .first()
    )
    if project is None:
        project = _new_project(
            db,
            organization_id=organization.id,
            actor=actor,
            name="Breast Monitoring Demo",
            slug="breast-monitoring-demo",
            description="Synthetic demonstration application; not a patient-care service.",
        )
        _append_audit(
            db,
            organization_id=organization.id,
            project_id=project.id,
            actor=actor,
            action="demo_workspace_bootstrapped",
            target_type="project",
            target_id=project.id,
            details={"data_class": "synthetic_only"},
        )
    return organization


def create_organization(db: Any, *, actor: SaaSActor, name: str, slug: str | None = None) -> SaaSOrganization:
    if actor.application_role != "admin":
        raise SaaSAccessError("Only an application admin can create an organization in this preview.")
    clean_name = _required_text(name, "organization name", 120)
    clean_slug = _slug(slug or clean_name)
    if db.query(SaaSOrganization).filter(SaaSOrganization.slug == clean_slug).first() is not None:
        raise SaaSValidationError("Organization slug is already in use.")
    organization = SaaSOrganization(
        id=_id("org"),
        slug=clean_slug,
        name=clean_name,
        status="active",
        plan_code="engineering_preview",
        data_class="synthetic_only",
        created_by_subject=actor.subject,
    )
    db.add(organization)
    db.flush()
    db.add(SaaSMembership(
        id=_id("mem"),
        organization_id=organization.id,
        subject=actor.subject,
        role="owner",
        status="active",
    ))
    _seed_entitlements(db, organization.id)
    _append_audit(
        db,
        organization_id=organization.id,
        actor=actor,
        action="organization_created",
        target_type="organization",
        target_id=organization.id,
        details={"slug": clean_slug, "data_class": "synthetic_only"},
    )
    _append_outbox(
        db,
        organization_id=organization.id,
        aggregate_type="organization",
        aggregate_id=organization.id,
        event_type="organization.created",
        payload={"organization_id": organization.id, "data_class": "synthetic_only"},
        idempotency_key=f"organization.created:{organization.id}",
    )
    return organization


def list_organizations_for_actor(db: Any, actor: SaaSActor) -> list[dict[str, Any]]:
    rows = (
        db.query(SaaSMembership, SaaSOrganization)
        .join(SaaSOrganization, SaaSOrganization.id == SaaSMembership.organization_id)
        .filter(
            SaaSMembership.subject == actor.subject,
            SaaSMembership.status == "active",
            SaaSOrganization.status == "active",
        )
        .order_by(SaaSOrganization.name.asc())
        .all()
    )
    return [
        {**serialize_organization(organization), "membership_role": membership.role}
        for membership, organization in rows
    ]


def require_membership(
    db: Any,
    *,
    organization_id: str,
    actor: SaaSActor,
    allowed_roles: set[str] | None = None,
) -> SaaSMembership:
    membership = _membership(db, organization_id, actor.subject)
    if membership is None or membership.status != "active":
        raise SaaSAccessError("Workspace not found or access is not permitted.")
    if allowed_roles is not None and membership.role not in allowed_roles:
        raise SaaSAccessError("Your workspace role does not permit this action.")
    return membership


def create_project(
    db: Any,
    *,
    organization_id: str,
    actor: SaaSActor,
    name: str,
    slug: str | None = None,
    description: str | None = None,
) -> SaaSProject:
    require_membership(db, organization_id=organization_id, actor=actor, allowed_roles=WRITE_ROLES)
    entitlement = entitlement_status(db, organization_id=organization_id, metric_key="project_count")
    if entitlement["used"] >= entitlement["hard_limit"]:
        raise SaaSQuotaExceeded("Project quota reached for the engineering preview plan.")
    project = _new_project(
        db,
        organization_id=organization_id,
        actor=actor,
        name=name,
        slug=slug,
        description=description,
    )
    _append_audit(
        db,
        organization_id=organization_id,
        project_id=project.id,
        actor=actor,
        action="project_created",
        target_type="project",
        target_id=project.id,
        details={"slug": project.slug, "data_class": project.data_class},
    )
    _append_outbox(
        db,
        organization_id=organization_id,
        project_id=project.id,
        aggregate_type="project",
        aggregate_id=project.id,
        event_type="project.created",
        payload={"organization_id": organization_id, "project_id": project.id},
        idempotency_key=f"project.created:{project.id}",
    )
    return project


def list_projects(db: Any, *, organization_id: str, actor: SaaSActor) -> list[dict[str, Any]]:
    require_membership(db, organization_id=organization_id, actor=actor)
    projects = (
        db.query(SaaSProject)
        .filter(SaaSProject.organization_id == organization_id, SaaSProject.status == "active")
        .order_by(SaaSProject.created_at.asc(), SaaSProject.id.asc())
        .all()
    )
    return [serialize_project(db, project) for project in projects]


def enqueue_platform_job(
    db: Any,
    *,
    organization_id: str,
    project_id: str,
    actor: SaaSActor,
    job_type: str,
    idempotency_key: str,
    payload: Mapping[str, Any] | None = None,
    environment_id: str | None = None,
) -> tuple[SaaSPlatformJob, bool]:
    require_membership(db, organization_id=organization_id, actor=actor, allowed_roles=RUN_ROLES)
    project = _scoped_project(db, organization_id, project_id)
    if project is None:
        raise SaaSAccessError("Project not found or access is not permitted.")
    clean_type = str(job_type or "").strip()
    if clean_type not in ALLOWED_JOB_TYPES:
        raise SaaSValidationError(f"Unsupported job type: {clean_type}")
    clean_key = _required_text(idempotency_key, "idempotency key", 200)
    existing = (
        db.query(SaaSPlatformJob)
        .filter(
            SaaSPlatformJob.organization_id == organization_id,
            SaaSPlatformJob.idempotency_key == clean_key,
        )
        .first()
    )
    if existing is not None:
        if existing.project_id != project_id or existing.job_type != clean_type:
            raise SaaSValidationError("Idempotency key was already used for a different job.")
        return existing, True
    _assert_entitled(db, organization_id, "evaluation_runs", 1.0)
    sanitized = sanitize_job_payload(payload or {})
    job = SaaSPlatformJob(
        id=_id("job"),
        organization_id=organization_id,
        project_id=project_id,
        environment_id=_scoped_environment_id(db, organization_id, project_id, environment_id),
        job_type=clean_type,
        status="queued",
        payload_json=json.dumps(sanitized, sort_keys=True),
        progress_percent=0,
        attempts=0,
        max_attempts=3,
        idempotency_key=clean_key,
        created_by_subject=actor.subject,
        available_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.flush()
    record_usage_event(
        db,
        organization_id=organization_id,
        project_id=project_id,
        environment_id=job.environment_id,
        metric_key="evaluation_runs",
        quantity=1.0,
        unit="runs",
        source="control_plane_job_accepted",
        idempotency_key=f"usage:job:{job.id}",
        metadata={"job_type": clean_type},
    )
    _append_audit(
        db,
        organization_id=organization_id,
        project_id=project_id,
        actor=actor,
        action="evaluation_job_queued",
        target_type="platform_job",
        target_id=job.id,
        details={"job_type": clean_type},
    )
    _append_outbox(
        db,
        organization_id=organization_id,
        project_id=project_id,
        aggregate_type="platform_job",
        aggregate_id=job.id,
        event_type="evaluation.job.queued",
        payload={"job_id": job.id, "job_type": clean_type, "project_id": project_id},
        idempotency_key=f"job.queued:{job.id}",
    )
    return job, False


def list_platform_jobs(
    db: Any,
    *,
    organization_id: str,
    actor: SaaSActor,
    project_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    require_membership(db, organization_id=organization_id, actor=actor)
    query = db.query(SaaSPlatformJob).filter(SaaSPlatformJob.organization_id == organization_id)
    if project_id:
        if _scoped_project(db, organization_id, project_id) is None:
            raise SaaSAccessError("Project not found or access is not permitted.")
        query = query.filter(SaaSPlatformJob.project_id == project_id)
    rows = query.order_by(SaaSPlatformJob.queued_at.desc()).limit(max(1, min(limit, 200))).all()
    return [serialize_job(row) for row in rows]


def cancel_platform_job(
    db: Any,
    *,
    organization_id: str,
    job_id: str,
    actor: SaaSActor,
) -> SaaSPlatformJob:
    require_membership(db, organization_id=organization_id, actor=actor, allowed_roles=RUN_ROLES)
    job = (
        db.query(SaaSPlatformJob)
        .filter(SaaSPlatformJob.organization_id == organization_id, SaaSPlatformJob.id == job_id)
        .first()
    )
    if job is None:
        raise SaaSAccessError("Job not found or access is not permitted.")
    if job.status not in {"queued", "running"}:
        raise SaaSValidationError(f"Job cannot be cancelled from status={job.status}.")
    job.status = "cancelled"
    job.cancelled_at = datetime.now(timezone.utc)
    job.lease_owner = None
    job.lease_token = None
    job.lease_expires_at = None
    _append_audit(
        db,
        organization_id=organization_id,
        project_id=job.project_id,
        actor=actor,
        action="evaluation_job_cancelled",
        target_type="platform_job",
        target_id=job.id,
        details={"prior_status": "queued_or_running"},
    )
    return job


def record_usage_event(
    db: Any,
    *,
    organization_id: str,
    metric_key: str,
    quantity: float,
    unit: str,
    source: str,
    idempotency_key: str,
    project_id: str | None = None,
    environment_id: str | None = None,
    provider_request_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> tuple[SaaSUsageEvent, bool]:
    if project_id is not None and _scoped_project(db, organization_id, project_id) is None:
        raise SaaSAccessError("Project not found or access is not permitted.")
    if environment_id is not None:
        if project_id is None:
            raise SaaSValidationError("An environment-scoped usage event requires a project_id.")
        environment_id = _scoped_environment_id(
            db,
            organization_id,
            project_id,
            environment_id,
        )
    clean_key = _required_text(idempotency_key, "idempotency key", 200)
    existing = (
        db.query(SaaSUsageEvent)
        .filter(
            SaaSUsageEvent.organization_id == organization_id,
            SaaSUsageEvent.idempotency_key == clean_key,
        )
        .first()
    )
    if existing is not None:
        if existing.metric_key != metric_key or float(existing.quantity) != float(quantity):
            raise SaaSValidationError("Usage idempotency key conflicts with an existing event.")
        return existing, True
    if quantity <= 0:
        raise SaaSValidationError("Usage quantity must be greater than zero.")
    _assert_entitled(db, organization_id, metric_key, float(quantity))
    event = SaaSUsageEvent(
        id=_id("use"),
        organization_id=organization_id,
        project_id=project_id,
        environment_id=environment_id,
        metric_key=metric_key,
        quantity=float(quantity),
        unit=_required_text(unit, "usage unit", 40),
        source=_required_text(source, "usage source", 80),
        billable=0,
        provider_request_id=(str(provider_request_id).strip() or None) if provider_request_id else None,
        idempotency_key=clean_key,
        metadata_json=json.dumps(_safe_metadata(metadata or {}), sort_keys=True),
        occurred_at=occurred_at or datetime.now(timezone.utc),
    )
    db.add(event)
    db.flush()
    return event, False


def usage_summary(db: Any, *, organization_id: str, actor: SaaSActor) -> list[dict[str, Any]]:
    require_membership(db, organization_id=organization_id, actor=actor)
    return [
        entitlement_status(db, organization_id=organization_id, metric_key=metric_key)
        for metric_key in sorted(DEFAULT_ENTITLEMENTS)
    ]


def entitlement_status(db: Any, *, organization_id: str, metric_key: str) -> dict[str, Any]:
    entitlement = (
        db.query(SaaSEntitlement)
        .filter(
            SaaSEntitlement.organization_id == organization_id,
            SaaSEntitlement.metric_key == metric_key,
            SaaSEntitlement.enabled == 1,
        )
        .first()
    )
    if entitlement is None:
        raise SaaSQuotaExceeded(f"No active entitlement for metric={metric_key}.")
    if metric_key == "project_count":
        used = float(
            db.query(func.count(SaaSProject.id))
            .filter(SaaSProject.organization_id == organization_id, SaaSProject.status == "active")
            .scalar()
            or 0
        )
    else:
        period_start = _month_start()
        used = float(
            db.query(func.coalesce(func.sum(SaaSUsageEvent.quantity), 0.0))
            .filter(
                SaaSUsageEvent.organization_id == organization_id,
                SaaSUsageEvent.metric_key == metric_key,
                SaaSUsageEvent.occurred_at >= period_start,
            )
            .scalar()
            or 0.0
        )
    hard_limit = float(entitlement.hard_limit)
    return {
        "metric_key": metric_key,
        "unit": entitlement.unit,
        "used": used,
        "soft_limit": float(entitlement.soft_limit) if entitlement.soft_limit is not None else None,
        "hard_limit": hard_limit,
        "remaining": max(0.0, hard_limit - used),
        "utilization": round(used / hard_limit, 6) if hard_limit else 1.0,
        "period": entitlement.period,
        "billing_authoritative": False,
    }


def workspace_overview(db: Any, *, organization_id: str, actor: SaaSActor) -> dict[str, Any]:
    membership = require_membership(db, organization_id=organization_id, actor=actor)
    organization = db.query(SaaSOrganization).filter(SaaSOrganization.id == organization_id).first()
    projects = list_projects(db, organization_id=organization_id, actor=actor)
    jobs = list_platform_jobs(db, organization_id=organization_id, actor=actor, limit=20)
    audit_count = int(
        db.query(func.count(SaaSAuditEvent.id))
        .filter(SaaSAuditEvent.organization_id == organization_id)
        .scalar()
        or 0
    )
    pending_outbox = int(
        db.query(func.count(SaaSOutboxEvent.id))
        .filter(SaaSOutboxEvent.organization_id == organization_id, SaaSOutboxEvent.status == "pending")
        .scalar()
        or 0
    )
    return {
        "schema_version": "nlcare_saas_workspace_overview_v1",
        "organization": serialize_organization(organization),
        "membership_role": membership.role,
        "projects": projects,
        "recent_jobs": jobs,
        "usage": usage_summary(db, organization_id=organization_id, actor=actor),
        "audit_event_count": audit_count,
        "pending_outbox_event_count": pending_outbox,
        "synthetic_only": True,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "billing_enabled": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def sanitize_job_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SaaSValidationError("Job payload must be an object.")
    return _sanitize_mapping(payload, depth=0)


def append_outbox_event(
    db: Any,
    *,
    organization_id: str,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: Mapping[str, Any],
    idempotency_key: str,
    project_id: str | None = None,
) -> SaaSOutboxEvent:
    return _append_outbox(
        db,
        organization_id=organization_id,
        project_id=project_id,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        event_type=event_type,
        payload=payload,
        idempotency_key=idempotency_key,
    )


def append_audit_event(
    db: Any,
    *,
    organization_id: str,
    actor: SaaSActor,
    action: str,
    target_type: str,
    target_id: str | None,
    details: Mapping[str, Any],
    project_id: str | None = None,
) -> SaaSAuditEvent:
    return _append_audit(
        db,
        organization_id=organization_id,
        project_id=project_id,
        actor=actor,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=details,
    )


def serialize_organization(row: SaaSOrganization) -> dict[str, Any]:
    return {
        "id": row.id,
        "slug": row.slug,
        "name": row.name,
        "status": row.status,
        "plan_code": row.plan_code,
        "data_class": row.data_class,
        "created_at": _iso(row.created_at),
    }


def serialize_project(db: Any, row: SaaSProject) -> dict[str, Any]:
    environments = (
        db.query(SaaSEnvironment)
        .filter(
            SaaSEnvironment.organization_id == row.organization_id,
            SaaSEnvironment.project_id == row.id,
            SaaSEnvironment.status == "active",
        )
        .order_by(SaaSEnvironment.environment_key.asc())
        .all()
    )
    return {
        "id": row.id,
        "organization_id": row.organization_id,
        "slug": row.slug,
        "name": row.name,
        "description": row.description,
        "status": row.status,
        "data_class": row.data_class,
        "created_at": _iso(row.created_at),
        "environments": [
            {
                "id": environment.id,
                "key": environment.environment_key,
                "name": environment.name,
                "status": environment.status,
                "retrieval_profile": environment.retrieval_profile,
                "data_class": environment.data_class,
            }
            for environment in environments
        ],
    }


def serialize_job(row: SaaSPlatformJob) -> dict[str, Any]:
    return {
        "id": row.id,
        "organization_id": row.organization_id,
        "project_id": row.project_id,
        "environment_id": row.environment_id,
        "job_type": row.job_type,
        "status": row.status,
        "error_message": row.error_message,
        "progress_percent": int(row.progress_percent or 0),
        "attempts": int(row.attempts or 0),
        "max_attempts": int(row.max_attempts or 0),
        "payload": json.loads(row.payload_json or "{}"),
        "queued_at": _iso(row.queued_at),
        "available_at": _iso(row.available_at),
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "cancelled_at": _iso(row.cancelled_at),
        "recovery_count": int(row.recovery_count or 0),
        "billing_authoritative": False,
        "clinical_validation": False,
    }


def _new_project(
    db: Any,
    *,
    organization_id: str,
    actor: SaaSActor,
    name: str,
    slug: str | None,
    description: str | None,
) -> SaaSProject:
    clean_name = _required_text(name, "project name", 120)
    clean_slug = _slug(slug or clean_name)
    existing = (
        db.query(SaaSProject)
        .filter(SaaSProject.organization_id == organization_id, SaaSProject.slug == clean_slug)
        .first()
    )
    if existing is not None:
        raise SaaSValidationError("Project slug is already in use in this organization.")
    clean_description = str(description or "").strip()[:500] or None
    project = SaaSProject(
        id=_id("prj"),
        organization_id=organization_id,
        slug=clean_slug,
        name=clean_name,
        description=clean_description,
        status="active",
        data_class="synthetic_only",
        created_by_subject=actor.subject,
    )
    db.add(project)
    db.flush()
    db.add(SaaSEnvironment(
        id=_id("env"),
        organization_id=organization_id,
        project_id=project.id,
        environment_key="synthetic-staging",
        name="Synthetic staging",
        status="active",
        retrieval_profile="sparse_governed",
        data_class="synthetic_only",
    ))
    db.flush()
    return project


def _seed_entitlements(db: Any, organization_id: str) -> None:
    for metric_key, (hard_limit, soft_limit, unit) in DEFAULT_ENTITLEMENTS.items():
        db.add(SaaSEntitlement(
            id=_id("ent"),
            organization_id=organization_id,
            metric_key=metric_key,
            unit=unit,
            hard_limit=hard_limit,
            soft_limit=soft_limit,
            period="current" if metric_key == "project_count" else "monthly",
            enabled=1,
            source="engineering_preview",
        ))
    db.flush()


def _assert_entitled(db: Any, organization_id: str, metric_key: str, requested: float) -> None:
    state = entitlement_status(db, organization_id=organization_id, metric_key=metric_key)
    if requested > state["remaining"]:
        raise SaaSQuotaExceeded(
            f"Quota exceeded for {metric_key}: requested={requested:g}, remaining={state['remaining']:g}."
        )


def _membership(db: Any, organization_id: str, subject: str) -> SaaSMembership | None:
    return (
        db.query(SaaSMembership)
        .filter(
            SaaSMembership.organization_id == organization_id,
            SaaSMembership.subject == subject,
        )
        .first()
    )


def _scoped_project(db: Any, organization_id: str, project_id: str) -> SaaSProject | None:
    return (
        db.query(SaaSProject)
        .filter(SaaSProject.organization_id == organization_id, SaaSProject.id == project_id)
        .first()
    )


def _scoped_environment_id(
    db: Any,
    organization_id: str,
    project_id: str,
    environment_id: str | None,
) -> str | None:
    query = db.query(SaaSEnvironment).filter(
        SaaSEnvironment.organization_id == organization_id,
        SaaSEnvironment.project_id == project_id,
    )
    if environment_id:
        query = query.filter(SaaSEnvironment.id == environment_id)
    row = query.order_by(SaaSEnvironment.environment_key.asc()).first()
    if row is None:
        raise SaaSAccessError("Environment not found or access is not permitted.")
    return row.id


def _append_outbox(
    db: Any,
    *,
    organization_id: str,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: Mapping[str, Any],
    idempotency_key: str,
    project_id: str | None = None,
) -> SaaSOutboxEvent:
    event = SaaSOutboxEvent(
        id=_id("evt"),
        organization_id=organization_id,
        project_id=project_id,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        event_type=event_type,
        payload_json=json.dumps(_safe_metadata(payload), sort_keys=True),
        status="pending",
        attempts=0,
        idempotency_key=idempotency_key,
        available_at=datetime.now(timezone.utc),
    )
    db.add(event)
    db.flush()
    return event


def _append_audit(
    db: Any,
    *,
    organization_id: str,
    actor: SaaSActor,
    action: str,
    target_type: str,
    target_id: str | None,
    details: Mapping[str, Any],
    project_id: str | None = None,
) -> SaaSAuditEvent:
    event = SaaSAuditEvent(
        id=_id("aud"),
        organization_id=organization_id,
        project_id=project_id,
        actor_subject=actor.subject,
        actor_role=actor.application_role,
        action=action,
        target_type=target_type,
        target_id=target_id,
        request_id=get_request_id(),
        details_json=json.dumps(_safe_metadata(details), sort_keys=True),
    )
    db.add(event)
    db.flush()
    return event


def _sanitize_mapping(value: Mapping[str, Any], *, depth: int) -> dict[str, Any]:
    if depth > 3:
        raise SaaSValidationError("Job payload nesting is too deep.")
    output: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        normalized = key.lower().replace("-", "_")
        if any(part in normalized for part in FORBIDDEN_PAYLOAD_KEY_PARTS):
            raise SaaSValidationError(f"Job payload key is not allowed in the synthetic control plane: {key}")
        if isinstance(raw_value, Mapping):
            output[key] = _sanitize_mapping(raw_value, depth=depth + 1)
        elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
            if len(raw_value) > 100:
                raise SaaSValidationError(f"Job payload list is too large: {key}")
            output[key] = [_safe_scalar(item, key) for item in raw_value]
        else:
            output[key] = _safe_scalar(raw_value, key)
    encoded = json.dumps(output, default=str)
    if len(encoded.encode("utf-8")) > 16_384:
        raise SaaSValidationError("Job payload exceeds the 16 KiB control-plane limit.")
    return output


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)[:80]
        if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            output[key] = str(raw_value)[:240] if isinstance(raw_value, str) else raw_value
        elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
            output[key] = [str(item)[:120] for item in list(raw_value)[:20]]
        else:
            output[key] = "[structured_metadata_redacted]"
    return output


def _safe_scalar(value: Any, key: str) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > 500:
            raise SaaSValidationError(f"Job payload value is too long: {key}")
        return value
    raise SaaSValidationError(f"Job payload contains unsupported value type for key: {key}")


def _required_text(value: Any, label: str, maximum: int) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise SaaSValidationError(f"{label.capitalize()} is required.")
    if len(clean) > maximum:
        raise SaaSValidationError(f"{label.capitalize()} must be at most {maximum} characters.")
    return clean


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    if len(clean) < 3:
        raise SaaSValidationError("Slug must contain at least three letters or numbers.")
    return clean[:80]


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _month_start() -> datetime:
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


__all__ = [
    "ALLOWED_JOB_TYPES",
    "CLAIM_BOUNDARY",
    "MEMBERSHIP_ROLES",
    "RUN_ROLES",
    "SaaSAccessError",
    "SaaSActor",
    "SaaSQuotaExceeded",
    "SaaSValidationError",
    "actor_from_access_context",
    "append_audit_event",
    "append_outbox_event",
    "bootstrap_demo_workspace",
    "cancel_platform_job",
    "create_organization",
    "create_project",
    "enqueue_platform_job",
    "entitlement_status",
    "list_organizations_for_actor",
    "list_platform_jobs",
    "list_projects",
    "record_usage_event",
    "require_membership",
    "sanitize_job_payload",
    "serialize_job",
    "usage_summary",
    "workspace_overview",
]
