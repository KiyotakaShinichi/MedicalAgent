"""Organization lifecycle, membership bootstrap, usage metering, and overview.

An organization is the billing and tenancy root: projects and jobs are always
scoped beneath one. Usage recording lives here rather than with jobs because it
is metered per organization and read back by the entitlement checks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import func

from backend.models import (
    SaaSAuditEvent,
    SaaSMembership,
    SaaSOrganization,
    SaaSOutboxEvent,
    SaaSProject,
    SaaSUsageEvent,
)

from backend.services.saas_common import (
    CLAIM_BOUNDARY,
    DEFAULT_ENTITLEMENTS,
    SaaSAccessError,
    SaaSActor,
    SaaSValidationError,
    _append_audit,
    _append_outbox,
    _assert_entitled,
    _id,
    _iso,
    _membership,
    _required_text,
    _safe_metadata,
    _scoped_environment_id,
    _scoped_project,
    _seed_entitlements,
    _slug,
    entitlement_status,
    require_membership,
)
from backend.services.saas_projects import _new_project, list_projects


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


def workspace_overview(db: Any, *, organization_id: str, actor: SaaSActor) -> dict[str, Any]:
    # Imported here rather than at module scope: saas_jobs imports
    # record_usage_event from this module, because enqueueing a job meters
    # usage. A module-level import back into saas_jobs would close that cycle.
    # This overview is the only place the organization layer reads job state.
    from backend.services.saas_jobs import list_platform_jobs

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
