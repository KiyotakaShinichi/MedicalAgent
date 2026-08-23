"""Project creation and listing within an organization.

A project is the unit a job runs against, and the unit quota is counted in.
Creation is quota-checked against the organization's entitlements before the
row is written.
"""

from __future__ import annotations

from typing import Any


from backend.models import (
    SaaSEnvironment,
    SaaSProject,
)

from backend.services.saas_common import (
    WRITE_ROLES,
    SaaSActor,
    SaaSQuotaExceeded,
    SaaSValidationError,
    _append_audit,
    _append_outbox,
    _id,
    _iso,
    _required_text,
    _slug,
    entitlement_status,
    require_membership,
)


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
