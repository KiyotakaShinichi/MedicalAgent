"""Synthetic SaaS assurance workspace API.

Every resource query is scoped by organization and membership. The clinical
demo remains available under its existing role routes and is not presented as
tenant-isolated patient-care infrastructure.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.api.deps import get_access_context, get_db
from backend.services.saas_control_plane import (
    ALLOWED_JOB_TYPES,
    CLAIM_BOUNDARY,
    SaaSAccessError,
    SaaSQuotaExceeded,
    SaaSValidationError,
    actor_from_access_context,
    bootstrap_demo_workspace,
    cancel_platform_job,
    create_organization,
    create_project,
    enqueue_platform_job,
    list_organizations_for_actor,
    list_platform_jobs,
    list_projects,
    serialize_job,
    usage_summary,
    workspace_overview,
)


router = APIRouter(prefix="/platform", tags=["saas-platform"])


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    slug: str | None = Field(default=None, min_length=3, max_length=80)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    slug: str | None = Field(default=None, min_length=3, max_length=80)
    description: str | None = Field(default=None, max_length=500)


class PlatformJobCreate(BaseModel):
    job_type: str
    environment_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=200)


@router.get("/session")
def platform_session(context=Depends(get_access_context), db=Depends(get_db)):
    actor = actor_from_access_context(context)
    bootstrap_demo_workspace(db, actor)
    db.commit()
    return {
        "actor": {
            "subject": actor.subject,
            "application_role": actor.application_role,
            "auth_source": actor.auth_source,
        },
        "organizations": list_organizations_for_actor(db, actor),
        "synthetic_only": True,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "billing_enabled": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


@router.post("/organizations", status_code=status.HTTP_201_CREATED)
def add_organization(
    payload: OrganizationCreate,
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    try:
        organization = create_organization(
            db,
            actor=actor,
            name=payload.name,
            slug=payload.slug,
        )
        db.commit()
        db.refresh(organization)
        organizations = list_organizations_for_actor(db, actor)
        return next(item for item in organizations if item["id"] == organization.id)
    except (SaaSAccessError, SaaSValidationError) as exc:
        db.rollback()
        raise _http_error(exc) from exc


@router.get("/organizations/{organization_id}/overview")
def get_workspace_overview(
    organization_id: str,
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    try:
        return workspace_overview(db, organization_id=organization_id, actor=actor)
    except (SaaSAccessError, SaaSValidationError) as exc:
        raise _http_error(exc) from exc


@router.get("/organizations/{organization_id}/projects")
def get_projects(
    organization_id: str,
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    try:
        return {
            "projects": list_projects(db, organization_id=organization_id, actor=actor),
            "synthetic_only": True,
            "clinical_validation": False,
        }
    except (SaaSAccessError, SaaSValidationError) as exc:
        raise _http_error(exc) from exc


@router.post("/organizations/{organization_id}/projects", status_code=status.HTTP_201_CREATED)
def add_project(
    organization_id: str,
    payload: ProjectCreate,
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    try:
        project = create_project(
            db,
            organization_id=organization_id,
            actor=actor,
            name=payload.name,
            slug=payload.slug,
            description=payload.description,
        )
        db.commit()
        projects = list_projects(db, organization_id=organization_id, actor=actor)
        return next(item for item in projects if item["id"] == project.id)
    except (SaaSAccessError, SaaSValidationError) as exc:
        db.rollback()
        raise _http_error(exc) from exc


@router.get("/organizations/{organization_id}/usage")
def get_usage(
    organization_id: str,
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    try:
        return {
            "usage": usage_summary(db, organization_id=organization_id, actor=actor),
            "billing_authoritative": False,
            "billing_enabled": False,
            "synthetic_only": True,
            "clinical_validation": False,
            "claim_boundary": (
                "Usage is an engineering ledger for quotas and capacity planning. "
                "It is not an invoice or audited billing record."
            ),
        }
    except (SaaSAccessError, SaaSValidationError) as exc:
        raise _http_error(exc) from exc


@router.get("/organizations/{organization_id}/jobs")
def get_jobs(
    organization_id: str,
    project_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    try:
        return {
            "jobs": list_platform_jobs(
                db,
                organization_id=organization_id,
                actor=actor,
                project_id=project_id,
                limit=limit,
            ),
            "clinical_validation": False,
        }
    except (SaaSAccessError, SaaSValidationError) as exc:
        raise _http_error(exc) from exc


@router.post(
    "/organizations/{organization_id}/projects/{project_id}/jobs",
    status_code=status.HTTP_202_ACCEPTED,
)
def add_job(
    organization_id: str,
    project_id: str,
    payload: PlatformJobCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    effective_key = idempotency_key or payload.idempotency_key
    if not effective_key:
        raise HTTPException(status_code=400, detail="Idempotency-Key header or body field is required.")
    try:
        job, reused = enqueue_platform_job(
            db,
            organization_id=organization_id,
            project_id=project_id,
            actor=actor,
            job_type=payload.job_type,
            idempotency_key=effective_key,
            payload=payload.payload,
            environment_id=payload.environment_id,
        )
        db.commit()
        db.refresh(job)
        return {
            "job": serialize_job(job),
            "idempotent_reuse": reused,
            "allowed_job_types": sorted(ALLOWED_JOB_TYPES),
            "synthetic_only": True,
            "clinical_validation": False,
        }
    except (SaaSAccessError, SaaSValidationError) as exc:
        db.rollback()
        raise _http_error(exc) from exc


@router.delete("/organizations/{organization_id}/jobs/{job_id}")
def cancel_job(
    organization_id: str,
    job_id: str,
    context=Depends(get_access_context),
    db=Depends(get_db),
):
    actor = actor_from_access_context(context)
    try:
        job = cancel_platform_job(
            db,
            organization_id=organization_id,
            job_id=job_id,
            actor=actor,
        )
        db.commit()
        db.refresh(job)
        return {"job": serialize_job(job), "clinical_validation": False}
    except (SaaSAccessError, SaaSValidationError) as exc:
        db.rollback()
        raise _http_error(exc) from exc


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SaaSAccessError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, SaaSQuotaExceeded):
        return HTTPException(status_code=429, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


__all__ = ["router"]
