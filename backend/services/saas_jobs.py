"""Platform job enqueue, listing, and cancellation.

Jobs are the only control-plane resource that consumes metered quota at
submission time, so enqueueing both validates the payload and records a usage
event. Payload sanitisation is deliberately strict: a job payload is
caller-supplied data that ends up persisted and replayed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping


from backend.models import (
    SaaSPlatformJob,
)

from backend.services.saas_common import (
    ALLOWED_JOB_TYPES,
    RUN_ROLES,
    SaaSAccessError,
    SaaSActor,
    SaaSValidationError,
    _append_audit,
    _append_outbox,
    _assert_entitled,
    _id,
    _iso,
    _required_text,
    _scoped_environment_id,
    _scoped_project,
    require_membership,
    sanitize_job_payload,
)
from backend.services.saas_organizations import record_usage_event


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
