"""Shared control-plane primitives: identity, membership, entitlements, and audit.

These are the pieces every control-plane responsibility needs, so they live in
one place rather than being duplicated across the organization, project, and
job modules. Nothing here is specific to a single resource type.

Entitlements and usage scoping sit here for a structural reason: projects check
quota before creation and jobs check it before enqueueing, so putting the check
in either of those modules would make the other import it and create a cycle.
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
    SaaSOutboxEvent,
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


def _membership(db: Any, organization_id: str, subject: str) -> SaaSMembership | None:
    return (
        db.query(SaaSMembership)
        .filter(
            SaaSMembership.organization_id == organization_id,
            SaaSMembership.subject == subject,
        )
        .first()
    )


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


def _assert_entitled(db: Any, organization_id: str, metric_key: str, requested: float) -> None:
    state = entitlement_status(db, organization_id=organization_id, metric_key=metric_key)
    if requested > state["remaining"]:
        raise SaaSQuotaExceeded(
            f"Quota exceeded for {metric_key}: requested={requested:g}, remaining={state['remaining']:g}."
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


def sanitize_job_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SaaSValidationError("Job payload must be an object.")
    return _sanitize_mapping(payload, depth=0)


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
