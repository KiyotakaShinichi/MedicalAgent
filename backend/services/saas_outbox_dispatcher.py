"""Leased delivery for redacted SaaS control-plane events."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from backend.models import SaaSOutboxEvent
from backend.services.n8n_webhook_dispatcher import dispatch_signed_webhook


WORKFLOW_ID = "saas_workspace_event"
CLAIM_BOUNDARY = (
    "This dispatcher sends redacted synthetic engineering events only. It cannot send PHI, "
    "perform clinical actions, establish billing truth, or prove healthcare production readiness."
)


def recover_expired_outbox_events(db: Any, *, now: datetime | None = None) -> int:
    current = now or datetime.now(timezone.utc)
    rows = (
        db.query(SaaSOutboxEvent)
        .filter(
            SaaSOutboxEvent.status == "dispatching",
            SaaSOutboxEvent.lease_expires_at.is_not(None),
            SaaSOutboxEvent.lease_expires_at < current,
        )
        .all()
    )
    for row in rows:
        row.status = "pending"
        row.available_at = current
        row.lease_owner = None
        row.lease_token = None
        row.lease_expires_at = None
        row.recovery_count = int(row.recovery_count or 0) + 1
        row.last_error = "Expired outbox lease recovered."
    db.commit()
    return len(rows)


def claim_next_outbox_event(
    db: Any,
    *,
    worker_id: str,
    lease_seconds: int = 120,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    current = now or datetime.now(timezone.utc)
    query = (
        db.query(SaaSOutboxEvent)
        .filter(
            SaaSOutboxEvent.status == "pending",
            SaaSOutboxEvent.available_at <= current,
        )
        .order_by(SaaSOutboxEvent.available_at.asc(), SaaSOutboxEvent.id.asc())
    )
    try:
        row = query.with_for_update(skip_locked=True).first()
    except Exception:
        row = query.first()
    if row is None:
        return None
    row.status = "dispatching"
    row.attempts = int(row.attempts or 0) + 1
    row.lease_owner = str(worker_id)
    row.lease_token = f"outbox_lease_{uuid4().hex}"
    row.lease_expires_at = current + timedelta(seconds=max(30, min(lease_seconds, 600)))
    db.commit()
    db.refresh(row)
    return serialize_outbox_event(row) | {"lease_token": row.lease_token}


def dispatch_outbox_event(
    db: Any,
    *,
    event_id: str,
    lease_token: str,
    environment: Mapping[str, str] | None = None,
    dispatcher: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    row = (
        db.query(SaaSOutboxEvent)
        .filter(
            SaaSOutboxEvent.id == event_id,
            SaaSOutboxEvent.status == "dispatching",
            SaaSOutboxEvent.lease_token == lease_token,
        )
        .first()
    )
    if row is None:
        raise PermissionError("Outbox lease is missing, expired, or owned by another worker.")
    payload = json.loads(row.payload_json or "{}")
    envelope_payload = {
        "organization_id": row.organization_id,
        "project_id": row.project_id,
        "event_type": row.event_type,
        "aggregate_type": row.aggregate_type,
        "aggregate_id": row.aggregate_id,
        "event": payload,
        "synthetic_only": True,
        "clinical_validation": False,
    }
    sender = dispatcher or dispatch_signed_webhook
    try:
        result = dict(
            sender(
                workflow_id=WORKFLOW_ID,
                payload=envelope_payload,
                event_id=row.id,
                env=dict(os.environ if environment is None else environment),
            )
        )
    except Exception as exc:
        return _fail_delivery(db, row, exc)

    if result.get("sent") is not True:
        # Disabled delivery is a configuration state, not an exhausted attempt.
        row.attempts = max(0, int(row.attempts or 0) - 1)
        row.status = "pending"
        row.available_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        row.last_error = "External automation delivery is disabled; event retained in the outbox."
        _clear_lease(row)
        db.commit()
        db.refresh(row)
        return serialize_outbox_event(row) | {"delivery": result}

    row.status = "dispatched"
    row.dispatched_at = datetime.now(timezone.utc)
    row.last_error = None
    _clear_lease(row)
    db.commit()
    db.refresh(row)
    return serialize_outbox_event(row) | {"delivery": result}


def run_outbox_once(
    db: Any,
    *,
    worker_id: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    values = dict(os.environ if environment is None else environment)
    if not _truthy(values.get("N8N_WEBHOOK_DISPATCH_ENABLED")):
        return None
    recover_expired_outbox_events(db)
    claimed = claim_next_outbox_event(db, worker_id=worker_id)
    if claimed is None:
        return None
    return dispatch_outbox_event(
        db,
        event_id=str(claimed["id"]),
        lease_token=str(claimed["lease_token"]),
        environment=values,
    )


def serialize_outbox_event(row: SaaSOutboxEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "organization_id": row.organization_id,
        "project_id": row.project_id,
        "aggregate_type": row.aggregate_type,
        "aggregate_id": row.aggregate_id,
        "event_type": row.event_type,
        "status": row.status,
        "attempts": int(row.attempts or 0),
        "max_attempts": int(row.max_attempts or 5),
        "available_at": _iso(row.available_at),
        "dispatched_at": _iso(row.dispatched_at),
        "lease_expires_at": _iso(row.lease_expires_at),
        "recovery_count": int(row.recovery_count or 0),
        "last_error": row.last_error,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _fail_delivery(db: Any, row: SaaSOutboxEvent, exc: Exception) -> dict[str, Any]:
    current = datetime.now(timezone.utc)
    exhausted = int(row.attempts or 0) >= int(row.max_attempts or 5)
    row.status = "dead_lettered" if exhausted else "pending"
    row.last_error = str(exc)[:2_000]
    row.available_at = current if exhausted else current + timedelta(
        seconds=min(600, 5 * (2 ** int(row.attempts or 1)))
    )
    _clear_lease(row)
    db.commit()
    db.refresh(row)
    return serialize_outbox_event(row)


def _clear_lease(row: SaaSOutboxEvent) -> None:
    row.lease_owner = None
    row.lease_token = None
    row.lease_expires_at = None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "CLAIM_BOUNDARY",
    "WORKFLOW_ID",
    "claim_next_outbox_event",
    "dispatch_outbox_event",
    "recover_expired_outbox_events",
    "run_outbox_once",
    "serialize_outbox_event",
]
