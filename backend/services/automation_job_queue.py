from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from backend.models import AsyncTask
from backend.services.background_eval_worker import enqueue_job
from backend.services.task_queue import run_task, task_to_dict


TASK_PREFIX = "safe_automation:"
MAX_AUTOMATION_ATTEMPTS = 3


def enqueue_automation_task(
    db,
    *,
    job_type: str,
    requested_by: str,
    payload: Mapping[str, Any] | None = None,
    dry_run: bool = True,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    job = enqueue_job(
        job_type=job_type,
        requested_by=requested_by,
        payload=payload,
        dry_run=dry_run,
    )
    if not job["accepted"]:
        raise ValueError(job["rejected_reason"] or "automation_job_rejected")
    effective_key = str(idempotency_key or uuid4())
    key_hash = hashlib.sha256(effective_key.encode("utf-8")).hexdigest()
    existing = _find_by_idempotency_key(db, key_hash)
    if existing is not None:
        result = automation_task_to_dict(existing)
        result["idempotent_reuse"] = True
        return result
    job["idempotency_key_hash"] = key_hash
    row = AsyncTask(
        task_type=f"{TASK_PREFIX}{job_type}",
        status="queued",
        payload_json=json.dumps({"job": job}, default=str),
        created_by=requested_by,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    result = automation_task_to_dict(row)
    result["idempotent_reuse"] = False
    return result


def run_automation_task(db, task_id: int) -> dict[str, Any]:
    row = _automation_row(db, task_id)
    if row is None:
        raise ValueError(f"Automation task not found: {task_id}")
    result = run_task(db, task_id)
    if result.get("status") == "failed" and int(result.get("attempts") or 0) >= MAX_AUTOMATION_ATTEMPTS:
        row = _automation_row(db, task_id)
        row.status = "dead_lettered"
        db.commit()
        db.refresh(row)
        result = automation_task_to_dict(row)
    return result


def requeue_automation_task(db, task_id: int, *, requested_by: str) -> dict[str, Any]:
    row = _automation_row(db, task_id)
    if row is None:
        raise ValueError(f"Automation task not found: {task_id}")
    if row.status not in {"failed", "dead_lettered"}:
        raise ValueError(f"Automation task cannot be requeued from status={row.status}")
    stored = json.loads(row.payload_json or "{}")
    job = stored.setdefault("job", {})
    history = list(job.get("requeue_history") or [])
    history.append({"requested_by": requested_by, "requested_at": datetime.now(timezone.utc).isoformat(), "prior_attempts": int(row.attempts or 0)})
    job["requeue_history"] = history
    row.payload_json = json.dumps(stored, default=str)
    row.status = "queued"
    row.error_message = None
    row.started_at = None
    row.finished_at = None
    row.available_at = datetime.now(timezone.utc)
    row.lease_owner = None
    row.lease_token = None
    row.lease_expires_at = None
    row.heartbeat_at = None
    db.commit()
    db.refresh(row)
    return automation_task_to_dict(row)


def get_automation_task(db, task_id: int) -> dict[str, Any] | None:
    row = _automation_row(db, task_id)
    return automation_task_to_dict(row) if row is not None else None


def list_automation_tasks(db, *, limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    rows = (
        db.query(AsyncTask)
        .filter(AsyncTask.task_type.like(f"{TASK_PREFIX}%"))
        .order_by(AsyncTask.queued_at.desc(), AsyncTask.id.desc())
        .limit(safe_limit)
        .all()
    )
    return [automation_task_to_dict(row) for row in rows]


def automation_task_to_dict(row: AsyncTask) -> dict[str, Any]:
    base = task_to_dict(row)
    stored = base.get("payload") or {}
    job = stored.get("job") or {}
    base["job_type"] = job.get("job_type") or str(row.task_type).removeprefix(TASK_PREFIX)
    base["dry_run"] = bool(job.get("dry_run", True))
    base["payload"] = job.get("sanitized_payload") or {}
    base["payload_redacted"] = True
    base["clinical_validation"] = False
    base["idempotency_key_hash"] = job.get("idempotency_key_hash")
    base["max_attempts"] = MAX_AUTOMATION_ATTEMPTS
    base["dead_lettered"] = row.status == "dead_lettered"
    base["requeue_history"] = job.get("requeue_history") or []
    base["available_at"] = str(row.available_at) if row.available_at else None
    base["lease_owner"] = row.lease_owner
    base["lease_expires_at"] = str(row.lease_expires_at) if row.lease_expires_at else None
    base["heartbeat_at"] = str(row.heartbeat_at) if row.heartbeat_at else None
    base["recovery_count"] = int(row.recovery_count or 0)
    base["delivery_event_id"] = row.delivery_event_id
    base["delivery_receipt_id"] = row.delivery_receipt_id
    base["delivery_receipt_status"] = row.delivery_receipt_status or "not_applicable"
    base["delivery_receipt_at"] = str(row.delivery_receipt_at) if row.delivery_receipt_at else None
    base["delivery_receipt_is_human_acknowledgement"] = False
    return base


def _automation_row(db, task_id: int) -> AsyncTask | None:
    return (
        db.query(AsyncTask)
        .filter(AsyncTask.id == task_id, AsyncTask.task_type.like(f"{TASK_PREFIX}%"))
        .first()
    )


def _find_by_idempotency_key(db, key_hash: str) -> AsyncTask | None:
    rows = (
        db.query(AsyncTask)
        .filter(AsyncTask.task_type.like(f"{TASK_PREFIX}%"))
        .order_by(AsyncTask.id.desc())
        .limit(500)
        .all()
    )
    for row in rows:
        try:
            job = (json.loads(row.payload_json or "{}") or {}).get("job") or {}
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if job.get("idempotency_key_hash") == key_hash:
            return row
    return None


__all__ = [
    "TASK_PREFIX",
    "automation_task_to_dict",
    "enqueue_automation_task",
    "get_automation_task",
    "list_automation_tasks",
    "requeue_automation_task",
    "run_automation_task",
]
