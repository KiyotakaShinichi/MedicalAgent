"""Database-leased worker for redacted NLCare engineering automation jobs."""

from __future__ import annotations

import json
import os
import socket
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import or_

from backend.database import SessionLocal
from backend.models import AsyncTask
from backend.services.automation_job_queue import (
    MAX_AUTOMATION_ATTEMPTS,
    TASK_PREFIX,
    automation_task_to_dict,
)
from backend.services.background_eval_worker import execute_job


CLAIM_BOUNDARY = (
    "The durable worker executes redacted engineering automation only. Delivery receipts prove channel state, "
    "not clinician acknowledgement, clinical review, patient contact, or clinical action."
)
DEFAULT_LEASE_SECONDS = 120
RETRY_DELAYS_SECONDS = (5, 30, 120)


def recover_expired_automation_leases(db, *, now: datetime | None = None) -> dict[str, int]:
    current = now or _utc_now()
    rows = (
        db.query(AsyncTask)
        .filter(
            AsyncTask.task_type.like(f"{TASK_PREFIX}%"),
            AsyncTask.status == "running",
            AsyncTask.lease_expires_at.is_not(None),
            AsyncTask.lease_expires_at <= current,
        )
        .all()
    )
    recovered = 0
    dead_lettered = 0
    for row in rows:
        row.recovery_count = int(row.recovery_count or 0) + 1
        row.lease_owner = None
        row.lease_token = None
        row.lease_expires_at = None
        row.heartbeat_at = None
        row.error_message = "worker_lease_expired"
        if int(row.attempts or 0) >= MAX_AUTOMATION_ATTEMPTS:
            row.status = "dead_lettered"
            row.finished_at = current
            dead_lettered += 1
        else:
            row.status = "queued"
            row.available_at = current
            recovered += 1
    if rows:
        db.commit()
    return {"recovered": recovered, "dead_lettered": dead_lettered}


def claim_next_automation_task(
    db,
    *,
    worker_id: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    current = now or _utc_now()
    recover_expired_automation_leases(db, now=current)
    eligible = or_(AsyncTask.available_at.is_(None), AsyncTask.available_at <= current)
    candidates = (
        db.query(AsyncTask)
        .filter(
            AsyncTask.task_type.like(f"{TASK_PREFIX}%"),
            AsyncTask.status.in_(("queued", "failed")),
            AsyncTask.attempts < MAX_AUTOMATION_ATTEMPTS,
            eligible,
        )
        .order_by(AsyncTask.queued_at.asc(), AsyncTask.id.asc())
        .limit(8)
        .all()
    )
    for candidate in candidates:
        token = str(uuid4())
        prior_status = candidate.status
        updated = (
            db.query(AsyncTask)
            .filter(AsyncTask.id == candidate.id, AsyncTask.status == prior_status)
            .update(
                {
                    AsyncTask.status: "running",
                    AsyncTask.started_at: current,
                    AsyncTask.finished_at: None,
                    AsyncTask.attempts: int(candidate.attempts or 0) + 1,
                    AsyncTask.lease_owner: worker_id,
                    AsyncTask.lease_token: token,
                    AsyncTask.lease_expires_at: current + timedelta(seconds=max(5, lease_seconds)),
                    AsyncTask.heartbeat_at: current,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if updated:
            row = db.query(AsyncTask).filter(AsyncTask.id == candidate.id).one()
            task = automation_task_to_dict(row)
            task["lease_token"] = token
            task["claim_boundary"] = CLAIM_BOUNDARY
            return task
    return None


def heartbeat_automation_task(
    db,
    *,
    task_id: int,
    worker_id: str,
    lease_token: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
) -> bool:
    current = now or _utc_now()
    updated = (
        db.query(AsyncTask)
        .filter(
            AsyncTask.id == task_id,
            AsyncTask.status == "running",
            AsyncTask.lease_owner == worker_id,
            AsyncTask.lease_token == lease_token,
        )
        .update(
            {
                AsyncTask.heartbeat_at: current,
                AsyncTask.lease_expires_at: current + timedelta(seconds=max(5, lease_seconds)),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return bool(updated)


def execute_claimed_automation_task(
    db,
    *,
    task_id: int,
    worker_id: str,
    lease_token: str,
    env: dict[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    row = (
        db.query(AsyncTask)
        .filter(
            AsyncTask.id == task_id,
            AsyncTask.task_type.like(f"{TASK_PREFIX}%"),
            AsyncTask.status == "running",
            AsyncTask.lease_owner == worker_id,
            AsyncTask.lease_token == lease_token,
        )
        .first()
    )
    if row is None:
        raise ValueError("Automation lease is missing, expired, or owned by another worker")
    current = now or _utc_now()
    stored = _json_loads(row.payload_json) or {}
    job = stored.get("job")
    if not isinstance(job, dict):
        return _finish_failure(db, row, ValueError("validated job envelope missing"), current)
    try:
        result = execute_job(job, env=env)
    except Exception as exc:  # worker must persist every terminal attempt
        return _finish_failure(db, row, exc, current)

    row.status = "completed"
    row.result_json = json.dumps(result, default=str)
    row.error_message = None
    row.finished_at = current
    row.available_at = None
    row.delivery_event_id = result.get("event_id")
    if result.get("sent") and row.delivery_event_id:
        row.delivery_receipt_status = "awaiting_receipt"
    elif result.get("status") == "disabled_dry_run":
        row.delivery_receipt_status = "dispatch_disabled"
    else:
        row.delivery_receipt_status = "not_applicable"
    _clear_lease(row)
    db.commit()
    db.refresh(row)
    return automation_task_to_dict(row)


def record_automation_delivery_receipt(
    db,
    *,
    event_id: str,
    receipt_id: str,
    delivery_status: str,
    occurred_at: datetime,
) -> AsyncTask:
    row = db.query(AsyncTask).filter(AsyncTask.delivery_event_id == event_id).first()
    if row is None:
        raise LookupError(f"No automation task found for delivery event_id={event_id}")
    if row.delivery_receipt_id:
        if row.delivery_receipt_id == receipt_id and row.delivery_receipt_status == delivery_status:
            return row
        raise ValueError("Automation delivery event already has a different receipt")
    row.delivery_receipt_id = receipt_id
    row.delivery_receipt_status = delivery_status
    row.delivery_receipt_at = occurred_at
    db.flush()
    return row


def run_automation_worker_once(
    *,
    session_factory: Callable[[], Any] = SessionLocal,
    worker_id: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    env: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    identity = worker_id or default_worker_id()
    claim_db = session_factory()
    try:
        claimed = claim_next_automation_task(claim_db, worker_id=identity, lease_seconds=lease_seconds)
    finally:
        claim_db.close()
    if claimed is None:
        return None

    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "session_factory": session_factory,
            "stop": stop,
            "task_id": int(claimed["id"]),
            "worker_id": identity,
            "lease_token": str(claimed["lease_token"]),
            "lease_seconds": lease_seconds,
        },
        daemon=True,
    )
    heartbeat.start()
    execution_db = session_factory()
    try:
        return execute_claimed_automation_task(
            execution_db,
            task_id=int(claimed["id"]),
            worker_id=identity,
            lease_token=str(claimed["lease_token"]),
            env=env,
        )
    finally:
        stop.set()
        heartbeat.join(timeout=2)
        execution_db.close()


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _heartbeat_loop(*, session_factory, stop, task_id, worker_id, lease_token, lease_seconds) -> None:
    interval = max(1.0, lease_seconds / 3)
    while not stop.wait(interval):
        db = session_factory()
        try:
            if not heartbeat_automation_task(
                db,
                task_id=task_id,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_seconds=lease_seconds,
            ):
                return
        finally:
            db.close()


def _finish_failure(db, row: AsyncTask, exc: Exception, now: datetime) -> dict[str, Any]:
    row.error_message = f"{type(exc).__name__}: {exc}"
    row.result_json = None
    if int(row.attempts or 0) >= MAX_AUTOMATION_ATTEMPTS:
        row.status = "dead_lettered"
        row.finished_at = now
        row.available_at = None
    else:
        row.status = "queued"
        delay = RETRY_DELAYS_SECONDS[min(int(row.attempts or 1) - 1, len(RETRY_DELAYS_SECONDS) - 1)]
        row.available_at = now + timedelta(seconds=delay)
        row.finished_at = None
    _clear_lease(row)
    db.commit()
    db.refresh(row)
    return automation_task_to_dict(row)


def _clear_lease(row: AsyncTask) -> None:
    row.lease_owner = None
    row.lease_token = None
    row.lease_expires_at = None
    row.heartbeat_at = None


def _json_loads(value: str | None) -> Any:
    try:
        return json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "CLAIM_BOUNDARY",
    "claim_next_automation_task",
    "default_worker_id",
    "execute_claimed_automation_task",
    "heartbeat_automation_task",
    "record_automation_delivery_receipt",
    "recover_expired_automation_leases",
    "run_automation_worker_once",
]
