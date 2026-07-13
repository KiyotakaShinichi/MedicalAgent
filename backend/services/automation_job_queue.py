from __future__ import annotations

import json
from typing import Any, Mapping

from backend.models import AsyncTask
from backend.services.background_eval_worker import enqueue_job
from backend.services.task_queue import run_task, task_to_dict


TASK_PREFIX = "safe_automation:"


def enqueue_automation_task(
    db,
    *,
    job_type: str,
    requested_by: str,
    payload: Mapping[str, Any] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    job = enqueue_job(
        job_type=job_type,
        requested_by=requested_by,
        payload=payload,
        dry_run=dry_run,
    )
    if not job["accepted"]:
        raise ValueError(job["rejected_reason"] or "automation_job_rejected")
    row = AsyncTask(
        task_type=f"{TASK_PREFIX}{job_type}",
        status="queued",
        payload_json=json.dumps({"job": job}, default=str),
        created_by=requested_by,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return automation_task_to_dict(row)


def run_automation_task(db, task_id: int) -> dict[str, Any]:
    row = _automation_row(db, task_id)
    if row is None:
        raise ValueError(f"Automation task not found: {task_id}")
    return run_task(db, task_id)


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
    return base


def _automation_row(db, task_id: int) -> AsyncTask | None:
    return (
        db.query(AsyncTask)
        .filter(AsyncTask.id == task_id, AsyncTask.task_type.like(f"{TASK_PREFIX}%"))
        .first()
    )


__all__ = [
    "TASK_PREFIX",
    "automation_task_to_dict",
    "enqueue_automation_task",
    "get_automation_task",
    "list_automation_tasks",
    "run_automation_task",
]
