from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_admin_access_context, get_db
from backend.database import SessionLocal
from backend.services.background_eval_worker import ALLOWED_JOB_TYPES, BLOCKED_JOB_TYPES


router = APIRouter(prefix="/admin/automation", tags=["admin-automation"])


class AutomationJobRequest(BaseModel):
    job_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True
    run_in_background: bool = False


def _run_in_worker_session(task_id: int) -> None:
    from backend.services.automation_job_queue import run_automation_task

    db = SessionLocal()
    try:
        run_automation_task(db, task_id)
    finally:
        db.close()


@router.get("/capabilities")
def get_automation_capabilities(context=Depends(get_admin_access_context)):
    execution_enabled = str(os.getenv("NLCARE_AUTOMATION_EXECUTION_ENABLED", "")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "status": "available",
        "allowed_job_types": sorted(ALLOWED_JOB_TYPES),
        "blocked_job_types": sorted(BLOCKED_JOB_TYPES),
        "execution_enabled": execution_enabled,
        "default_mode": "dry_run",
        "phi_allowed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Admin automation is limited to redacted engineering jobs. It cannot automate medical decisions, "
            "send PHI, or establish clinical or healthcare production readiness."
        ),
    }


@router.post("/jobs", status_code=202)
def create_automation_job(
    payload: AutomationJobRequest,
    background_tasks: BackgroundTasks,
    context=Depends(get_admin_access_context),
    db=Depends(get_db),
):
    from backend.services.automation_job_queue import enqueue_automation_task

    try:
        task = enqueue_automation_task(
            db,
            job_type=payload.job_type,
            requested_by=context.role,
            payload=payload.payload,
            dry_run=payload.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload.run_in_background:
        background_tasks.add_task(_run_in_worker_session, int(task["id"]))
    return {
        "message": "Safe automation job queued.",
        "run_scheduled": payload.run_in_background,
        "task": task,
    }


@router.get("/jobs")
def get_automation_jobs(
    limit: int = Query(default=50, ge=1, le=200),
    context=Depends(get_admin_access_context),
    db=Depends(get_db),
):
    from backend.services.automation_job_queue import list_automation_tasks

    return {"tasks": list_automation_tasks(db, limit=limit), "clinical_validation": False}


@router.get("/jobs/{task_id}")
def get_automation_job(
    task_id: int,
    context=Depends(get_admin_access_context),
    db=Depends(get_db),
):
    from backend.services.automation_job_queue import get_automation_task

    task = get_automation_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Automation task not found")
    return task


@router.post("/jobs/{task_id}/run", status_code=202)
def run_automation_job(
    task_id: int,
    background_tasks: BackgroundTasks,
    context=Depends(get_admin_access_context),
    db=Depends(get_db),
):
    from backend.services.automation_job_queue import get_automation_task

    task = get_automation_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Automation task not found")
    if task["status"] not in {"queued", "failed"}:
        raise HTTPException(status_code=409, detail=f"Task cannot run from status={task['status']}")
    background_tasks.add_task(_run_in_worker_session, task_id)
    return {"message": "Automation job scheduled for background execution.", "task_id": task_id}


__all__ = ["router"]
