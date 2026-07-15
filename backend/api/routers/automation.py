from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.api.deps import get_admin_access_context, get_db
from backend.database import SessionLocal
from backend.services.background_eval_worker import ALLOWED_JOB_TYPES, BLOCKED_JOB_TYPES
from backend.services.high_risk_conversation_alerts import (
    process_due_alert_deliveries,
    record_delivery_receipt,
    serialize_alert,
)
from backend.services.n8n_webhook_dispatcher import validate_signed_receipt


router = APIRouter(prefix="/admin/automation", tags=["admin-automation"])


class AutomationJobRequest(BaseModel):
    job_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True
    run_in_background: bool = False


@router.post("/delivery-receipts")
async def receive_delivery_receipt(
    request: Request,
    x_nlcare_receipt_signature: str = Header(default=""),
    db=Depends(get_db),
):
    """Receive a redacted, signed channel receipt; this is not clinician acknowledgement."""
    secret = str(os.getenv("N8N_WEBHOOK_SIGNING_SECRET") or "")
    if not secret:
        raise HTTPException(status_code=503, detail="Delivery receipt verification is not configured")
    body = await request.body()
    result = validate_signed_receipt(
        body=body,
        signature=x_nlcare_receipt_signature,
        secret=secret,
    )
    if not result.get("valid"):
        raise HTTPException(status_code=401, detail=f"Invalid delivery receipt: {result.get('reason')}")
    receipt = result["receipt"]
    try:
        alert = record_delivery_receipt(
            db,
            event_id=receipt["event_id"],
            receipt_id=receipt["receipt_id"],
            delivery_status=receipt["delivery_status"],
            occurred_at=datetime.fromisoformat(str(receipt["occurred_at"]).replace("Z", "+00:00")),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.commit()
    db.refresh(alert)
    return {
        "status": "receipt_recorded",
        "alert": serialize_alert(alert),
        "clinical_validation": False,
        "claim_boundary": "A channel receipt is not proof of clinician review, contact, or clinical action.",
    }


@router.post("/alert-deliveries/process-due")
def process_due_alert_delivery_retries(
    context=Depends(get_admin_access_context),
    db=Depends(get_db),
):
    result = process_due_alert_deliveries(db)
    db.commit()
    return result


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
