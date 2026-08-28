"""
Patient router — /me/*, /patients/*, /patient-report/*, /summary-reviews, /clinician/review-queue.

Includes the streaming chat endpoint at POST /me/chat/stream and
POST /patients/{patient_id}/chat/stream.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.api.deps import (
    get_db,
    get_patient_access_context,
    get_clinician_or_admin_context,
)
from backend.api.schemas.patient import (
    AgentFeedbackRequest,
    PatientUploadCreate,
)
from backend.models import (
    PatientUpload,
)
from backend.services.app_logging import log_app_event

from backend.api.routers.patient import _invalidate_report_cache


# No `tags=` here on purpose. This router is mounted only by
# `backend/api/routers/patient.py`, whose own router already declares
# `tags=["patient"]`, and FastAPI concatenates the parent's tags with the
# child's. Declaring the tag in both places emitted `["patient", "patient"]`
# for all 22 operations in this module, which is what the committed OpenAPI
# schema (single tag) shows was intended.
router = APIRouter()

# ─── Feedback ─────────────────────────────────────────────────────────────────


@router.post("/me/agent-feedback")
def create_my_agent_feedback(
    payload: AgentFeedbackRequest,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.agent_feedback import create_agent_response_feedback

    try:
        feedback = create_agent_response_feedback(
            db=db,
            patient_id=context.patient_id,
            chat_message_id=payload.chat_message_id,
            rating=payload.rating,
            thumbs_up=payload.thumbs_up,
            feedback_text=payload.feedback_text,
            feedback_context={"source": "patient_portal"},
        )
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="agent_feedback_error",
            actor_role=context.role,
            patient_id=context.patient_id,
            route="/me/agent-feedback",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_app_event(
        db=db,
        event_type="agent_feedback",
        actor_role=context.role,
        patient_id=context.patient_id,
        route="/me/agent-feedback",
        status="ok",
        input_payload={
            "chat_message_id": payload.chat_message_id,
            "rating": payload.rating,
            "thumbs_up": payload.thumbs_up,
        },
        output_payload={"feedback_id": feedback["id"]},
    )
    return {
        "message": "Agent feedback saved.",
        "feedback": feedback,
        "safety_note": "Feedback improves the support workflow. It is not clinical ground truth.",
    }


@router.get("/agent-feedback")
def list_agent_feedback_endpoint(
    patient_id: str | None = None,
    limit: int = 50,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.agent_feedback import build_agent_feedback_summary, list_agent_feedback

    return {
        "summary": build_agent_feedback_summary(db),
        "feedback": list_agent_feedback(db, patient_id=patient_id, limit=limit),
    }


# ─── Uploads ─────────────────────────────────────────────────────────────────


@router.get("/me/uploads")
def get_my_uploads(
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.patient_uploads import get_patient_uploads as get_safe_patient_uploads

    return {
        "patient_id": context.patient_id,
        "uploads": get_safe_patient_uploads(db, context.patient_id, limit=50),
    }


@router.get("/me/uploads/{upload_id}/content")
def get_my_upload_content(
    upload_id: int,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Return one patient-owned upload without exposing the Data tree."""
    from backend.config import UPLOAD_DIR

    upload = (
        db.query(PatientUpload)
        .filter(
            PatientUpload.id == upload_id,
            PatientUpload.patient_id == context.patient_id,
        )
        .first()
    )
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    root = Path(UPLOAD_DIR).resolve()
    path = Path(upload.local_path).resolve()
    if root != path and root not in path.parents:
        raise HTTPException(
            status_code=403, detail="Upload path is outside the protected upload directory"
        )
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Upload content not found")

    return FileResponse(
        path,
        media_type=upload.content_type or "application/octet-stream",
        filename=upload.original_filename,
        headers={"Cache-Control": "private, no-store"},
    )


@router.post("/me/uploads")
def create_my_upload(
    payload: PatientUploadCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.patient_uploads import save_patient_upload

    try:
        upload = save_patient_upload(
            db=db,
            patient_id=context.patient_id,
            upload_type=payload.upload_type,
            file_name=payload.file_name,
            content_type=payload.content_type,
            content_base64=payload.content_base64,
            notes=payload.notes,
            scan_date=payload.scan_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _invalidate_report_cache(context.patient_id)
    return {"message": "Upload saved to patient record", "upload": upload}


# ─── Clinical data entry ─────────────────────────────────────────────────────
