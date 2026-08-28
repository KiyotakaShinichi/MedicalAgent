"""
Patient router — /me/*, /patients/*, /patient-report/*, /summary-reviews, /clinician/review-queue.

Includes the streaming chat endpoint at POST /me/chat/stream and
POST /patients/{patient_id}/chat/stream.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api.deps import (
    get_db,
    get_patient_access_context,
    get_clinician_or_admin_context,
)
from backend.api.schemas.patient import (
    PatientChatRequest,
)
from backend.crud import (
    get_chat_messages,
    get_patient,
)
from backend.processing.patient_state import build_patient_state
from backend.services.app_logging import log_app_event
from backend.services.input_validation import (
    validate_chat_message,
    validation_error_payload,
)
from backend.services.support_chat_agent import handle_patient_chat
from backend.services.rag_evidence_envelope import enforce_transport_release

from backend.api.routers.patient import _invalidate_report_cache
from backend.api.routers.patient_interaction_records import router as records_router
from backend.api.routers.patient_interaction_genetics import router as genetics_router
from backend.api.routers.patient_interaction_support import router as support_router


# No `tags=` here on purpose. This router is mounted only by
# `backend/api/routers/patient.py`, whose own router already declares
# `tags=["patient"]`, and FastAPI concatenates the parent's tags with the
# child's. Declaring the tag in both places emitted `["patient", "patient"]`
# for all 22 operations in this module, which is what the committed OpenAPI
# schema (single tag) shows was intended.
router = APIRouter()

router.include_router(records_router)
router.include_router(genetics_router)


# Timeline-question, summary-review, summary-reviews list, and review-queue
# endpoints have moved to backend/api/routers/clinician_review.py. All route
# paths are preserved exactly; the only change is which router file owns them.


# ─── Chat (standard) ──────────────────────────────────────────────────────────


@router.get("/patients/{patient_id}/chat")
def get_patient_chat(
    patient_id: str,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"patient_id": patient_id, "messages": get_chat_messages(db, patient_id, limit=50)}


@router.post("/patients/{patient_id}/chat")
def chat_with_patient_agent(
    patient_id: str,
    payload: PatientChatRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        validate_chat_message(payload.message)
        result = handle_patient_chat(db, patient_id, payload.message)
        result = enforce_transport_release(result, query=payload.message)
        if result.get("saved_actions"):
            _invalidate_report_cache(patient_id)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="chat_error",
            patient_id=patient_id,
            route="/patients/{patient_id}/chat",
            status="error",
            input_payload={"message": payload.message},
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/patients/{patient_id}/chat"),
        ) from exc
    return result


@router.get("/me/chat")
def get_my_patient_chat(
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    return {
        "patient_id": context.patient_id,
        "messages": get_chat_messages(db, context.patient_id, limit=50),
    }


@router.post("/me/chat")
def chat_with_my_patient_agent(
    payload: PatientChatRequest,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    try:
        validate_chat_message(payload.message)
        result = handle_patient_chat(db, context.patient_id, payload.message)
        result = enforce_transport_release(result, query=payload.message)
        if result.get("saved_actions"):
            _invalidate_report_cache(context.patient_id)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="chat_error",
            actor_role=context.role,
            patient_id=context.patient_id,
            route="/me/chat",
            status="error",
            input_payload={"message": payload.message},
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=400, detail=validation_error_payload(exc, route="/me/chat")
        ) from exc
    return result


@router.delete("/me/record-write-actions/{audit_id}")
def undo_my_confirmed_record_write(
    audit_id: int,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Undo one patient-confirmed support-chat write.

    The underlying portal row is removed while its provenance audit remains
    marked as undone. This is an engineering traceability feature, not a
    clinical record-retention claim.
    """
    from backend.services.confirmed_record_write import undo_record_write

    try:
        action = undo_record_write(db, context.patient_id, audit_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _invalidate_report_cache(context.patient_id)
    return {"message": action["message"], "action": action}


# ─── Streaming chat (SSE) ─────────────────────────────────────────────────────


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _chunk_text(text: str, chunk_size: int = 42):
    """Yield small display chunks after all safety/output guardrails have passed."""
    text = text or ""
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def _stream_agent_pipeline(
    db: Session, patient_id: str, message: str, *, persist_support_chat: bool = False
):
    """
    Generator that streams pipeline stage events then the final answer as SSE.

    Events emitted (all prefixed event: <name>):
      pipeline_stage  — progress label with stage name
      answer          — final answer payload (reply, citations, intent, safety_level)
      error           — error detail if pipeline fails
      done            — stream termination signal

    Safety rules preserved:
      - Input validation runs before any streaming begins.
      - Output guardrails run before streaming the answer.
      - Chain-of-thought and internal guardrail details are never emitted.
      - Patient scoping is enforced by the route's access context (caller's responsibility).
    """
    from backend.services.agent_rag import run_patient_agent_pipeline

    try:
        validate_chat_message(message)
    except ValueError as exc:
        yield _sse_event("error", {"error": str(exc), "code": "validation_error"})
        yield _sse_event("done", {})
        return

    yield _sse_event(
        "stream_mode",
        {
            "mode": "post_guardrail_display_stream",
            "reason": "Patient-facing medical replies are chunked only after safety/output checks pass.",
        },
    )
    yield _sse_event(
        "streaming_authorization",
        {
            "status": "blocked_pending_validation",
            "evidence_content_emitted": False,
        },
    )
    yield _sse_event("pipeline_stage", {"stage": "safety_gate", "label": "Checking safety gate…"})

    yield _sse_event("pipeline_stage", {"stage": "intent_routing", "label": "Routing intent..."})

    try:
        if persist_support_chat:
            result = handle_patient_chat(db, patient_id, message)
            if result.get("saved_actions"):
                _invalidate_report_cache(patient_id)
        else:
            patient_context = (
                build_patient_state(db, patient_id)
                if db is not None
                else {"patient_id": patient_id}
            )
            result = run_patient_agent_pipeline(
                db=db,
                patient_id=patient_id,
                query=message,
                patient_context=patient_context,
                fallback_response=(
                    "I can provide monitoring support only. I cannot diagnose, predict outcomes, "
                    "or recommend treatment changes. Please review concerns with the oncology care team."
                ),
            )
    except Exception:
        yield _sse_event(
            "error", {"error": "Agent pipeline failed. Please try again.", "code": "pipeline_error"}
        )
        yield _sse_event("done", {})
        return

    # Final transport boundary: no answer token is emitted before the exact
    # reply/envelope pair has been re-authorized.
    result = enforce_transport_release(result, query=message)
    agent_pipeline = result.get("agent_pipeline") or {}
    release_authorization = (
        result.get("release_authorization") or agent_pipeline.get("release_authorization") or {}
    )
    yield _sse_event(
        "streaming_authorization",
        {
            "status": "authorized_safe_payload",
            "disposition": release_authorization.get("disposition"),
            "evidence_content_emitted": False,
        },
    )

    yield _sse_event("pipeline_stage", {"stage": "intent_routing", "label": "Routing intent…"})
    yield _sse_event("pipeline_stage", {"stage": "retrieval", "label": "Retrieving context…"})
    yield _sse_event("pipeline_stage", {"stage": "generation", "label": "Generating response…"})

    agent_pipeline = result.get("agent_pipeline") or {}
    citations = result.get("citations") or agent_pipeline.get("citations") or []
    reply = result.get("reply") or ""
    for chunk in _chunk_text(reply):
        yield _sse_event("answer_delta", {"text": chunk, "mode": "post_guardrail_display_stream"})

    answer_payload = {
        "reply": reply,
        "citations": [
            {
                "id": c.get("id"),
                "title": c.get("title"),
                "source_name": c.get("source_name"),
                "source_url": c.get("source_url"),
            }
            for c in citations
            if isinstance(c, dict)
        ],
        "intent": result.get("intent") or agent_pipeline.get("intent"),
        "safety_level": (result.get("safety") or agent_pipeline.get("safety") or {}).get("level"),
        "cache_status": (result.get("cache") or agent_pipeline.get("cache") or {}).get("status"),
        "saved_actions": result.get("saved_actions") or [],
        "assistant_message_id": result.get("assistant_message_id"),
        "evidence_disposition": (
            result.get("release_authorization") or agent_pipeline.get("release_authorization") or {}
        ).get("disposition"),
    }
    yield _sse_event("answer", answer_payload)
    yield _sse_event("done", {})


@router.post("/me/chat/stream")
def stream_my_patient_chat(
    payload: PatientChatRequest,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """
    Streaming SSE chat for the patient portal.
    Emits pipeline_stage events then a final answer event.
    Preserves all safety guardrails; chain-of-thought is never exposed.
    """
    return StreamingResponse(
        _stream_agent_pipeline(db, context.patient_id, payload.message, persist_support_chat=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/patients/{patient_id}/chat/stream")
def stream_patient_chat(
    patient_id: str,
    payload: PatientChatRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    """
    Streaming SSE chat for clinician/admin view of a patient.
    Preserves all safety guardrails and patient scoping.
    """
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return StreamingResponse(
        _stream_agent_pipeline(db, patient_id, payload.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


router.include_router(support_router)
