"""
Patient router — /me/*, /patients/*, /patient-report/*, /summary-reviews, /clinician/review-queue.

Includes the streaming chat endpoint at POST /me/chat/stream and
POST /patients/{patient_id}/chat/stream.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from backend.api.deps import (
    get_db,
    get_patient_access_context,
    get_clinician_or_admin_context,
)
from backend.api.schemas.patient import (
    AgentFeedbackRequest,
    BiomarkerRecordCreate,
    FamilyHistoryCreate,
    GeneticReviewCreate,
    GeneticTestRecordCreate,
    MyImagingReportCreate,
    MyLabCreate,
    MyMedicationCreate,
    MySymptomCreate,
    MyTreatmentCreate,
    PatientChatRequest,
    PatientUploadCreate,
    TumorMarkerRecordCreate,
)
from backend.crud import (
    get_chat_messages,
    get_patient,
)
from backend.models import (
    ImagingReport,
    LabResult,
    MedicationLog,
    PatientUpload,
    SymptomReport,
    Treatment,
)
from backend.processing.patient_state import build_patient_state
from backend.services.app_logging import log_app_event
from backend.services.input_validation import (
    validate_cbc_values,
    validate_chat_message,
    validate_imaging_report_payload,
    validate_symptom_payload,
    validate_treatment_payload,
    validation_error_payload,
)
from backend.services.ctcae_mapping import map_symptom_to_ctcae_review_hint
from backend.services.lab_reference_context import build_cbc_reference_context
from backend.services.support_chat_agent import handle_patient_chat
from backend.services.rag_evidence_envelope import enforce_transport_release

from backend.api.routers.patient import _invalidate_report_cache


router = APIRouter(tags=["patient"])

@router.post("/me/symptoms")
def add_my_symptom(
    payload: MySymptomCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped symptom save (manual-entry form).

    Mirrors the clinician-side :func:`add_symptom_report` route but trusts
    the bearer-token's patient_id rather than reading it from the URL.  The
    new ``duration`` and ``urgent_flag`` fields are folded into the notes
    column so the existing schema does not need to change — the clinician
    review queue picks the urgent tag up via its existing text scan.
    """
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    composed_notes_parts: list[str] = []
    if payload.urgent_flag:
        composed_notes_parts.append("[urgent flag set by patient]")
    if payload.duration:
        composed_notes_parts.append(f"Duration: {payload.duration.strip()}")
    if payload.notes:
        composed_notes_parts.append(payload.notes.strip())
    composed_notes = " | ".join(composed_notes_parts) or None
    ctcae_hint = map_symptom_to_ctcae_review_hint(
        symptom=payload.symptom,
        severity=payload.severity,
        urgent_flag=payload.urgent_flag,
        notes=composed_notes,
    )

    try:
        validation_warnings = validate_symptom_payload(payload.symptom, payload.severity, composed_notes)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/me/symptoms",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/symptoms"),
        ) from exc

    symptom = SymptomReport(
        patient_id=patient_id,
        date=payload.date,
        symptom=payload.symptom,
        severity=payload.severity,
        notes=composed_notes,
    )
    db.add(symptom)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/me/symptoms",
        status="ok",
        input_payload=payload.dict(),
        output_payload={
            "symptom_id": symptom.id,
            "warning_count": len(validation_warnings),
            "urgent_flag": payload.urgent_flag,
            "ctcae_hint": ctcae_hint,
        },
    )
    return {
        "message": "Symptom logged to your patient record.",
        "symptom_id": symptom.id,
        "validation_warnings": validation_warnings,
        "urgent_flag": payload.urgent_flag,
        "ctcae_review_hint": ctcae_hint,
        "safety_note": (
            "This record is for monitoring only and does not replace clinician judgement. "
            "Urgent symptoms should be discussed with your care team."
        ),
    }


@router.post("/me/labs")
def add_my_lab(
    payload: MyLabCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped CBC save.  Mirrors :func:`add_lab_result` but trusts
    the bearer-token's patient_id rather than the URL.  ``anc`` and
    ``lab_source``/``notes`` are folded into ``source_note`` to avoid a
    schema migration on the LabResult table."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validation_warnings = validate_cbc_values(payload.wbc, payload.hemoglobin, payload.platelets)
    except ValueError as exc:
        log_app_event(
            db=db, event_type="validation_error", patient_id=patient_id,
            route="/me/labs", status="error",
            input_payload=payload.dict(), error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/labs"),
        ) from exc

    note_parts: list[str] = []
    if payload.anc is not None:
        note_parts.append(f"ANC={payload.anc:g} K/uL")
    if payload.lab_source:
        note_parts.append(f"Source: {payload.lab_source.strip()}")
    if payload.notes:
        note_parts.append(payload.notes.strip())
    source_note = " | ".join(note_parts) or None
    reference_context = build_cbc_reference_context(
        wbc=payload.wbc,
        hemoglobin=payload.hemoglobin,
        platelets=payload.platelets,
    )

    lab = LabResult(
        patient_id=patient_id,
        date=payload.date,
        wbc=payload.wbc,
        hemoglobin=payload.hemoglobin,
        platelets=payload.platelets,
        source="manual",
        source_note=source_note,
    )
    db.add(lab)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db, event_type="patient_input", patient_id=patient_id,
        route="/me/labs", status="ok",
        input_payload=payload.dict(),
        output_payload={
            "lab_id": lab.id,
            "warning_count": len(validation_warnings),
            "reference_context": reference_context,
        },
    )
    return {
        "message": "Lab values logged to your patient record.",
        "lab_id": lab.id,
        "validation_warnings": validation_warnings,
        "reference_context": reference_context,
        "safety_note": (
            "Lab entries here are for tracking. Reference ranges shown in the portal "
            "are general guides — your clinical team uses lab-specific ranges."
        ),
    }


@router.post("/me/imaging-reports")
def add_my_imaging_report(
    payload: MyImagingReportCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped imaging save.  Either ``findings`` or ``impression``
    must be present; otherwise the entry is rejected so we never store an
    empty imaging row."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    findings = (payload.findings or "").strip()
    impression = (payload.impression or "").strip()
    if not findings and not impression:
        raise HTTPException(
            status_code=400,
            detail="Please paste either the report findings or the impression text.",
        )
    body_site = (payload.body_site or "Breast").strip() or "Breast"
    report_type = (payload.report_type or "Patient-entered report").strip()

    try:
        validation_warnings = validate_imaging_report_payload(
            payload.modality, report_type, findings or impression, impression or findings, body_site,
        )
    except ValueError as exc:
        log_app_event(
            db=db, event_type="validation_error", patient_id=patient_id,
            route="/me/imaging-reports", status="error",
            input_payload=payload.dict(), error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/imaging-reports"),
        ) from exc

    composed_impression = impression
    if payload.notes:
        composed_impression = f"{composed_impression}\n[Patient note] {payload.notes.strip()}".strip()

    report = ImagingReport(
        patient_id=patient_id,
        date=payload.date,
        modality=payload.modality,
        report_type=report_type,
        body_site=body_site,
        findings=findings,
        impression=composed_impression,
    )
    db.add(report)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db, event_type="patient_input", patient_id=patient_id,
        route="/me/imaging-reports", status="ok",
        # Redact the free-text fields from the audit log — keep only metadata.
        input_payload={
            **payload.dict(),
            "findings": "[redacted report text]",
            "impression": "[redacted report text]",
        },
        output_payload={"imaging_report_id": report.id, "warning_count": len(validation_warnings)},
    )
    return {
        "message": "Imaging report logged to your patient record.",
        "imaging_report_id": report.id,
        "modality": payload.modality,
        "validation_warnings": validation_warnings,
        "safety_note": (
            "Imaging text is recorded as-is; this system does not interpret images. "
            "Your care team makes any clinical decisions."
        ),
    }


@router.post("/me/medications")
def add_my_medication(
    payload: MyMedicationCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped medication save.  Stored in the MedicationLog table
    (the same table the chat agent's save_medication path writes to)."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    medication = payload.medication.strip()
    if not medication:
        raise HTTPException(status_code=400, detail="Medication name is required.")
    if len(medication) > 120:
        raise HTTPException(status_code=400, detail="Medication name must be 120 characters or less.")

    note_parts: list[str] = []
    if payload.side_effects:
        note_parts.append(f"Side effects: {payload.side_effects.strip()}")
    if payload.notes:
        note_parts.append(payload.notes.strip())
    notes = " | ".join(note_parts) or None
    from backend.services.medication_interactions import check_medication_interactions

    current_medications = [
        row.medication
        for row in db.query(MedicationLog)
        .filter(MedicationLog.patient_id == patient_id)
        .order_by(MedicationLog.date.desc(), MedicationLog.id.desc())
        .limit(25)
        .all()
    ]
    interaction_check = check_medication_interactions(
        medication,
        current_medications=current_medications,
        notes=notes,
    )

    med = MedicationLog(
        patient_id=patient_id,
        date=payload.date,
        medication=medication,
        dose=(payload.dose or "").strip() or None,
        frequency=(payload.frequency or "").strip() or None,
        notes=notes,
    )
    db.add(med)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db, event_type="patient_input", patient_id=patient_id,
        route="/me/medications", status="ok",
        input_payload=payload.dict(),
        output_payload={"medication_id": med.id},
    )
    return {
        "message": "Medication logged to your patient record.",
        "medication_id": med.id,
        "safety_note": (
            "Use this to track what you are taking. Dose changes must be agreed with your care team."
        ),
        "interaction_check": interaction_check,
    }


@router.post("/me/treatments")
def add_my_treatment(
    payload: MyTreatmentCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Patient-scoped treatment-cycle note.  Cycle defaults to 0 when the
    patient doesn't remember the number; the existing Treatment row schema
    requires an integer."""
    patient_id = context.patient_id
    if not patient_id:
        raise HTTPException(status_code=403, detail="Patient context required.")
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    drug = payload.drug.strip()
    if not drug:
        raise HTTPException(status_code=400, detail="Treatment/drug name is required.")
    cycle = payload.cycle if payload.cycle is not None else 0
    try:
        validation_warnings = validate_treatment_payload(cycle, drug)
    except ValueError as exc:
        log_app_event(
            db=db, event_type="validation_error", patient_id=patient_id,
            route="/me/treatments", status="error",
            input_payload=payload.dict(), error_message=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=validation_error_payload(exc, route="/me/treatments"),
        ) from exc

    treatment = Treatment(
        patient_id=patient_id, date=payload.date, cycle=cycle, drug=drug,
    )
    db.add(treatment)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(
        db=db, event_type="patient_input", patient_id=patient_id,
        route="/me/treatments", status="ok",
        input_payload=payload.dict(),
        output_payload={
            "treatment_id": treatment.id,
            "warning_count": len(validation_warnings),
        },
    )
    return {
        "message": "Treatment note logged to your patient record.",
        "treatment_id": treatment.id,
        "validation_warnings": validation_warnings,
        "safety_note": (
            "This is a tracking note. Treatment decisions stay with your oncology team."
        ),
    }


@router.post("/me/family-history")
def add_my_family_history(
    payload: FamilyHistoryCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import GENETIC_BOUNDARY_NOTE, create_family_history_record

    record = create_family_history_record(db, context.patient_id, payload.dict(), actor_role=context.role)
    _invalidate_report_cache(context.patient_id)
    return {"message": "Family history saved for review.", "record": record, "safety_note": GENETIC_BOUNDARY_NOTE}


@router.post("/me/genetic-test-records")
def add_my_genetic_test_record(
    payload: GeneticTestRecordCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import GENETIC_BOUNDARY_NOTE, create_genetic_test_record

    record = create_genetic_test_record(db, context.patient_id, payload.dict(), actor_role=context.role)
    _invalidate_report_cache(context.patient_id)
    return {"message": "Genetic test record saved for genetics/oncology review.", "record": record, "safety_note": GENETIC_BOUNDARY_NOTE}


@router.post("/me/biomarker-records")
def add_my_biomarker_record(
    payload: BiomarkerRecordCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import GENETIC_BOUNDARY_NOTE, create_biomarker_record

    record = create_biomarker_record(db, context.patient_id, payload.dict(), actor_role=context.role)
    _invalidate_report_cache(context.patient_id)
    return {"message": "Biomarker/pathology record saved for review.", "record": record, "safety_note": GENETIC_BOUNDARY_NOTE}


@router.post("/me/tumor-marker-records")
def add_my_tumor_marker_record(
    payload: TumorMarkerRecordCreate,
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import GENETIC_BOUNDARY_NOTE, create_tumor_marker_record

    record = create_tumor_marker_record(db, context.patient_id, payload.dict(), actor_role=context.role)
    _invalidate_report_cache(context.patient_id)
    return {"message": "Tumor marker record saved for review.", "record": record, "safety_note": GENETIC_BOUNDARY_NOTE}


@router.post("/patients/{patient_id}/genetic-counseling-review")
def save_genetic_counseling_review(
    patient_id: str,
    payload: GeneticReviewCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import create_genetic_review_note

    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    record = create_genetic_review_note(db, patient_id, context.role, payload.decision, payload.notes)
    _invalidate_report_cache(patient_id)
    return {"message": "Genetic counseling readiness review saved.", "review": record}


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
            db=db, event_type="chat_error", patient_id=patient_id,
            route="/patients/{patient_id}/chat", status="error",
            input_payload={"message": payload.message}, error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/chat")) from exc
    return result


@router.get("/me/chat")
def get_my_patient_chat(
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    return {"patient_id": context.patient_id, "messages": get_chat_messages(db, context.patient_id, limit=50)}


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
            db=db, event_type="chat_error", actor_role=context.role,
            patient_id=context.patient_id, route="/me/chat", status="error",
            input_payload={"message": payload.message}, error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/me/chat")) from exc
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
        yield text[index:index + chunk_size]


def _stream_agent_pipeline(db: Session, patient_id: str, message: str, *, persist_support_chat: bool = False):
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

    yield _sse_event("stream_mode", {
        "mode": "post_guardrail_display_stream",
        "reason": "Patient-facing medical replies are chunked only after safety/output checks pass.",
    })
    yield _sse_event("streaming_authorization", {
        "status": "blocked_pending_validation",
        "evidence_content_emitted": False,
    })
    yield _sse_event("pipeline_stage", {"stage": "safety_gate", "label": "Checking safety gate…"})

    yield _sse_event("pipeline_stage", {"stage": "intent_routing", "label": "Routing intent..."})

    try:
        if persist_support_chat:
            result = handle_patient_chat(db, patient_id, message)
            if result.get("saved_actions"):
                _invalidate_report_cache(patient_id)
        else:
            patient_context = build_patient_state(db, patient_id) if db is not None else {"patient_id": patient_id}
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
        yield _sse_event("error", {"error": "Agent pipeline failed. Please try again.", "code": "pipeline_error"})
        yield _sse_event("done", {})
        return

    # Final transport boundary: no answer token is emitted before the exact
    # reply/envelope pair has been re-authorized.
    result = enforce_transport_release(result, query=message)
    agent_pipeline = result.get("agent_pipeline") or {}
    release_authorization = (
        result.get("release_authorization")
        or agent_pipeline.get("release_authorization")
        or {}
    )
    yield _sse_event("streaming_authorization", {
        "status": "authorized_safe_payload",
        "disposition": release_authorization.get("disposition"),
        "evidence_content_emitted": False,
    })

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
            {"id": c.get("id"), "title": c.get("title"), "source_name": c.get("source_name"), "source_url": c.get("source_url")}
            for c in citations
            if isinstance(c, dict)
        ],
        "intent": result.get("intent") or agent_pipeline.get("intent"),
        "safety_level": (result.get("safety") or agent_pipeline.get("safety") or {}).get("level"),
        "cache_status": (result.get("cache") or agent_pipeline.get("cache") or {}).get("status"),
        "saved_actions": result.get("saved_actions") or [],
        "assistant_message_id": result.get("assistant_message_id"),
        "evidence_disposition": (
            result.get("release_authorization")
            or agent_pipeline.get("release_authorization")
            or {}
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
            db=db, event_type="agent_feedback_error", actor_role=context.role,
            patient_id=context.patient_id, route="/me/agent-feedback",
            status="error", input_payload=payload.dict(), error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_app_event(
        db=db, event_type="agent_feedback", actor_role=context.role,
        patient_id=context.patient_id, route="/me/agent-feedback", status="ok",
        input_payload={"chat_message_id": payload.chat_message_id, "rating": payload.rating, "thumbs_up": payload.thumbs_up},
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
        raise HTTPException(status_code=403, detail="Upload path is outside the protected upload directory")
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
