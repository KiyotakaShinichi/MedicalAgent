"""
Patient router — /me/*, /patients/*, /patient-report/*, /summary-reviews, /clinician/review-queue.

Includes the streaming chat endpoint at POST /me/chat/stream and
POST /patients/{patient_id}/chat/stream.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import (
    get_db,
    get_patient_access_context,
    get_clinician_or_admin_context,
)
from backend.crud import (
    get_all_patients,
    get_breast_cancer_profile,
    get_chat_messages,
    get_clinical_interventions,
    get_ct_reports_df,
    get_imaging_reports_df,
    get_labs_df,
    get_medication_logs,
    get_mri_registry,
    get_mri_series_index,
    get_patient,
    get_patient_uploads,
    get_symptoms_df,
    get_treatment_outcome,
    get_treatments_df,
)
from backend.models import (
    BreastCancerProfile,
    CTReport,
    ImagingReport,
    LabResult,
    MRIFileRegistry,
    Patient,
    SymptomReport,
    Treatment,
)
from backend.processing.radiology_analysis import analyze_breast_imaging_reports
from backend.processing.patient_state import build_patient_state
from backend.processing.risk_engine import (
    detect_clinical_rule_risks,
    detect_risks,
    detect_symptom_risks,
    detect_trend_risk,
)
from backend.processing.timeline import build_clinical_timeline
from backend.processing.treatment_analysis import align_labs_with_treatment
from backend.processing.trend_analysis import analyze_labs
from backend.processing.clinical_summary import generate_clinical_summary
from backend.reports.patient_report import build_patient_report
from backend.services.app_logging import log_app_event
from backend.services.input_validation import (
    validate_cbc_values,
    validate_chat_message,
    validate_imaging_report_payload,
    validate_patient_payload,
    validate_symptom_payload,
    validate_treatment_payload,
    validation_error_payload,
)
from backend.services.multimodal_fusion import build_multimodal_assessment
from backend.services.patient_timeline_summary import build_patient_timeline_risk_summary
from backend.services.support_chat_agent import handle_patient_chat
from backend.services.timeline_intelligence import answer_timeline_question, build_timeline_intelligence
from backend.services.data_availability import build_data_availability

router = APIRouter(tags=["patient"])


# ─── Request models ───────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    id: str
    name: str
    diagnosis: str | None = None
    cancer_stage: str | None = None
    er_status: str | None = None
    pr_status: str | None = None
    her2_status: str | None = None
    molecular_subtype: str | None = None
    treatment_intent: str | None = None
    menopausal_status: str | None = None


class LabCreate(BaseModel):
    date: date
    wbc: float
    hemoglobin: float
    platelets: float
    source: str | None = "manual"
    source_note: str | None = None


class TreatmentCreate(BaseModel):
    date: date
    cycle: int
    drug: str


class SymptomCreate(BaseModel):
    date: date
    symptom: str
    severity: int
    notes: str | None = None


class ImagingReportCreate(BaseModel):
    date: date
    modality: str
    report_type: str
    body_site: str | None = "Breast"
    findings: str
    impression: str


class CTReportCreate(BaseModel):
    date: date
    report_type: str
    findings: str
    impression: str


class MRIRegistryCreate(BaseModel):
    scan_date: date | None = None
    modality: str = "Breast MRI"
    series_description: str | None = None
    local_path: str
    notes: str | None = None


class PatientChatRequest(BaseModel):
    message: str


class AgentFeedbackRequest(BaseModel):
    chat_message_id: int | None = None
    rating: int
    thumbs_up: bool | None = None
    feedback_text: str | None = None


class TimelineQuestionRequest(BaseModel):
    question: str


class ClinicianSummaryReviewRequest(BaseModel):
    decision: str
    clinician_notes: str | None = None
    edited_patient_summary: str | None = None
    explanation_quality_score: int | None = None
    model_usefulness_score: int | None = None


class PatientUploadCreate(BaseModel):
    upload_type: str = "document"
    file_name: str
    content_type: str | None = None
    content_base64: str
    notes: str | None = None
    scan_date: date | None = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _profile_to_dict(profile):
    if profile is None:
        return None
    return {
        "cancer_stage": profile.cancer_stage,
        "er_status": profile.er_status,
        "pr_status": profile.pr_status,
        "her2_status": profile.her2_status,
        "molecular_subtype": profile.molecular_subtype,
        "treatment_intent": profile.treatment_intent,
        "menopausal_status": profile.menopausal_status,
    }


def _combined_imaging_reports(imaging_reports, ct_reports):
    columns = ["date", "modality", "report_type", "body_site", "findings", "impression"]
    frames = []

    if imaging_reports is not None and not imaging_reports.empty:
        frame = imaging_reports.copy()
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frames.append(frame[columns])

    if ct_reports is not None and not ct_reports.empty:
        frame = ct_reports.copy()
        frame["modality"] = "CT chest/abdomen/pelvis"
        frame["body_site"] = "Chest/abdomen/pelvis"
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frames.append(frame[columns])

    if not frames:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date")


def build_patient_report_response(patient_id: str, db: Session):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    labs = get_labs_df(db, patient_id)
    treatments = get_treatments_df(db, patient_id)
    imaging_reports = get_imaging_reports_df(db, patient_id)
    ct_reports = get_ct_reports_df(db, patient_id)
    combined_imaging_reports = _combined_imaging_reports(imaging_reports, ct_reports)
    symptoms = get_symptoms_df(db, patient_id)
    mri_registry = get_mri_registry(db, patient_id)
    mri_series_index = get_mri_series_index(db, patient_id)
    medication_logs = get_medication_logs(db, patient_id)
    chat_history = get_chat_messages(db, patient_id, limit=12)
    clinical_interventions = get_clinical_interventions(db, patient_id)
    treatment_outcome = get_treatment_outcome(db, patient_id)
    breast_profile = get_breast_cancer_profile(db, patient_id)

    trends = {}
    risks = []
    trend_risks = []
    if not labs.empty:
        trends = analyze_labs(labs)
        risks = detect_risks(labs)
        trend_risks = detect_trend_risk(labs)
    symptom_risks = detect_symptom_risks(symptoms)
    clinical_rule_risks = detect_clinical_rule_risks(labs, symptoms, treatments)

    treatment_effects = []
    if not treatments.empty and not labs.empty:
        treatment_effects = align_labs_with_treatment(labs, treatments)

    radiology_summary = None
    if not combined_imaging_reports.empty:
        radiology_summary = analyze_breast_imaging_reports(combined_imaging_reports)

    radiology_risks = []
    if radiology_summary:
        radiology_risks = [
            {
                "type": "possible_metastatic_indicator",
                "category": "radiology_nlp",
                "severity": "urgent_review",
                "message": indicator["message"],
                "evidence": {
                    "date": indicator["date"],
                    "site": indicator["site"],
                },
            }
            for indicator in radiology_summary.get("possible_metastatic_indicators", [])
        ]

    all_risks = risks + trend_risks + symptom_risks + clinical_rule_risks + radiology_risks
    timeline = build_clinical_timeline(
        labs=labs,
        treatments=treatments,
        imaging_reports=combined_imaging_reports,
        symptoms=symptoms,
        risks=all_risks,
    )
    patient_state = build_patient_state(
        patient=patient,
        breast_profile=breast_profile,
        labs=labs,
        trends=trends,
        risks=all_risks,
        treatment_effects=treatment_effects,
        radiology_summary=radiology_summary,
        symptoms=symptoms,
    )
    summary = generate_clinical_summary(patient_state)
    report = build_patient_report(
        patient_state=patient_state,
        labs=labs,
        trends=trends,
        risks=all_risks,
        treatment_effects=treatment_effects,
        radiology_summary=radiology_summary,
        symptoms=symptoms,
        timeline=timeline,
        ai_summary=summary,
    )

    report["patient_id"] = patient.id
    report["patient_name"] = patient.name
    report["diagnosis"] = patient.diagnosis
    report["breast_cancer_profile"] = _profile_to_dict(breast_profile)
    report["mri_registry"] = mri_registry
    report["mri_series_index"] = mri_series_index
    report["medication_logs"] = medication_logs
    report["chat_history"] = chat_history
    report["uploads"] = get_patient_uploads(db, patient.id, limit=25)
    report["clinical_interventions"] = clinical_interventions
    report["treatment_outcome"] = treatment_outcome
    try:
        from backend.services.complete_synthetic_xai import (
            load_complete_synthetic_patient_prediction,
            load_complete_synthetic_patient_xai,
        )
        report["synthetic_model_prediction"] = load_complete_synthetic_patient_prediction(patient.id)
        report["synthetic_model_explanation"] = load_complete_synthetic_patient_xai(patient.id)
    except Exception:
        report["synthetic_model_prediction"] = None
        report["synthetic_model_explanation"] = None
    report["multimodal_assessment"] = build_multimodal_assessment(patient.id, report)
    report["patient_timeline_summary"] = build_patient_timeline_risk_summary(report)
    report["timeline_intelligence"] = build_timeline_intelligence(report)
    report["data_availability"] = build_data_availability(report)
    try:
        from backend.services.clinician_feedback import latest_clinical_summary_review
        report["latest_clinician_review"] = latest_clinical_summary_review(db, patient.id)
    except Exception:
        report["latest_clinician_review"] = None

    return report


# ─── Patient CRUD ─────────────────────────────────────────────────────────────

@router.get("/patients")
def list_patients(
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    patients = get_all_patients(db)
    return [
        {
            "id": patient.id,
            "name": patient.name,
            "diagnosis": patient.diagnosis,
            "breast_cancer_profile": _profile_to_dict(get_breast_cancer_profile(db, patient.id)),
        }
        for patient in patients
    ]


@router.post("/patients")
def create_patient(
    payload: PatientCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    try:
        validate_patient_payload(payload.id, payload.name)
    except ValueError as exc:
        log_app_event(
            db=db, event_type="validation_error", route="/patients",
            status="error", input_payload=payload.dict(), error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients")) from exc

    if get_patient(db, payload.id):
        raise HTTPException(status_code=400, detail="Patient already exists")

    patient = Patient(
        id=payload.id,
        name=payload.name,
        diagnosis=payload.diagnosis or "Breast cancer - doctor-confirmed",
    )
    db.add(patient)
    db.add(BreastCancerProfile(
        patient_id=patient.id,
        cancer_stage=payload.cancer_stage,
        er_status=payload.er_status,
        pr_status=payload.pr_status,
        her2_status=payload.her2_status,
        molecular_subtype=payload.molecular_subtype,
        treatment_intent=payload.treatment_intent,
        menopausal_status=payload.menopausal_status,
    ))
    db.commit()
    return {"message": "Patient created", "patient_id": patient.id}


# ─── Patient report ───────────────────────────────────────────────────────────

@router.get("/patient-report/{patient_id}")
def generate_patient_report_endpoint(
    patient_id: str,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    return build_patient_report_response(patient_id, db)


@router.get("/me/patient-report")
def get_my_patient_report(
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    return build_patient_report_response(context.patient_id, db)


# ─── Timeline intelligence ────────────────────────────────────────────────────

@router.post("/patients/{patient_id}/timeline-question")
def answer_patient_timeline_question_endpoint(
    patient_id: str,
    payload: TimelineQuestionRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    report = build_patient_report_response(patient_id, db)
    return answer_timeline_question(report, payload.question)


# ─── Clinician summary review ────────────────────────────────────────────────

@router.post("/patients/{patient_id}/summary-review")
def create_patient_summary_review_endpoint(
    patient_id: str,
    payload: ClinicianSummaryReviewRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    from backend.services.clinician_feedback import create_clinical_summary_review
    report = build_patient_report_response(patient_id, db)
    try:
        review = create_clinical_summary_review(
            db=db,
            patient_id=patient_id,
            reviewer_role=context.role,
            decision=payload.decision,
            summary_snapshot=report.get("patient_timeline_summary") or {},
            clinician_notes=payload.clinician_notes,
            edited_patient_summary=payload.edited_patient_summary,
            explanation_quality_score=payload.explanation_quality_score,
            model_usefulness_score=payload.model_usefulness_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Clinician summary review saved",
        "review": review,
        "safety_note": "Review feedback is audit data. It does not change the patient record automatically.",
    }


@router.get("/summary-reviews")
def list_summary_reviews_endpoint(
    patient_id: str | None = None,
    limit: int = 50,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.clinician_feedback import list_clinical_summary_reviews
    return {"summary_reviews": list_clinical_summary_reviews(db, patient_id=patient_id, limit=limit)}


@router.get("/clinician/review-queue")
def clinician_review_queue_endpoint(
    limit: int = 25,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    patients = get_all_patients(db)[: max(1, min(limit, 100))]
    rows = []
    for patient in patients:
        report = build_patient_report_response(patient.id, db)
        assessment = report.get("multimodal_assessment") or {}
        summary = report.get("patient_timeline_summary") or {}
        intelligence = report.get("timeline_intelligence") or {}
        latest_review = report.get("latest_clinician_review")
        risks = report.get("risks") or []
        urgent_count = sum(1 for risk in risks if risk.get("severity") == "urgent_review")
        review_flags = summary.get("review_flags") or []
        missing = intelligence.get("missing_data_warnings") or []
        status = assessment.get("overall_status") or "unknown"
        priority_score = (
            urgent_count * 5
            + len(review_flags) * 2
            + (0 if latest_review else 2)
            + (2 if status in {"needs_clinician_review", "watch_closely"} else 0)
            + min(len(missing), 3)
        )
        rows.append({
            "patient_id": patient.id,
            "patient_name": patient.name,
            "overall_status": status,
            "priority_score": priority_score,
            "urgent_flag_count": urgent_count,
            "review_flag_count": len(review_flags),
            "missing_data_count": len(missing),
            "latest_review_decision": latest_review.get("decision") if latest_review else None,
            "headline": summary.get("headline"),
            "top_review_flags": review_flags[:3],
            "top_missing_data_warnings": missing[:3],
            "recommended_action": assessment.get("recommended_action"),
        })

    rows = sorted(rows, key=lambda row: (row["priority_score"], row["urgent_flag_count"]), reverse=True)
    return {
        "queue": rows,
        "safety_note": "Review queue prioritizes monitoring signals for clinician attention. It does not diagnose or choose treatment.",
    }


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
    except ValueError as exc:
        log_app_event(
            db=db, event_type="chat_error", actor_role=context.role,
            patient_id=context.patient_id, route="/me/chat", status="error",
            input_payload={"message": payload.message}, error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/me/chat")) from exc
    return result


# ─── Streaming chat (SSE) ─────────────────────────────────────────────────────

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_agent_pipeline(db: Session, patient_id: str, message: str):
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

    yield _sse_event("pipeline_stage", {"stage": "safety_gate", "label": "Checking safety gate…"})

    try:
        result = run_patient_agent_pipeline(
            db=db,
            patient_id=patient_id,
            query=message,
        )
    except Exception as exc:
        yield _sse_event("error", {"error": "Agent pipeline failed. Please try again.", "code": "pipeline_error"})
        yield _sse_event("done", {})
        return

    yield _sse_event("pipeline_stage", {"stage": "intent_routing", "label": "Routing intent…"})
    yield _sse_event("pipeline_stage", {"stage": "retrieval", "label": "Retrieving context…"})
    yield _sse_event("pipeline_stage", {"stage": "generation", "label": "Generating response…"})

    citations = result.get("citations") or []
    answer_payload = {
        "reply": result.get("reply") or "",
        "citations": [
            {"id": c.get("id"), "title": c.get("title"), "source_name": c.get("source_name"), "source_url": c.get("source_url")}
            for c in citations
            if isinstance(c, dict)
        ],
        "intent": result.get("intent"),
        "safety_level": (result.get("safety") or {}).get("level"),
        "cache_status": (result.get("cache") or {}).get("status"),
        "saved_actions": result.get("saved_actions") or [],
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
        _stream_agent_pipeline(db, context.patient_id, payload.message),
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
    return {"patient_id": context.patient_id, "uploads": get_patient_uploads(db, context.patient_id, limit=50)}


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

    return {"message": "Upload saved to patient record", "upload": upload}


# ─── Clinical data entry ─────────────────────────────────────────────────────

@router.post("/patients/{patient_id}/labs")
def add_lab_result(
    patient_id: str,
    payload: LabCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        validation_warnings = validate_cbc_values(payload.wbc, payload.hemoglobin, payload.platelets)
    except ValueError as exc:
        log_app_event(db=db, event_type="validation_error", patient_id=patient_id, route="/patients/{patient_id}/labs", status="error", input_payload=payload.dict(), error_message=str(exc))
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/labs")) from exc

    lab = LabResult(patient_id=patient_id, date=payload.date, wbc=payload.wbc, hemoglobin=payload.hemoglobin, platelets=payload.platelets, source=payload.source or "manual", source_note=payload.source_note)
    db.add(lab)
    db.commit()
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/labs", status="ok", input_payload=payload.dict(), output_payload={"lab_id": lab.id, "warning_count": len(validation_warnings)})
    return {"message": "Lab result added", "validation_warnings": validation_warnings, "error_state": None}


@router.post("/patients/{patient_id}/treatments")
def add_treatment(
    patient_id: str,
    payload: TreatmentCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        validation_warnings = validate_treatment_payload(payload.cycle, payload.drug)
    except ValueError as exc:
        log_app_event(db=db, event_type="validation_error", patient_id=patient_id, route="/patients/{patient_id}/treatments", status="error", input_payload=payload.dict(), error_message=str(exc))
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/treatments")) from exc

    treatment = Treatment(patient_id=patient_id, date=payload.date, cycle=payload.cycle, drug=payload.drug)
    db.add(treatment)
    db.commit()
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/treatments", status="ok", input_payload=payload.dict(), output_payload={"treatment_id": treatment.id})
    return {"message": "Treatment added", "validation_warnings": validation_warnings, "error_state": None}


@router.post("/patients/{patient_id}/symptoms")
def add_symptom_report(
    patient_id: str,
    payload: SymptomCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        validation_warnings = validate_symptom_payload(payload.symptom, payload.severity, payload.notes)
    except ValueError as exc:
        log_app_event(db=db, event_type="validation_error", patient_id=patient_id, route="/patients/{patient_id}/symptoms", status="error", input_payload=payload.dict(), error_message=str(exc))
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/symptoms")) from exc

    symptom = SymptomReport(patient_id=patient_id, date=payload.date, symptom=payload.symptom, severity=payload.severity, notes=payload.notes)
    db.add(symptom)
    db.commit()
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/symptoms", status="ok", input_payload=payload.dict(), output_payload={"symptom_id": symptom.id, "warning_count": len(validation_warnings)})
    return {"message": "Symptom report added", "validation_warnings": validation_warnings, "error_state": None}


@router.post("/patients/{patient_id}/imaging-reports")
def add_imaging_report(
    patient_id: str,
    payload: ImagingReportCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        validation_warnings = validate_imaging_report_payload(payload.modality, payload.report_type, payload.findings, payload.impression, payload.body_site)
    except ValueError as exc:
        log_app_event(db=db, event_type="validation_error", patient_id=patient_id, route="/patients/{patient_id}/imaging-reports", status="error", input_payload=payload.dict(), error_message=str(exc))
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/imaging-reports")) from exc

    report = ImagingReport(patient_id=patient_id, date=payload.date, modality=payload.modality, report_type=payload.report_type, body_site=payload.body_site, findings=payload.findings, impression=payload.impression)
    db.add(report)
    db.commit()
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/imaging-reports", status="ok", input_payload={**payload.dict(), "findings": "[redacted]", "impression": "[redacted]"}, output_payload={"imaging_report_id": report.id, "warning_count": len(validation_warnings)})
    return {"message": "Imaging report added", "validation_warnings": validation_warnings, "error_state": None}


@router.post("/patients/{patient_id}/mri-registry")
def add_mri_registry_entry(
    patient_id: str,
    payload: MRIRegistryCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    entry = MRIFileRegistry(patient_id=patient_id, scan_date=payload.scan_date, modality=payload.modality, series_description=payload.series_description, local_path=payload.local_path, notes=payload.notes)
    db.add(entry)
    db.commit()
    return {"message": "MRI registry entry added", "id": entry.id}


@router.post("/patients/{patient_id}/ct-reports")
def add_ct_report(
    patient_id: str,
    payload: CTReportCreate,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        validation_warnings = validate_imaging_report_payload("CT", payload.report_type, payload.findings, payload.impression, body_site="Chest/abdomen/pelvis")
    except ValueError as exc:
        log_app_event(db=db, event_type="validation_error", patient_id=patient_id, route="/patients/{patient_id}/ct-reports", status="error", input_payload={**payload.dict(), "findings": "[redacted]", "impression": "[redacted]"}, error_message=str(exc))
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/ct-reports")) from exc

    report = CTReport(patient_id=patient_id, date=payload.date, report_type=payload.report_type, findings=payload.findings, impression=payload.impression)
    db.add(report)
    db.commit()
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/ct-reports", status="ok", input_payload={**payload.dict(), "findings": "[redacted]", "impression": "[redacted]"}, output_payload={"ct_report_id": report.id, "warning_count": len(validation_warnings)})
    return {"message": "CT report added", "validation_warnings": validation_warnings, "error_state": None}
