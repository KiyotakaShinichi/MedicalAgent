"""
Patient router — /me/*, /patients/*, /patient-report/*, /summary-reviews, /clinician/review-queue.

Includes the streaming chat endpoint at POST /me/chat/stream and
POST /patients/{patient_id}/chat/stream.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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
    PatientUpload,
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
from backend.services.ctcae_mapping import map_symptom_to_ctcae_review_hint
from backend.services.lab_reference_context import build_cbc_reference_context
from backend.services.multimodal_fusion import build_multimodal_assessment
from backend.services.patient_timeline_summary import build_patient_timeline_risk_summary
from backend.services.support_chat_agent import handle_patient_chat
from backend.services.timeline_intelligence import answer_timeline_question, build_timeline_intelligence
from backend.services.data_availability import build_data_availability
from backend.services.patient_report_enrichment_jobs import (
    get_patient_enrichment_job,
    invalidate_patient_enrichment,
    schedule_patient_enrichment,
)

router = APIRouter(tags=["patient"])
_REPORT_CACHE_TTL_SECONDS = max(120, int(os.getenv("NLCARE_PATIENT_REPORT_CACHE_TTL_SECONDS", "900")))
_REPORT_CORE_CACHE_TTL_SECONDS = max(30, int(os.getenv("NLCARE_PATIENT_CORE_CACHE_TTL_SECONDS", "120")))
_REPORT_CACHE: dict[str, tuple[float, dict]] = {}
_REPORT_CORE_CACHE: dict[str, tuple[float, dict]] = {}


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


class MySymptomCreate(BaseModel):
    """Patient-scoped variant of :class:`SymptomCreate`.

    Has the same data shape but is submitted by the patient themselves via
    ``POST /me/symptoms``.  Adds two optional fields the manual-entry form
    surfaces but the clinician-side route never needed:

    - ``duration``     — free text describing how long the symptom has lasted
                         (e.g. "since this morning", "2 days").
    - ``urgent_flag``  — checkbox the patient set explicitly.  When True we
                         tag the saved record's notes with ``[urgent flag]``
                         so the clinician review queue picks it up — we do
                         NOT auto-route anything; the safety promise is that
                         the system only *surfaces* the flag, it never decides
                         on it.
    """

    date: date
    symptom: str
    severity: int
    notes: str | None = None
    duration: str | None = None
    urgent_flag: bool = False


class MyLabCreate(BaseModel):
    """Patient-scoped CBC save (manual-entry form).

    Adds ``anc`` and ``lab_source`` over the clinician-side :class:`LabCreate`.
    ``anc`` is not part of the LabResult table schema yet — when present we
    fold it into ``source_note`` so we do not need a migration.  Replace
    with a first-class column when the schema is next updated.
    """

    date: date
    wbc: float
    hemoglobin: float
    platelets: float
    anc: float | None = None
    lab_source: str | None = None  # e.g. "Quest", "Home draw", or a clinic name
    notes: str | None = None


class MyImagingReportCreate(BaseModel):
    """Patient-scoped imaging save (manual-entry form).

    Modality is one of MRI/CT/Ultrasound/Mammogram/Other; the backend stores
    it verbatim.  ``report_type`` defaults to "Patient-entered report" when
    the patient hasn't typed one — keeps the existing schema happy.
    """

    date: date
    modality: str
    report_type: str | None = None
    body_site: str | None = None
    findings: str | None = None
    impression: str | None = None
    notes: str | None = None


class MyMedicationCreate(BaseModel):
    medication: str
    dose: str | None = None
    frequency: str | None = None
    date: date
    side_effects: str | None = None
    notes: str | None = None


class MyTreatmentCreate(BaseModel):
    """Patient-scoped treatment-cycle note.

    The clinician :class:`TreatmentCreate` requires an integer cycle number.
    Patients often don't remember the cycle — accept it as optional and
    default to 0 so the row still slots into the treatments table.
    """

    date: date
    drug: str
    cycle: int | None = None
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


class FamilyHistoryCreate(BaseModel):
    relationship: str
    family_side: str | None = None
    cancer_type: str
    age_at_diagnosis: int | None = None
    relative_status: str | None = None
    multiple_relatives_affected: str | None = "unknown"
    male_breast_cancer: str | None = "unknown"
    known_familial_mutation: str | None = "unknown"
    bilateral_breast_cancer: str | None = "unknown"
    multiple_primary_cancers: str | None = "unknown"
    ancestry_ethnicity: str | None = None
    prior_breast_biopsy_atypia: str | None = "unknown"
    relation_degree: str | None = None
    notes: str | None = None


class GeneticTestRecordCreate(BaseModel):
    test_type: str
    sample_type: str | None = "unknown"
    gene: str | None = None
    variant_text: str | None = None
    classification: str | None = "unknown"
    report_date: date | None = None
    lab_provider: str | None = None
    upload_reference: str | None = None
    reviewed_by_genetic_counselor: str | None = "unknown"
    clinician_review_status: str | None = "pending"
    notes: str | None = None


class BiomarkerRecordCreate(BaseModel):
    source: str
    er_status: str | None = "unknown"
    pr_status: str | None = "unknown"
    her2_status: str | None = "unknown"
    ki67_percent: float | None = None
    grade: str | None = None
    stage: str | None = None
    report_date: date | None = None
    report_text: str | None = None
    upload_reference: str | None = None
    clinician_review_needed: str | None = "yes"


class TumorMarkerRecordCreate(BaseModel):
    marker: str
    value: float
    unit: str | None = None
    reference_range: str | None = None
    date_collected: date
    trend_direction: str | None = "unknown"
    notes: str | None = None


class GeneticReviewCreate(BaseModel):
    decision: str
    notes: str | None = None


# Note: TimelineQuestionRequest and ClinicianSummaryReviewRequest have moved
# to backend/api/routers/clinician_review.py alongside the endpoints that
# consume them.


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


def build_patient_report_response(patient_id: str, db: Session, *, include_enrichment: bool = True):
    started_at = time.perf_counter()
    cached = _get_cached_report(patient_id) if include_enrichment else _get_cached_core_report(patient_id)
    if cached is not None:
        return cached

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
    patient_uploads = get_patient_uploads(db, patient.id, limit=25)
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
        media_records=[*mri_registry, *patient_uploads],
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
    report["mri_registry"] = [
        {key: value for key, value in entry.items() if key != "local_path"}
        for entry in mri_registry
    ]
    report["mri_series_index"] = [
        {key: value for key, value in entry.items() if key != "folder"}
        for entry in mri_series_index
    ]
    report["medication_logs"] = medication_logs
    report["chat_history"] = chat_history
    report["uploads"] = [
        {
            **{key: value for key, value in entry.items() if key != "local_path"},
            "content_url": f"/me/uploads/{entry['id']}/content",
        }
        for entry in patient_uploads
    ]
    report["clinical_interventions"] = clinical_interventions
    report["treatment_outcome"] = treatment_outcome

    if not include_enrichment:
        report["synthetic_model_prediction"] = None
        report["synthetic_model_explanation"] = None
        report["hybrid_prediction"] = None
        report["evidence_aware_prediction"] = None
        _attach_derived_report_sections(report, patient, db)
        report["report_enrichment"] = {
            "status": "deferred",
            "profile": "records_first",
            "generated_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "message": "Patient records are ready; synthetic engineering details load separately.",
            "clinical_validation": False,
        }
        _set_cached_core_report(patient_id, report)
        return report

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
    # Live hybrid prediction: runs classification + regression + toxicity
    # through the abstention layer, persists one trace per head, and embeds
    # the bundle on the report.  Also publishes the classification slice
    # under the old `evidence_aware_prediction` key for backward compat with
    # the existing patient-dashboard card.  None when the patient is not in
    # the synthetic cohort or the trained model isn't on disk.
    try:
        from backend.services.live_evidence_prediction import build_hybrid_prediction
        bundle = build_hybrid_prediction(patient.id, db, actor_role="patient")
        report["hybrid_prediction"] = bundle
        report["evidence_aware_prediction"] = (
            bundle.get("classification") if bundle else None
        )
    except Exception:
        report["hybrid_prediction"] = None
        report["evidence_aware_prediction"] = None
    _attach_derived_report_sections(report, patient, db)
    report["report_enrichment"] = {
        "status": "complete",
        "profile": "full_engineering_bundle",
        "generated_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "message": "Records and synthetic engineering details are ready.",
        "clinical_validation": False,
    }

    _set_cached_report(patient_id, report)
    return report


def _attach_derived_report_sections(report: dict, patient: Patient, db: Session) -> None:
    report["multimodal_assessment"] = build_multimodal_assessment(patient.id, report)
    # Retained for clinician/admin backward compatibility. Patient headlines
    # intentionally do not display this synthetic workflow index as a health score.
    report["monitoring_score"] = (
        report["multimodal_assessment"] or {}
    ).get("treatment_monitoring_score")
    report["patient_timeline_summary"] = build_patient_timeline_risk_summary(report)
    report["timeline_intelligence"] = build_timeline_intelligence(report)
    report["data_availability"] = build_data_availability(report)
    from backend.services.patient_xai_envelope import build_patient_xai_envelope

    report["xai_explanation_envelope"] = build_patient_xai_envelope(
        prediction=report.get("synthetic_model_prediction"),
        explanation=report.get("synthetic_model_explanation"),
        hybrid_prediction=report.get("hybrid_prediction"),
        data_availability=report.get("data_availability"),
    )
    try:
        from backend.services.genetic_counseling import build_genetic_counseling_readiness
        report["genetic_counseling_readiness"] = build_genetic_counseling_readiness(db, patient.id)
    except Exception:
        report["genetic_counseling_readiness"] = None
    try:
        from backend.services.clinician_feedback import latest_clinical_summary_review
        report["latest_clinician_review"] = latest_clinical_summary_review(db, patient.id)
    except Exception:
        report["latest_clinician_review"] = None


def _get_cached_report(patient_id: str) -> dict | None:
    item = _REPORT_CACHE.get(patient_id)
    if not item:
        return None
    created_at, report = item
    if time.monotonic() - created_at > _REPORT_CACHE_TTL_SECONDS:
        _REPORT_CACHE.pop(patient_id, None)
        return None
    return report


def _set_cached_report(patient_id: str, report: dict) -> None:
    _REPORT_CACHE[patient_id] = (time.monotonic(), report)


def _get_cached_core_report(patient_id: str) -> dict | None:
    item = _REPORT_CORE_CACHE.get(patient_id)
    if not item:
        return None
    created_at, report = item
    if time.monotonic() - created_at > _REPORT_CORE_CACHE_TTL_SECONDS:
        _REPORT_CORE_CACHE.pop(patient_id, None)
        return None
    return report


def _set_cached_core_report(patient_id: str, report: dict) -> None:
    _REPORT_CORE_CACHE[patient_id] = (time.monotonic(), report)


def _invalidate_report_cache(patient_id: str | None = None) -> None:
    invalidate_patient_enrichment(patient_id)
    if patient_id is None:
        _REPORT_CACHE.clear()
        _REPORT_CORE_CACHE.clear()
    else:
        _REPORT_CACHE.pop(patient_id, None)
        _REPORT_CORE_CACHE.pop(patient_id, None)


def _discard_stale_enrichment_result(patient_id: str, report: dict) -> None:
    cached = _REPORT_CACHE.get(patient_id)
    if cached is not None and cached[1] is report:
        _REPORT_CACHE.pop(patient_id, None)


def _schedule_report_enrichment(patient_id: str) -> dict:
    return schedule_patient_enrichment(
        patient_id,
        build=lambda item, worker_db: build_patient_report_response(item, worker_db, include_enrichment=True),
        discard_stale_result=_discard_stale_enrichment_result,
    )


def warm_patient_report_enrichment_cache() -> None:
    """Schedule demo prewarming without delaying application startup."""
    environment = (os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or "development").strip().lower()
    enabled = os.getenv("NLCARE_PATIENT_ENRICHMENT_PREWARM_ENABLED")
    if enabled is None:
        enabled = "false" if environment in {"test", "testing"} else "true"
    if str(enabled).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    from backend.database import SessionLocal

    db = SessionLocal()
    try:
        patient_ids = [patient.id for patient in get_all_patients(db)]
    finally:
        db.close()
    for patient_id in patient_ids:
        if _get_cached_report(patient_id) is None:
            _schedule_report_enrichment(patient_id)


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


@router.get("/me/patient-report/core")
def get_my_patient_report_core(
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Return patient records before deferred synthetic model enrichment."""
    report = build_patient_report_response(context.patient_id, db, include_enrichment=False)
    if _get_cached_report(context.patient_id) is None:
        _schedule_report_enrichment(context.patient_id)
    return report


@router.get("/me/patient-report/enrichment")
def get_my_patient_report_enrichment(
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    """Read deferred fields without running the synthetic model in this request."""
    keys = (
        "synthetic_model_prediction",
        "synthetic_model_explanation",
        "hybrid_prediction",
        "evidence_aware_prediction",
        "multimodal_assessment",
        "monitoring_score",
        "patient_timeline_summary",
        "timeline_intelligence",
        "data_availability",
        "xai_explanation_envelope",
        "report_enrichment",
    )
    report = _get_cached_report(context.patient_id)
    if report is not None:
        return {key: report.get(key) for key in keys}

    job = get_patient_enrichment_job(context.patient_id)
    if job is None or job.get("status") not in {"queued", "running"}:
        job = _schedule_report_enrichment(context.patient_id)
    pending = {key: None for key in keys if key != "report_enrichment"}
    pending["report_enrichment"] = {
        **job,
        "profile": "background_single_flight",
        "generated_ms": job.get("generated_ms") or 0.0,
        "message": (
            "Synthetic engineering details are being prepared outside the request path."
            if job.get("status") in {"queued", "running"}
            else "Synthetic engineering details could not be prepared; patient records remain available."
        ),
    }
    return pending


@router.get("/me/genetic-counseling-readiness")
def get_my_genetic_counseling_readiness(
    context=Depends(get_patient_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import build_genetic_counseling_readiness

    return build_genetic_counseling_readiness(db, context.patient_id)


@router.get("/patients/{patient_id}/genetic-counseling-readiness")
def get_patient_genetic_counseling_readiness(
    patient_id: str,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.genetic_counseling import build_genetic_counseling_readiness

    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return build_genetic_counseling_readiness(db, patient_id)


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
    except Exception as exc:
        yield _sse_event("error", {"error": "Agent pipeline failed. Please try again.", "code": "pipeline_error"})
        yield _sse_event("done", {})
        return

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

    reference_context = build_cbc_reference_context(wbc=payload.wbc, hemoglobin=payload.hemoglobin, platelets=payload.platelets)
    lab = LabResult(patient_id=patient_id, date=payload.date, wbc=payload.wbc, hemoglobin=payload.hemoglobin, platelets=payload.platelets, source=payload.source or "manual", source_note=payload.source_note)
    db.add(lab)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/labs", status="ok", input_payload=payload.dict(), output_payload={"lab_id": lab.id, "warning_count": len(validation_warnings), "reference_context": reference_context})
    return {"message": "Lab result added", "validation_warnings": validation_warnings, "reference_context": reference_context, "error_state": None}


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
    _invalidate_report_cache(patient_id)
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

    ctcae_hint = map_symptom_to_ctcae_review_hint(symptom=payload.symptom, severity=payload.severity, notes=payload.notes)
    symptom = SymptomReport(patient_id=patient_id, date=payload.date, symptom=payload.symptom, severity=payload.severity, notes=payload.notes)
    db.add(symptom)
    db.commit()
    _invalidate_report_cache(patient_id)
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/symptoms", status="ok", input_payload=payload.dict(), output_payload={"symptom_id": symptom.id, "warning_count": len(validation_warnings), "ctcae_hint": ctcae_hint})
    return {"message": "Symptom report added", "validation_warnings": validation_warnings, "ctcae_review_hint": ctcae_hint, "error_state": None}


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
    _invalidate_report_cache(patient_id)
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
    _invalidate_report_cache(patient_id)
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
    _invalidate_report_cache(patient_id)
    log_app_event(db=db, event_type="patient_input", patient_id=patient_id, route="/patients/{patient_id}/ct-reports", status="ok", input_payload={**payload.dict(), "findings": "[redacted]", "impression": "[redacted]"}, output_payload={"ct_report_id": report.id, "warning_count": len(validation_warnings)})
    return {"message": "CT report added", "validation_warnings": validation_warnings, "error_state": None}
