"""
Patient router — /me/*, /patients/*, /patient-report/*, /summary-reviews, /clinician/review-queue.

Includes the streaming chat endpoint at POST /me/chat/stream and
POST /patients/{patient_id}/chat/stream.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
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
    CTReportCreate,
    FamilyHistoryCreate,
    GeneticReviewCreate,
    GeneticTestRecordCreate,
    ImagingReportCreate,
    LabCreate,
    MRIRegistryCreate,
    MyImagingReportCreate,
    MyLabCreate,
    MyMedicationCreate,
    MySymptomCreate,
    MyTreatmentCreate,
    PatientChatRequest,
    PatientCreate,
    PatientUploadCreate,
    SymptomCreate,
    TreatmentCreate,
    TumorMarkerRecordCreate,
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
    from backend.services.record_change_explanation import build_record_change_explanation

    report["record_change_explanation"] = build_record_change_explanation(
        lab_history=report.get("lab_history") or [],
        symptoms=report.get("symptoms") or [],
        imaging_reports=(report.get("radiology_summary") or {}).get("reports") or [],
    )
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

# Import after report helpers and legacy clinician-write routes are defined.
# The subrouter imports only the shared cache invalidator from this module.
from backend.api.routers.patient_interactions import router as patient_interactions_router

router.include_router(patient_interactions_router)
