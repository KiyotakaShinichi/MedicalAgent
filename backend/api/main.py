from datetime import date

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

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
from backend.database import SessionLocal
from backend.models import (
    BreastCancerProfile,
    CTReport,
    ImagingReport,
    LabResult,
    MRIFileRegistry,
    MRISeriesIndex,
    Patient,
    SymptomReport,
    Treatment,
)
from backend.processing.radiology_analysis import analyze_breast_imaging_reports, analyze_radiology_reports
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
from backend.schema_migrations import ensure_schema
from backend.services.csv_importer import (
    DATASET_ADAPTERS,
    SUPPORTED_IMPORT_TYPES,
    import_csv,
    import_qin_breast_02_clinical_xlsx,
)
from backend.services.synthetic_cbc import generate_synthetic_cbc_for_qin_patients
from backend.services.synthetic_journey import generate_synthetic_breast_cancer_journeys
from backend.services.mri_series_indexer import index_mri_series
from backend.services.mri_manifest import build_qin_mri_manifest
from backend.services.mri_preprocessing import preprocess_mri_manifest_previews
from backend.services.breastdcedl_inspector import build_breastdcedl_manifest, inspect_breastdcedl_dataset
from backend.services.multimodal_fusion import build_multimodal_assessment
from backend.services.patient_timeline_summary import build_patient_timeline_risk_summary
from backend.services.support_chat_agent import handle_patient_chat
from backend.services.timeline_intelligence import answer_timeline_question, build_timeline_intelligence
from backend.services.app_logging import log_app_event
from backend.services.data_availability import build_data_availability
from backend.services.input_validation import (
    validate_cbc_values,
    validate_chat_message,
    validate_imaging_report_payload,
    validate_patient_payload,
    validate_symptom_payload,
    validate_treatment_payload,
    validation_error_payload,
)

app = FastAPI(title="AI Breast Cancer Monitoring System")
ensure_schema()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_access_context(authorization: str | None = Header(None), db: Session = Depends(get_db)):
    from backend.services.auth import get_context_from_authorization

    try:
        return get_context_from_authorization(db, authorization)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def get_patient_access_context(context=Depends(get_access_context)):
    from backend.services.auth import require_patient_context

    try:
        return require_patient_context(context)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def get_clinician_or_admin_context(context=Depends(get_access_context)):
    from backend.services.auth import require_admin_or_clinician

    try:
        return require_admin_or_clinician(context)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def get_admin_access_context(context=Depends(get_access_context)):
    from backend.services.auth import require_admin_context

    try:
        return require_admin_context(context)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


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


class CTReportCreate(BaseModel):
    date: date
    report_type: str
    findings: str
    impression: str


class ImagingReportCreate(BaseModel):
    date: date
    modality: str
    report_type: str
    body_site: str | None = "Breast"
    findings: str
    impression: str


class CSVImportRequest(BaseModel):
    import_type: str
    dataset: str = "canonical"
    csv_text: str | None = None
    file_path: str | None = None


class QINBreast02ImportRequest(BaseModel):
    clinical_xlsx_path: str = "datasets/QIN-BREAST-02_clinicalData-Transformed-20191022-Revised20200210.xlsx"


class MRIRegistryCreate(BaseModel):
    scan_date: date | None = None
    modality: str = "Breast MRI"
    series_description: str | None = None
    local_path: str
    notes: str | None = None


class MRISeriesIndexRequest(BaseModel):
    root_path: str = "datasets/qin_breast_02"
    patient_id: str | None = None
    max_files: int | None = None


class MRIManifestRequest(BaseModel):
    clinical_xlsx_path: str | None = "Datasets/QIN-BREAST-02_clinicalData-Transformed-20191022-Revised20200210.xlsx"
    output_csv_path: str | None = "Data/qin_breast_02_mri_manifest.csv"


class MRIPreviewPreprocessRequest(BaseModel):
    manifest_csv_path: str = "Data/qin_breast_02_mri_manifest.csv"
    output_dir: str = "Data/qin_mri_previews"


class SyntheticJourneyRequest(BaseModel):
    count: int = 25
    seed: int = 42


class TemporalSyntheticJourneyRequest(BaseModel):
    count: int = 12
    seed: int = 2026
    cycles: int = 6


class CompleteSyntheticDatasetRequest(BaseModel):
    count: int = 60
    seed: int = 2027
    cycles: int = 6
    output_dir: str = "Data/complete_synthetic_breast_journeys"
    write_db: bool = True
    patient_prefix: str = "COMP-BRCA-"
    balanced_outcomes: bool = True
    missing_rate: float = 0.04
    noise_level: float = 0.03


class CompleteSyntheticTrainingRequest(BaseModel):
    ml_csv_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
    output_dir: str = "Data/complete_synthetic_training"
    target: str = "treatment_success_binary"
    test_size: float = 0.25
    seed: int = 42
    cnn_epochs: int = 20
    cnn_batch_size: int = 16


class CompleteSyntheticXAIRequest(BaseModel):
    ml_csv_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
    model_path: str = "Data/complete_synthetic_training/logistic_regression_treatment_success_binary.joblib"
    predictions_csv_path: str = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
    output_json_path: str = "Data/complete_synthetic_training/synthetic_xai_explanations.json"
    top_n: int = 6


class DemoLoginRequest(BaseModel):
    role: str = "patient"
    patient_id: str | None = "COMPV4-BRCA-0001"


class PatientUploadCreate(BaseModel):
    upload_type: str = "document"
    file_name: str
    content_type: str | None = None
    content_base64: str
    notes: str | None = None
    scan_date: date | None = None


class BreastDCEDLInspectRequest(BaseModel):
    path: str = "Datasets/BreastDCEDL_spy1"


class BreastDCEDLManifestRequest(BaseModel):
    root_path: str = "Datasets/BreastDCEDL_spy1"
    output_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"


class BreastDCEDLBaselineRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    metrics_json_path: str = "Data/breastdcedl_spy1_baseline_metrics.json"
    predictions_csv_path: str = "Data/breastdcedl_spy1_model_predictions.csv"
    max_patients: int | None = None


class BreastDCEDLPreviewRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    output_dir: str = "Data/breastdcedl_previews"
    max_patients: int = 40


class BreastDCEDLImportRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    limit: int = 25


class BreastDCEDLCNNRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    metrics_json_path: str = "Data/breastdcedl_spy1_cnn_metrics.json"
    max_patients: int = 120
    epochs: int = 4
    batch_size: int = 8


class BreastDCEDLXAIRequest(BaseModel):
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    output_json_path: str = "Data/breastdcedl_spy1_shap_explanations.json"
    top_n: int = 5


class BreastDCEDLModelTrainRequest(BaseModel):
    version: str = "v1"
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    metrics_path: str = "Data/breastdcedl_spy1_baseline_metrics.json"
    artifact_dir: str = "Data/models"


class BreastDCEDLModelPredictRequest(BaseModel):
    model_name: str = "breastdcedl_pcr_logreg"
    model_version: str = "v1"
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    shap_json_path: str = "Data/breastdcedl_spy1_shap_explanations.json"


class CompleteSyntheticRegisterRequest(BaseModel):
    version: str = "synthetic-v1"
    metrics_path: str = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
    training_data_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
    artifact_dir: str = "Data/complete_synthetic_training"
    promotion_status: str = "candidate"
    promotion_reason: str | None = None


class PatientChatRequest(BaseModel):
    message: str


class TimelineQuestionRequest(BaseModel):
    question: str


class ClinicianSummaryReviewRequest(BaseModel):
    decision: str
    clinician_notes: str | None = None
    edited_patient_summary: str | None = None
    explanation_quality_score: int | None = None
    model_usefulness_score: int | None = None


class EvaluationReportRequest(BaseModel):
    output_root: str = "Data/model_evaluation_reports"
    run_id: str | None = None


class ModelVersionActionRequest(BaseModel):
    reason: str | None = None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/frontend/index.html")


@app.get("/patient", include_in_schema=False)
def patient_portal():
    return RedirectResponse(url="/frontend/patient.html")


@app.get("/clinician", include_in_schema=False)
def clinician_dashboard():
    return RedirectResponse(url="/frontend/index.html")


@app.get("/admin", include_in_schema=False)
def admin_dashboard():
    return RedirectResponse(url="/frontend/admin.html")


# Serve frontend static files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.post("/auth/demo-login")
def demo_login(payload: DemoLoginRequest, db: Session = Depends(get_db)):
    from backend.services.auth import create_demo_session

    try:
        return create_demo_session(db, role=payload.role, patient_id=payload.patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/auth/whoami")
def whoami(context=Depends(get_access_context)):
    return {
        "role": context.role,
        "patient_id": context.patient_id,
        "safety_note": "Demo sessions are role-scoped for the PoC. They are not production identity management.",
    }


@app.get("/patients")
def list_patients(db: Session = Depends(get_db)):
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


@app.post("/patients")
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    try:
        validate_patient_payload(payload.id, payload.name)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            route="/patients",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients")) from exc

    existing = get_patient(db, payload.id)
    if existing:
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

@app.get("/patient-report/{patient_id}")
def generate_patient_report(patient_id: str, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    labs = get_labs_df(db, patient_id)
    treatments = get_treatments_df(db, patient_id)
    imaging_reports = get_imaging_reports_df(db, patient_id)
    ct_reports = get_ct_reports_df(db, patient_id)
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
    if not imaging_reports.empty:
        radiology_summary = analyze_breast_imaging_reports(imaging_reports)
    elif not ct_reports.empty:
        radiology_summary = analyze_radiology_reports(ct_reports)

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
        imaging_reports=imaging_reports,
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


@app.get("/me/patient-report")
def get_my_patient_report(context=Depends(get_patient_access_context), db: Session = Depends(get_db)):
    return generate_patient_report(context.patient_id, db)


@app.post("/patients/{patient_id}/timeline-question")
def answer_patient_timeline_question_endpoint(
    patient_id: str,
    payload: TimelineQuestionRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    report = generate_patient_report(patient_id, db)
    return answer_timeline_question(report, payload.question)


@app.post("/patients/{patient_id}/summary-review")
def create_patient_summary_review_endpoint(
    patient_id: str,
    payload: ClinicianSummaryReviewRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    from backend.services.clinician_feedback import create_clinical_summary_review

    report = generate_patient_report(patient_id, db)
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


@app.get("/summary-reviews")
def list_summary_reviews_endpoint(
    patient_id: str | None = None,
    limit: int = 50,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.clinician_feedback import list_clinical_summary_reviews

    return {
        "summary_reviews": list_clinical_summary_reviews(db, patient_id=patient_id, limit=limit),
    }


@app.get("/clinician/review-queue")
def clinician_review_queue_endpoint(
    limit: int = 25,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    patients = get_all_patients(db)[: max(1, min(limit, 100))]
    rows = []
    for patient in patients:
        report = generate_patient_report(patient.id, db)
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


@app.get("/admin/analytics")
def get_admin_analytics_endpoint(
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.admin_analytics import build_admin_analytics

    return build_admin_analytics(db)


@app.post("/admin/evaluation-report")
def generate_admin_evaluation_report_endpoint(
    payload: EvaluationReportRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.evaluation_reports import generate_versioned_evaluation_report

    return {
        "message": "Versioned evaluation report generated.",
        "result": generate_versioned_evaluation_report(
            db=db,
            output_root=payload.output_root,
            run_id=payload.run_id,
        ),
    }


@app.get("/patients/{patient_id}/chat")
def get_patient_chat(patient_id: str, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "patient_id": patient_id,
        "messages": get_chat_messages(db, patient_id, limit=50),
    }


@app.post("/patients/{patient_id}/chat")
def chat_with_patient_agent(patient_id: str, payload: PatientChatRequest, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validate_chat_message(payload.message)
        result = handle_patient_chat(db, patient_id, payload.message)
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
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/chat")) from exc

    return result


@app.get("/me/chat")
def get_my_patient_chat(context=Depends(get_patient_access_context), db: Session = Depends(get_db)):
    return {
        "patient_id": context.patient_id,
        "messages": get_chat_messages(db, context.patient_id, limit=50),
    }


@app.post("/me/chat")
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
            db=db,
            event_type="chat_error",
            actor_role=context.role,
            patient_id=context.patient_id,
            route="/me/chat",
            status="error",
            input_payload={"message": payload.message},
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/me/chat")) from exc

    return result


@app.get("/me/uploads")
def get_my_uploads(context=Depends(get_patient_access_context), db: Session = Depends(get_db)):
    return {
        "patient_id": context.patient_id,
        "uploads": get_patient_uploads(db, context.patient_id, limit=50),
    }


@app.post("/me/uploads")
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

    return {
        "message": "Upload saved to patient record",
        "upload": upload,
    }


@app.get("/import-schema")
def get_import_schema():
    return {
        "supported_import_types": sorted(SUPPORTED_IMPORT_TYPES),
        "supported_datasets": sorted(DATASET_ADAPTERS.keys()),
        "data_dictionary": "Data/breast_monitoring_data_dictionary.md",
    }


@app.post("/import-csv")
def import_csv_payload(payload: CSVImportRequest, db: Session = Depends(get_db)):
    if not payload.csv_text and not payload.file_path:
        raise HTTPException(status_code=400, detail="Provide csv_text or file_path")

    try:
        result = import_csv(
            db=db,
            import_type=payload.import_type,
            csv_text=payload.csv_text,
            file_path=payload.file_path,
            dataset=payload.dataset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "CSV import completed",
        "import_type": payload.import_type,
        "dataset": payload.dataset,
        "result": result,
    }


@app.post("/import-qin-breast-02")
def import_qin_breast_02(payload: QINBreast02ImportRequest, db: Session = Depends(get_db)):
    try:
        result = import_qin_breast_02_clinical_xlsx(
            db=db,
            file_path=payload.clinical_xlsx_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "QIN-BREAST-02 clinical workbook import completed",
        "result": result,
    }


@app.post("/generate-qin-synthetic-cbc")
def generate_qin_synthetic_cbc(db: Session = Depends(get_db)):
    return {
        "message": "Synthetic QIN CBC generation completed",
        "result": generate_synthetic_cbc_for_qin_patients(db),
        "warning": "These CBC values are synthetic demo data and are not part of QIN-BREAST-02.",
    }


@app.post("/generate-synthetic-breast-journeys")
def generate_synthetic_breast_journeys(payload: SyntheticJourneyRequest, db: Session = Depends(get_db)):
    if payload.count < 1 or payload.count > 500:
        raise HTTPException(status_code=400, detail="count must be between 1 and 500")

    return {
        "message": "Synthetic breast cancer longitudinal journeys generated",
        "result": generate_synthetic_breast_cancer_journeys(
            db=db,
            count=payload.count,
            seed=payload.seed,
        ),
        "warning": "Synthetic journeys are for software testing and demos only, not model validation or clinical evidence.",
    }


@app.post("/generate-temporal-synthetic-breast-journeys")
def generate_temporal_synthetic_breast_journeys(payload: TemporalSyntheticJourneyRequest, db: Session = Depends(get_db)):
    if payload.count < 1 or payload.count > 300:
        raise HTTPException(status_code=400, detail="count must be between 1 and 300")
    if payload.cycles < 2 or payload.cycles > 8:
        raise HTTPException(status_code=400, detail="cycles must be between 2 and 8")

    from backend.services.synthetic_journey import generate_temporal_breast_cancer_journeys

    return {
        "message": "Temporal synthetic breast cancer journeys generated",
        "result": generate_temporal_breast_cancer_journeys(
            db=db,
            count=payload.count,
            seed=payload.seed,
            cycles=payload.cycles,
        ),
        "warning": "Synthetic temporal journeys are for engineering demos and model practice only, not clinical evidence.",
    }


@app.post("/generate-complete-synthetic-breast-dataset")
def generate_complete_synthetic_breast_dataset_endpoint(
    payload: CompleteSyntheticDatasetRequest,
    db: Session = Depends(get_db),
):
    if payload.count < 1 or payload.count > 1000:
        raise HTTPException(status_code=400, detail="count must be between 1 and 1000")
    if payload.cycles < 2 or payload.cycles > 10:
        raise HTTPException(status_code=400, detail="cycles must be between 2 and 10")
    if payload.missing_rate < 0 or payload.missing_rate > 0.35:
        raise HTTPException(status_code=400, detail="missing_rate must be between 0 and 0.35")
    if payload.noise_level < 0 or payload.noise_level > 0.25:
        raise HTTPException(status_code=400, detail="noise_level must be between 0 and 0.25")
    if not payload.patient_prefix or len(payload.patient_prefix) > 32:
        raise HTTPException(status_code=400, detail="patient_prefix is required and must be 32 characters or less")

    from backend.services.complete_synthetic_dataset import generate_complete_synthetic_breast_dataset

    return {
        "message": "Complete synthetic breast cancer treatment dataset generated",
        "result": generate_complete_synthetic_breast_dataset(
            db=db,
            count=payload.count,
            seed=payload.seed,
            cycles=payload.cycles,
            output_dir=payload.output_dir,
            write_db=payload.write_db,
            patient_prefix=payload.patient_prefix,
            balanced_outcomes=payload.balanced_outcomes,
            missing_rate=payload.missing_rate,
            noise_level=payload.noise_level,
        ),
        "warning": "This dataset is fully synthetic and intended only for engineering demos and ML practice.",
    }


@app.post("/train-complete-synthetic-models")
def train_complete_synthetic_models_endpoint(payload: CompleteSyntheticTrainingRequest):
    allowed_targets = {
        "treatment_success_binary",
        "maintenance_needed",
        "toxicity_risk_binary",
        "support_intervention_needed",
        "urgent_intervention_needed",
    }
    if payload.target not in allowed_targets:
        raise HTTPException(status_code=400, detail=f"target must be one of {sorted(allowed_targets)}")
    if payload.test_size <= 0 or payload.test_size >= 0.5:
        raise HTTPException(status_code=400, detail="test_size must be greater than 0 and less than 0.5")
    if payload.cnn_epochs < 1 or payload.cnn_epochs > 200:
        raise HTTPException(status_code=400, detail="cnn_epochs must be between 1 and 200")

    from backend.services.complete_synthetic_training import train_complete_synthetic_models

    try:
        result = train_complete_synthetic_models(
            ml_csv_path=payload.ml_csv_path,
            output_dir=payload.output_dir,
            target=payload.target,
            test_size=payload.test_size,
            seed=payload.seed,
            cnn_epochs=payload.cnn_epochs,
            cnn_batch_size=payload.cnn_batch_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Complete synthetic model training finished",
        "result": result,
    }


@app.post("/generate-complete-synthetic-xai")
def generate_complete_synthetic_xai_endpoint(payload: CompleteSyntheticXAIRequest):
    if payload.top_n < 1 or payload.top_n > 20:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 20")

    from backend.services.complete_synthetic_xai import generate_complete_synthetic_xai

    try:
        result = generate_complete_synthetic_xai(
            ml_csv_path=payload.ml_csv_path,
            model_path=payload.model_path,
            predictions_csv_path=payload.predictions_csv_path,
            output_json_path=payload.output_json_path,
            top_n=payload.top_n,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Complete synthetic model explanations generated",
        "result": result,
        "warning": "Synthetic XAI explains simulator-trained model behavior only, not clinical causality.",
    }


@app.post("/index-qin-mri")
def index_qin_mri(payload: MRISeriesIndexRequest, db: Session = Depends(get_db)):
    try:
        result = index_mri_series(
            db=db,
            root_path=payload.root_path,
            patient_id=payload.patient_id,
            max_files=payload.max_files,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "message": "MRI DICOM series index completed",
        "result": result,
    }


@app.post("/build-qin-mri-manifest")
def build_qin_mri_manifest_endpoint(payload: MRIManifestRequest, db: Session = Depends(get_db)):
    return {
        "message": "QIN-BREAST-02 MRI modeling manifest built",
        "result": build_qin_mri_manifest(
            db=db,
            clinical_xlsx_path=payload.clinical_xlsx_path,
            output_csv_path=payload.output_csv_path,
        ),
    }


@app.post("/preprocess-qin-mri-previews")
def preprocess_qin_mri_previews(payload: MRIPreviewPreprocessRequest):
    return {
        "message": "QIN-BREAST-02 MRI preview preprocessing completed",
        "result": preprocess_mri_manifest_previews(
            manifest_csv_path=payload.manifest_csv_path,
            output_dir=payload.output_dir,
        ),
    }


@app.post("/inspect-breastdcedl")
def inspect_breastdcedl(payload: BreastDCEDLInspectRequest):
    try:
        result = inspect_breastdcedl_dataset(payload.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "BreastDCEDL local dataset inspection completed",
        "result": result,
    }


@app.post("/build-breastdcedl-manifest")
def build_breastdcedl_manifest_endpoint(payload: BreastDCEDLManifestRequest):
    try:
        result = build_breastdcedl_manifest(
            root_path=payload.root_path,
            output_csv_path=payload.output_csv_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "message": "BreastDCEDL manifest built",
        "result": result,
    }


@app.post("/run-breastdcedl-baseline")
def run_breastdcedl_baseline_endpoint(payload: BreastDCEDLBaselineRequest):
    from backend.services.breastdcedl_baseline import run_breastdcedl_baseline

    try:
        result = run_breastdcedl_baseline(
            manifest_csv_path=payload.manifest_csv_path,
            features_csv_path=payload.features_csv_path,
            metrics_json_path=payload.metrics_json_path,
            predictions_csv_path=payload.predictions_csv_path,
            max_patients=payload.max_patients,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "BreastDCEDL baseline completed",
        "result": result,
    }


@app.post("/generate-breastdcedl-previews")
def generate_breastdcedl_previews_endpoint(payload: BreastDCEDLPreviewRequest):
    from backend.services.breastdcedl_previews import generate_breastdcedl_previews

    return {
        "message": "BreastDCEDL preview generation completed",
        "result": generate_breastdcedl_previews(
            manifest_csv_path=payload.manifest_csv_path,
            output_dir=payload.output_dir,
            max_patients=payload.max_patients,
        ),
    }


@app.post("/import-breastdcedl-patients")
def import_breastdcedl_patients_endpoint(payload: BreastDCEDLImportRequest, db: Session = Depends(get_db)):
    from backend.services.breastdcedl_importer import import_breastdcedl_patients_to_dashboard

    return {
        "message": "BreastDCEDL patients imported to dashboard",
        "result": import_breastdcedl_patients_to_dashboard(
            db=db,
            manifest_csv_path=payload.manifest_csv_path,
            limit=payload.limit,
        ),
    }


@app.post("/run-breastdcedl-cnn")
def run_breastdcedl_cnn_endpoint(payload: BreastDCEDLCNNRequest):
    from backend.services.breastdcedl_cnn import run_breastdcedl_small_cnn

    try:
        result = run_breastdcedl_small_cnn(
            manifest_csv_path=payload.manifest_csv_path,
            metrics_json_path=payload.metrics_json_path,
            max_patients=payload.max_patients,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "BreastDCEDL small CNN baseline completed",
        "result": result,
    }


@app.post("/generate-breastdcedl-xai")
def generate_breastdcedl_xai_endpoint(payload: BreastDCEDLXAIRequest):
    from backend.services.breastdcedl_xai import generate_breastdcedl_shap_explanations

    return {
        "message": "BreastDCEDL SHAP explanations generated",
        "result": generate_breastdcedl_shap_explanations(
            features_csv_path=payload.features_csv_path,
            output_json_path=payload.output_json_path,
            top_n=payload.top_n,
        ),
    }


@app.post("/models/breastdcedl/train-final")
def train_breastdcedl_final_model_endpoint(payload: BreastDCEDLModelTrainRequest, db: Session = Depends(get_db)):
    from backend.services.model_artifacts import train_and_register_breastdcedl_model

    try:
        result = train_and_register_breastdcedl_model(
            db=db,
            version=payload.version,
            features_csv_path=payload.features_csv_path,
            metrics_path=payload.metrics_path,
            artifact_dir=payload.artifact_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.get("/models")
def list_models_endpoint(db: Session = Depends(get_db)):
    from backend.services.model_artifacts import list_registered_models

    return {"models": list_registered_models(db)}


@app.post("/models/complete-synthetic/register-champion")
def register_complete_synthetic_champion_endpoint(
    payload: CompleteSyntheticRegisterRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.model_artifacts import register_complete_synthetic_champion

    try:
        result = register_complete_synthetic_champion(
            db=db,
            version=payload.version,
            metrics_path=payload.metrics_path,
            training_data_path=payload.training_data_path,
            artifact_dir=payload.artifact_dir,
            promotion_status=payload.promotion_status,
            promotion_reason=payload.promotion_reason,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Complete synthetic champion registered.",
        "model": result,
        "warning": "Synthetic champion registration is for MLOps practice only, not clinical deployment.",
    }


@app.post("/models/{model_name}/{model_version}/promote")
def promote_model_version_endpoint(
    model_name: str,
    model_version: str,
    payload: ModelVersionActionRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.model_artifacts import promote_model_version

    try:
        model = promote_model_version(db, model_name=model_name, model_version=model_version, reason=payload.reason)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="model_lifecycle",
            actor_role=context.role,
            route="/models/{model_name}/{model_version}/promote",
            status="error",
            input_payload={"model_name": model_name, "model_version": model_version, "reason": payload.reason},
            error_message=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "message": "Model version promoted to champion.",
        "model": model,
        "safety_note": "Promotion is an MLOps registry action. It does not imply clinical validation.",
    }


@app.post("/models/{model_name}/{model_version}/rollback")
def rollback_model_version_endpoint(
    model_name: str,
    model_version: str,
    payload: ModelVersionActionRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.model_artifacts import rollback_model_version

    try:
        model = rollback_model_version(db, model_name=model_name, rollback_to_version=model_version, reason=payload.reason)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="model_lifecycle",
            actor_role=context.role,
            route="/models/{model_name}/{model_version}/rollback",
            status="error",
            input_payload={"model_name": model_name, "model_version": model_version, "reason": payload.reason},
            error_message=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "message": "Model rollback completed.",
        "model": model,
        "safety_note": "Rollback changes serving registry status only. Clinical use still requires review.",
    }


@app.post("/models/breastdcedl/predict/{patient_id}")
def predict_breastdcedl_patient_endpoint(
    patient_id: str,
    payload: BreastDCEDLModelPredictRequest,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    from backend.services.model_artifacts import predict_breastdcedl_patient

    try:
        result = predict_breastdcedl_patient(
            db=db,
            patient_id=patient_id,
            model_name=payload.model_name,
            model_version=payload.model_version,
            features_csv_path=payload.features_csv_path,
            shap_json_path=payload.shap_json_path,
        )
    except FileNotFoundError as exc:
        log_app_event(
            db=db,
            event_type="prediction",
            patient_id=patient_id,
            route="/models/breastdcedl/predict/{patient_id}",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="prediction",
            patient_id=patient_id,
            route="/models/breastdcedl/predict/{patient_id}",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.get("/prediction-audits")
def list_prediction_audits_endpoint(patient_id: str | None = None, limit: int = 50, db: Session = Depends(get_db)):
    from backend.services.model_artifacts import get_prediction_audit_logs

    safe_limit = max(1, min(limit, 200))
    return {"prediction_audits": get_prediction_audit_logs(db, patient_id=patient_id, limit=safe_limit)}


@app.post("/patients/{patient_id}/labs")
def add_lab_result(patient_id: str, payload: LabCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validation_warnings = validate_cbc_values(payload.wbc, payload.hemoglobin, payload.platelets)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/patients/{patient_id}/labs",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/labs")) from exc

    lab = LabResult(
        patient_id=patient_id,
        date=payload.date,
        wbc=payload.wbc,
        hemoglobin=payload.hemoglobin,
        platelets=payload.platelets,
        source=payload.source or "manual",
        source_note=payload.source_note,
    )

    db.add(lab)
    db.commit()

    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/patients/{patient_id}/labs",
        status="ok",
        input_payload=payload.dict(),
        output_payload={"lab_id": lab.id, "warning_count": len(validation_warnings)},
    )

    return {
        "message": "Lab result added",
        "validation_warnings": validation_warnings,
        "error_state": None,
    }


@app.post("/patients/{patient_id}/treatments")
def add_treatment(patient_id: str, payload: TreatmentCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validation_warnings = validate_treatment_payload(payload.cycle, payload.drug)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/patients/{patient_id}/treatments",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/treatments")) from exc

    treatment = Treatment(
        patient_id=patient_id,
        date=payload.date,
        cycle=payload.cycle,
        drug=payload.drug,
    )

    db.add(treatment)
    db.commit()

    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/patients/{patient_id}/treatments",
        status="ok",
        input_payload=payload.dict(),
        output_payload={"treatment_id": treatment.id},
    )

    return {"message": "Treatment added", "validation_warnings": validation_warnings, "error_state": None}


@app.post("/patients/{patient_id}/symptoms")
def add_symptom_report(patient_id: str, payload: SymptomCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validation_warnings = validate_symptom_payload(payload.symptom, payload.severity, payload.notes)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/patients/{patient_id}/symptoms",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/symptoms")) from exc

    symptom = SymptomReport(
        patient_id=patient_id,
        date=payload.date,
        symptom=payload.symptom,
        severity=payload.severity,
        notes=payload.notes,
    )

    db.add(symptom)
    db.commit()

    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/patients/{patient_id}/symptoms",
        status="ok",
        input_payload=payload.dict(),
        output_payload={"symptom_id": symptom.id, "warning_count": len(validation_warnings)},
    )

    return {"message": "Symptom report added", "validation_warnings": validation_warnings, "error_state": None}


@app.post("/patients/{patient_id}/imaging-reports")
def add_imaging_report(patient_id: str, payload: ImagingReportCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validation_warnings = validate_imaging_report_payload(
            payload.modality,
            payload.report_type,
            payload.findings,
            payload.impression,
            payload.body_site,
        )
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/patients/{patient_id}/imaging-reports",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/imaging-reports")) from exc

    report = ImagingReport(
        patient_id=patient_id,
        date=payload.date,
        modality=payload.modality,
        report_type=payload.report_type,
        body_site=payload.body_site,
        findings=payload.findings,
        impression=payload.impression,
    )

    db.add(report)
    db.commit()

    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/patients/{patient_id}/imaging-reports",
        status="ok",
        input_payload={**payload.dict(), "findings": "[redacted report text]", "impression": "[redacted report text]"},
        output_payload={"imaging_report_id": report.id, "warning_count": len(validation_warnings)},
    )

    return {"message": "Imaging report added", "validation_warnings": validation_warnings, "error_state": None}


@app.post("/patients/{patient_id}/mri-registry")
def add_mri_registry_entry(patient_id: str, payload: MRIRegistryCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    entry = MRIFileRegistry(
        patient_id=patient_id,
        scan_date=payload.scan_date,
        modality=payload.modality,
        series_description=payload.series_description,
        local_path=payload.local_path,
        notes=payload.notes,
    )

    db.add(entry)
    db.commit()

    return {"message": "MRI registry entry added", "id": entry.id}


@app.post("/patients/{patient_id}/ct-reports")
def add_ct_report(patient_id: str, payload: CTReportCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        validation_warnings = validate_imaging_report_payload(
            "CT",
            payload.report_type,
            payload.findings,
            payload.impression,
            body_site="Chest/abdomen/pelvis",
        )
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="validation_error",
            patient_id=patient_id,
            route="/patients/{patient_id}/ct-reports",
            status="error",
            input_payload={**payload.dict(), "findings": "[redacted report text]", "impression": "[redacted report text]"},
            error_message=str(exc),
        )
        raise HTTPException(status_code=400, detail=validation_error_payload(exc, route="/patients/{patient_id}/ct-reports")) from exc

    report = CTReport(
        patient_id=patient_id,
        date=payload.date,
        report_type=payload.report_type,
        findings=payload.findings,
        impression=payload.impression,
    )

    db.add(report)
    db.commit()

    log_app_event(
        db=db,
        event_type="patient_input",
        patient_id=patient_id,
        route="/patients/{patient_id}/ct-reports",
        status="ok",
        input_payload={**payload.dict(), "findings": "[redacted report text]", "impression": "[redacted report text]"},
        output_payload={"ct_report_id": report.id, "warning_count": len(validation_warnings)},
    )

    return {"message": "CT report added", "validation_warnings": validation_warnings, "error_state": None}


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
