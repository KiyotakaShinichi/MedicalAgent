from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.crud import (
    get_all_patients,
    get_breast_cancer_profile,
    get_chat_messages,
    get_ct_reports_df,
    get_imaging_reports_df,
    get_labs_df,
    get_medication_logs,
    get_mri_registry,
    get_mri_series_index,
    get_patient,
    get_symptoms_df,
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
from backend.processing.risk_engine import detect_risks, detect_symptom_risks, detect_trend_risk
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
from backend.services.support_chat_agent import handle_patient_chat

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


class PatientChatRequest(BaseModel):
    message: str

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/frontend/index.html")


# Serve frontend static files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


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
    breast_profile = get_breast_cancer_profile(db, patient_id)

    trends = {}
    risks = []
    trend_risks = []
    if not labs.empty:
        trends = analyze_labs(labs)
        risks = detect_risks(labs)
        trend_risks = detect_trend_risk(labs)
    symptom_risks = detect_symptom_risks(symptoms)

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

    all_risks = risks + trend_risks + symptom_risks + radiology_risks
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
    report["multimodal_assessment"] = build_multimodal_assessment(patient.id, report)

    return report


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
        result = handle_patient_chat(db, patient_id, payload.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


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


@app.post("/patients/{patient_id}/labs")
def add_lab_result(patient_id: str, payload: LabCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

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

    return {"message": "Lab result added"}


@app.post("/patients/{patient_id}/treatments")
def add_treatment(patient_id: str, payload: TreatmentCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    treatment = Treatment(
        patient_id=patient_id,
        date=payload.date,
        cycle=payload.cycle,
        drug=payload.drug,
    )

    db.add(treatment)
    db.commit()

    return {"message": "Treatment added"}


@app.post("/patients/{patient_id}/symptoms")
def add_symptom_report(patient_id: str, payload: SymptomCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if payload.severity < 0 or payload.severity > 10:
        raise HTTPException(status_code=400, detail="Severity must be between 0 and 10")

    symptom = SymptomReport(
        patient_id=patient_id,
        date=payload.date,
        symptom=payload.symptom,
        severity=payload.severity,
        notes=payload.notes,
    )

    db.add(symptom)
    db.commit()

    return {"message": "Symptom report added"}


@app.post("/patients/{patient_id}/imaging-reports")
def add_imaging_report(patient_id: str, payload: ImagingReportCreate, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

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

    return {"message": "Imaging report added"}


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

    report = CTReport(
        patient_id=patient_id,
        date=payload.date,
        report_type=payload.report_type,
        findings=payload.findings,
        impression=payload.impression,
    )

    db.add(report)
    db.commit()

    return {"message": "CT report added"}


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
