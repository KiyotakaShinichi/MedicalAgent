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
    get_ct_reports_df,
    get_imaging_reports_df,
    get_labs_df,
    get_patient,
    get_symptoms_df,
    get_treatments_df,
)
from backend.database import Base, SessionLocal, engine
from backend.models import (
    BreastCancerProfile,
    CTReport,
    ImagingReport,
    LabResult,
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
from backend.services.csv_importer import DATASET_ADAPTERS, SUPPORTED_IMPORT_TYPES, import_csv

app = FastAPI(title="AI Breast Cancer Monitoring System")
Base.metadata.create_all(bind=engine)

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
    if not treatments.empty:
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

    return report


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
