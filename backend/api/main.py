from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.crud import (
    get_all_patients,
    get_ct_reports_df,
    get_labs_df,
    get_patient,
    get_treatments_df,
)
from backend.database import SessionLocal
from backend.models import CTReport, LabResult, Patient, Treatment
from backend.processing.radiology_analysis import analyze_radiology_reports
from backend.processing.risk_engine import detect_risks, detect_trend_risk
from backend.processing.treatment_analysis import align_labs_with_treatment
from backend.processing.trend_analysis import analyze_labs
from backend.processing.clinical_summary import generate_clinical_summary
from backend.reports.patient_report import build_patient_report

app = FastAPI(title="AI Oncology Monitoring System")

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


class LabCreate(BaseModel):
    date: date
    wbc: float
    hemoglobin: float
    platelets: float


class TreatmentCreate(BaseModel):
    date: date
    cycle: int
    drug: str


class CTReportCreate(BaseModel):
    date: date
    report_type: str
    findings: str
    impression: str

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
        diagnosis=payload.diagnosis,
    )

    db.add(patient)
    db.commit()

    return {"message": "Patient created", "patient_id": patient.id}

@app.get("/patient-report/{patient_id}")
def generate_patient_report(patient_id: str, db: Session = Depends(get_db)):
    patient = get_patient(db, patient_id)

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    labs = get_labs_df(db, patient_id)
    treatments = get_treatments_df(db, patient_id)
    ct_reports = get_ct_reports_df(db, patient_id)

    if labs.empty:
        raise HTTPException(status_code=400, detail="No lab data found for patient")

    trends = analyze_labs(labs)
    risks = detect_risks(labs)
    trend_risks = detect_trend_risk(labs)

    treatment_effects = []
    if not treatments.empty:
        treatment_effects = align_labs_with_treatment(labs, treatments)

    radiology_summary = None
    if not ct_reports.empty:
        radiology_summary = analyze_radiology_reports(ct_reports)

    all_risks = risks + trend_risks
    summary = generate_clinical_summary(
        trends=trends,
        risks=all_risks,
        trend_risks=trend_risks,
        treatment_effects=treatment_effects,
        radiology_summary=radiology_summary,
    )

    report = build_patient_report(
        labs=labs,
        trends=trends,
        risks=risks,
        trend_risks=trend_risks,
        treatment_effects=treatment_effects,
        radiology_summary=radiology_summary,
        ai_summary=summary,
    )

    report["patient_id"] = patient.id
    report["patient_name"] = patient.name
    report["diagnosis"] = patient.diagnosis

    return report


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
