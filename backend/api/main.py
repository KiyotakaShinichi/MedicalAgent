from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from backend.database import SessionLocal
from backend.models import CtReport, LabResult, Patient, Treatment
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

@app.get("/")
def root():
    return {"message": "MedicalAgent API is running"}

@app.get("/patient-report/{patient_id}")
def get_patient_report(patient_id: str):
    session = SessionLocal()
    try:
        patient = session.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        labs = (
            session.query(LabResult)
            .filter(LabResult.patient_id == patient_id)
            .order_by(LabResult.date)
            .all()
        )
        treatments = (
            session.query(Treatment)
            .filter(Treatment.patient_id == patient_id)
            .order_by(Treatment.date)
            .all()
        )
        ct_reports = (
            session.query(CtReport)
            .filter(CtReport.patient_id == patient_id)
            .order_by(CtReport.date)
            .all()
        )

        if not labs:
            raise HTTPException(status_code=404, detail="No labs for patient")

        labs_df = pd.DataFrame(
            [
                {
                    "date": row.date,
                    "wbc": row.wbc,
                    "hemoglobin": row.hemoglobin,
                    "platelets": row.platelets,
                }
                for row in labs
            ]
        )
        treatments_df = pd.DataFrame(
            [
                {
                    "date": row.date,
                    "cycle": row.cycle,
                    "drug": row.drug,
                }
                for row in treatments
            ]
        )
        ct_df = pd.DataFrame(
            [
                {
                    "date": row.date,
                    "report_type": row.report_type,
                    "findings": row.findings,
                    "impression": row.impression,
                }
                for row in ct_reports
            ]
        )

        trends = analyze_labs(labs_df)
        risks = detect_risks(labs_df)
        trend_risks = detect_trend_risk(labs_df)
        treatment_effects = align_labs_with_treatment(labs_df, treatments_df)
        radiology_summary = (
            analyze_radiology_reports(ct_df) if not ct_df.empty else None
        )

        all_risks = risks + trend_risks
        summary = generate_clinical_summary(trends, all_risks, treatment_effects)

        return build_patient_report(
            labs=labs_df,
            trends=trends,
            risks=risks,
            trend_risks=trend_risks,
            treatment_effects=treatment_effects,
            radiology_summary=radiology_summary,
            ai_summary=summary,
        )
    finally:
        session.close()
