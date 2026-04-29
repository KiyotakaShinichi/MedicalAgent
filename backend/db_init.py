from datetime import datetime

import pandas as pd

from backend.database import Base, SessionLocal, engine
from backend.models import CtReport, LabResult, Patient, Treatment


PATIENT_ID = "P001"


def seed_patient(session):
    existing = session.query(Patient).filter(Patient.id == PATIENT_ID).first()
    if existing:
        return

    patient = Patient(
        id=PATIENT_ID,
        name_code=PATIENT_ID,
        diagnosis="Lung cancer",
        created_at=datetime.utcnow(),
    )
    session.add(patient)


def seed_labs(session):
    if session.query(LabResult).count() > 0:
        return

    labs = pd.read_csv("Data/labs.csv", parse_dates=["date"])
    for row in labs.to_dict(orient="records"):
        session.add(
            LabResult(
                patient_id=PATIENT_ID,
                date=row["date"].date(),
                wbc=float(row["wbc"]),
                hemoglobin=float(row["hemoglobin"]),
                platelets=float(row["platelets"]),
            )
        )


def seed_treatments(session):
    if session.query(Treatment).count() > 0:
        return

    treatments = pd.read_csv("Data/treatment.csv", parse_dates=["date"])
    for row in treatments.to_dict(orient="records"):
        session.add(
            Treatment(
                patient_id=PATIENT_ID,
                date=row["date"].date(),
                cycle=int(row["cycle"]),
                drug=row["drug"],
            )
        )


def seed_ct_reports(session):
    if session.query(CtReport).count() > 0:
        return

    reports = pd.read_csv("Data/ct_reports.csv", parse_dates=["date"])
    for row in reports.to_dict(orient="records"):
        session.add(
            CtReport(
                patient_id=PATIENT_ID,
                date=row["date"].date(),
                report_type=row["report_type"],
                findings=row["findings"],
                impression=row["impression"],
            )
        )


def init_db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        seed_patient(session)
        seed_labs(session)
        seed_treatments(session)
        seed_ct_reports(session)
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    print("SQLite database initialized and seeded.")
