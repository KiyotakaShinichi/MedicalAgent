import pandas as pd

from backend.models import Patient, LabResult, Treatment, CTReport


def get_all_patients(db):
    return db.query(Patient).all()


def get_patient(db, patient_id):
    return db.query(Patient).filter(Patient.id == patient_id).first()


def get_labs_df(db, patient_id):
    rows = (
        db.query(LabResult)
        .filter(LabResult.patient_id == patient_id)
        .order_by(LabResult.date)
        .all()
    )

    data = [
        {
            "date": row.date,
            "wbc": row.wbc,
            "hemoglobin": row.hemoglobin,
            "platelets": row.platelets,
        }
        for row in rows
    ]

    return pd.DataFrame(data)


def get_treatments_df(db, patient_id):
    rows = (
        db.query(Treatment)
        .filter(Treatment.patient_id == patient_id)
        .order_by(Treatment.date)
        .all()
    )

    data = [
        {
            "date": row.date,
            "cycle": row.cycle,
            "drug": row.drug,
        }
        for row in rows
    ]

    return pd.DataFrame(data)


def get_ct_reports_df(db, patient_id):
    rows = (
        db.query(CTReport)
        .filter(CTReport.patient_id == patient_id)
        .order_by(CTReport.date)
        .all()
    )

    data = [
        {
            "date": row.date,
            "report_type": row.report_type,
            "findings": row.findings,
            "impression": row.impression,
        }
        for row in rows
    ]

    return pd.DataFrame(data)
