import pandas as pd

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


def get_all_patients(db):
    return db.query(Patient).all()


def get_patient(db, patient_id):
    return db.query(Patient).filter(Patient.id == patient_id).first()


def get_breast_cancer_profile(db, patient_id):
    return (
        db.query(BreastCancerProfile)
        .filter(BreastCancerProfile.patient_id == patient_id)
        .first()
    )


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
            "source": row.source,
            "source_note": row.source_note,
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


def get_symptoms_df(db, patient_id):
    rows = (
        db.query(SymptomReport)
        .filter(SymptomReport.patient_id == patient_id)
        .order_by(SymptomReport.date)
        .all()
    )

    data = [
        {
            "date": row.date,
            "symptom": row.symptom,
            "severity": row.severity,
            "notes": row.notes,
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


def get_imaging_reports_df(db, patient_id):
    rows = (
        db.query(ImagingReport)
        .filter(ImagingReport.patient_id == patient_id)
        .order_by(ImagingReport.date)
        .all()
    )

    data = [
        {
            "date": row.date,
            "modality": row.modality,
            "report_type": row.report_type,
            "body_site": row.body_site,
            "findings": row.findings,
            "impression": row.impression,
        }
        for row in rows
    ]

    return pd.DataFrame(data)


def get_mri_registry(db, patient_id):
    rows = (
        db.query(MRIFileRegistry)
        .filter(MRIFileRegistry.patient_id == patient_id)
        .order_by(MRIFileRegistry.scan_date, MRIFileRegistry.id)
        .all()
    )

    return [
        {
            "id": row.id,
            "patient_id": row.patient_id,
            "scan_date": str(row.scan_date) if row.scan_date else None,
            "modality": row.modality,
            "series_description": row.series_description,
            "local_path": row.local_path,
            "notes": row.notes,
        }
        for row in rows
    ]
