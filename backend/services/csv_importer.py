from datetime import date
from io import StringIO

import pandas as pd

from backend.models import (
    BreastCancerProfile,
    ImagingReport,
    LabResult,
    Patient,
    SymptomReport,
    Treatment,
)


SUPPORTED_IMPORT_TYPES = {
    "patients",
    "breast_profiles",
    "labs",
    "treatments",
    "imaging_reports",
    "symptoms",
}


DATASET_ADAPTERS = {
    "canonical": {},
    "duke_breast_mri": {
        "patient_id": ["patient_id", "Patient ID", "PatientID", "Subject ID"],
        "er_status": ["er_status", "ER", "ER status"],
        "pr_status": ["pr_status", "PR", "PR status"],
        "her2_status": ["her2_status", "HER2", "HER2 status"],
        "molecular_subtype": ["molecular_subtype", "Molecular subtype", "Subtype"],
        "cancer_stage": ["cancer_stage", "Clinical stage", "Stage"],
        "modality": ["modality", "MRI sequence", "Series Description"],
        "findings": ["findings", "Radiology report", "Report", "Finding"],
        "impression": ["impression", "Impression"],
    },
    "ispy2": {
        "patient_id": ["patient_id", "SUBJECTID", "Subject ID", "Patient ID"],
        "er_status": ["er_status", "HR", "ER"],
        "her2_status": ["her2_status", "HER2"],
        "molecular_subtype": ["molecular_subtype", "HR_HER2_STATUS", "Subtype"],
        "treatment_intent": ["treatment_intent", "Arm", "Treatment Arm"],
        "modality": ["modality", "Modality"],
        "findings": ["findings", "Report", "Findings"],
        "impression": ["impression", "Impression"],
    },
    "mimic_labs": {
        "patient_id": ["patient_id", "subject_id", "hadm_id"],
        "date": ["date", "charttime", "storetime"],
        "wbc": ["wbc", "White Blood Cells", "WBC"],
        "hemoglobin": ["hemoglobin", "Hemoglobin"],
        "platelets": ["platelets", "Platelet Count", "Platelets"],
    },
}


def import_csv(db, import_type, csv_text=None, file_path=None, dataset="canonical"):
    if import_type not in SUPPORTED_IMPORT_TYPES:
        raise ValueError(f"Unsupported import type: {import_type}")

    df = _read_csv(csv_text=csv_text, file_path=file_path)
    df = _normalize_columns(df, DATASET_ADAPTERS.get(dataset, {}))

    if import_type == "patients":
        return _import_patients(db, df)
    if import_type == "breast_profiles":
        return _import_breast_profiles(db, df)
    if import_type == "labs":
        return _import_labs(db, df)
    if import_type == "treatments":
        return _import_treatments(db, df)
    if import_type == "imaging_reports":
        return _import_imaging_reports(db, df)
    if import_type == "symptoms":
        return _import_symptoms(db, df)

    raise ValueError(f"Unsupported import type: {import_type}")


def _read_csv(csv_text=None, file_path=None):
    if csv_text:
        return pd.read_csv(StringIO(csv_text))
    if file_path:
        return pd.read_csv(file_path)
    raise ValueError("Provide either csv_text or file_path")


def _normalize_columns(df, adapter):
    renamed = {}
    for canonical, candidates in adapter.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                renamed[candidate] = canonical
                break

    return df.rename(columns=renamed)


def _value(row, key, default=None):
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value


def _parse_date(value):
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _get_or_create_patient(db, patient_id, name=None, diagnosis=None):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient:
        return patient, False

    patient = Patient(
        id=patient_id,
        name=name or f"Patient {patient_id}",
        diagnosis=diagnosis or "Breast cancer - doctor-confirmed",
    )
    db.add(patient)
    return patient, True


def _import_patients(db, df):
    _require_columns(df, ["patient_id"])
    created = 0
    updated = 0

    for _, row in df.iterrows():
        patient_id = str(_value(row, "patient_id"))
        patient, was_created = _get_or_create_patient(
            db,
            patient_id=patient_id,
            name=_value(row, "name"),
            diagnosis=_value(row, "diagnosis"),
        )
        if was_created:
            created += 1
        else:
            patient.name = _value(row, "name", patient.name)
            patient.diagnosis = _value(row, "diagnosis", patient.diagnosis)
            updated += 1

    db.commit()
    return {"created": created, "updated": updated}


def _import_breast_profiles(db, df):
    _require_columns(df, ["patient_id"])
    created = 0
    updated = 0

    for _, row in df.iterrows():
        patient_id = str(_value(row, "patient_id"))
        _get_or_create_patient(db, patient_id)
        profile = (
            db.query(BreastCancerProfile)
            .filter(BreastCancerProfile.patient_id == patient_id)
            .first()
        )
        if not profile:
            profile = BreastCancerProfile(patient_id=patient_id)
            db.add(profile)
            created += 1
        else:
            updated += 1

        profile.cancer_stage = _value(row, "cancer_stage", profile.cancer_stage)
        profile.er_status = _value(row, "er_status", profile.er_status)
        profile.pr_status = _value(row, "pr_status", profile.pr_status)
        profile.her2_status = _value(row, "her2_status", profile.her2_status)
        profile.molecular_subtype = _value(row, "molecular_subtype", profile.molecular_subtype)
        profile.treatment_intent = _value(row, "treatment_intent", profile.treatment_intent)
        profile.menopausal_status = _value(row, "menopausal_status", profile.menopausal_status)

    db.commit()
    return {"created": created, "updated": updated}


def _import_labs(db, df):
    _require_columns(df, ["patient_id", "date", "wbc", "hemoglobin", "platelets"])
    created = 0

    for _, row in df.iterrows():
        patient_id = str(_value(row, "patient_id"))
        _get_or_create_patient(db, patient_id)
        db.add(LabResult(
            patient_id=patient_id,
            date=_parse_date(_value(row, "date")),
            wbc=float(_value(row, "wbc")),
            hemoglobin=float(_value(row, "hemoglobin")),
            platelets=float(_value(row, "platelets")),
        ))
        created += 1

    db.commit()
    return {"created": created}


def _import_treatments(db, df):
    _require_columns(df, ["patient_id", "date", "cycle", "drug"])
    created = 0

    for _, row in df.iterrows():
        patient_id = str(_value(row, "patient_id"))
        _get_or_create_patient(db, patient_id)
        db.add(Treatment(
            patient_id=patient_id,
            date=_parse_date(_value(row, "date")),
            cycle=int(_value(row, "cycle")),
            drug=str(_value(row, "drug")),
        ))
        created += 1

    db.commit()
    return {"created": created}


def _import_imaging_reports(db, df):
    _require_columns(df, ["patient_id", "date", "modality", "report_type", "findings", "impression"])
    created = 0

    for _, row in df.iterrows():
        patient_id = str(_value(row, "patient_id"))
        _get_or_create_patient(db, patient_id)
        db.add(ImagingReport(
            patient_id=patient_id,
            date=_parse_date(_value(row, "date")),
            modality=str(_value(row, "modality")),
            report_type=str(_value(row, "report_type")),
            body_site=_value(row, "body_site", "Breast"),
            findings=str(_value(row, "findings")),
            impression=str(_value(row, "impression")),
        ))
        created += 1

    db.commit()
    return {"created": created}


def _import_symptoms(db, df):
    _require_columns(df, ["patient_id", "date", "symptom", "severity"])
    created = 0

    for _, row in df.iterrows():
        patient_id = str(_value(row, "patient_id"))
        _get_or_create_patient(db, patient_id)
        db.add(SymptomReport(
            patient_id=patient_id,
            date=_parse_date(_value(row, "date")),
            symptom=str(_value(row, "symptom")),
            severity=int(_value(row, "severity")),
            notes=_value(row, "notes"),
        ))
        created += 1

    db.commit()
    return {"created": created}
