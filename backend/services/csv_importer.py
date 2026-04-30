from datetime import date, datetime
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
    "qin_breast_02": {
        "patient_id": ["patient_id", "Patient ID", "Subject ID", "Subject", "Participant ID"],
        "cancer_stage": ["cancer_stage", "Stage", "Clinical stage"],
        "er_status": ["er_status", "ER", "ER status", "Estrogen receptor"],
        "pr_status": ["pr_status", "PR", "PR status", "Progesterone receptor"],
        "her2_status": ["her2_status", "HER2", "HER2 status"],
        "molecular_subtype": ["molecular_subtype", "Subtype", "Tumor subtype"],
        "treatment_intent": ["treatment_intent", "Treatment", "Treatment regimen", "Treatment Arm"],
        "date": ["date", "Study Date", "Scan Date", "Visit Date"],
        "modality": ["modality", "Modality", "Image Modality"],
        "report_type": ["report_type", "Visit", "Time Point", "Scan Time Point"],
        "body_site": ["body_site", "Body Site", "Primary Site"],
        "findings": ["findings", "Findings", "Report", "Notes"],
        "impression": ["impression", "Impression", "Assessment"],
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


def import_qin_breast_02_clinical_xlsx(db, file_path):
    df = pd.read_excel(file_path)
    df = df[df["NBIA ID"].astype(str).str.startswith("QIN-BREAST-02")]

    patients_created = 0
    profiles_created = 0
    profiles_updated = 0
    treatments_created = 0
    imaging_reports_created = 0

    for _, row in df.iterrows():
        patient_id = str(_value(row, "NBIA ID"))
        _, was_created = _get_or_create_patient(
            db,
            patient_id=patient_id,
            name=f"QIN-BREAST-02 {patient_id[-4:]}",
            diagnosis="Breast cancer - doctor-confirmed",
        )
        patients_created += 1 if was_created else 0

        profile = (
            db.query(BreastCancerProfile)
            .filter(BreastCancerProfile.patient_id == patient_id)
            .first()
        )
        if not profile:
            profile = BreastCancerProfile(patient_id=patient_id)
            db.add(profile)
            profiles_created += 1
        else:
            profiles_updated += 1

        profile.cancer_stage = _value(row, "Clinical stage ", profile.cancer_stage)
        profile.er_status = _value(row, "ER status ", profile.er_status)
        profile.pr_status = _value(row, "PR status", profile.pr_status)
        profile.her2_status = _coalesce(
            _value(row, "HER2-Neu status by FISH"),
            _value(row, "HER2-Neu status by IHC"),
            profile.her2_status,
        )
        profile.molecular_subtype = _infer_subtype(
            profile.er_status,
            profile.pr_status,
            profile.her2_status,
        )
        profile.treatment_intent = "neoadjuvant therapy monitoring"

        affected_breast = _value(row, "Affected breast ", "breast")
        tumor_size = _value(row, "Size (cm)  ")
        response = _value(row, "Response")
        report_date = _parse_date(_value(row, "Date of diagnosis"))
        report_text = f"{affected_breast} breast tumor measuring {tumor_size} cm. Clinical stage {profile.cancer_stage}."
        impression = f"QIN-BREAST-02 baseline clinical imaging metadata. Recorded response outcome: {response or 'not available'}."

        if not _imaging_report_exists(db, patient_id, report_date, "Baseline QIN-BREAST-02 metadata"):
            db.add(ImagingReport(
                patient_id=patient_id,
                date=report_date,
                modality="Breast MRI",
                report_type="Baseline QIN-BREAST-02 metadata",
                body_site="Breast",
                findings=report_text,
                impression=impression,
            ))
            imaging_reports_created += 1

        for cycle, agent_number in enumerate(range(1, 6), start=1):
            agent_col = "NAC Agent  #5" if agent_number == 5 else f"NAC Agent #{agent_number}"
            start_col = f"Start date #{agent_number}"
            agent = _value(row, agent_col)
            start_date = _value(row, start_col)
            if not agent or not start_date:
                continue

            treatment_date = _parse_date(start_date)
            if not _treatment_exists(db, patient_id, treatment_date, cycle, str(agent)):
                db.add(Treatment(
                    patient_id=patient_id,
                    date=treatment_date,
                    cycle=cycle,
                    drug=str(agent),
                ))
                treatments_created += 1

    db.commit()
    return {
        "patients_created": patients_created,
        "profiles_created": profiles_created,
        "profiles_updated": profiles_updated,
        "treatments_created": treatments_created,
        "imaging_reports_created": imaging_reports_created,
    }


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


def _coalesce(*values):
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None


def _infer_subtype(er_status, pr_status, her2_status):
    er = str(er_status or "").lower()
    pr = str(pr_status or "").lower()
    her2 = str(her2_status or "").lower()

    hormone_positive = "positive" in er or "positive" in pr
    her2_positive = ("amplified" in her2 and "not amplified" not in her2) or "positive" in her2

    if hormone_positive and her2_positive:
        return "HR-positive / HER2-positive"
    if hormone_positive and not her2_positive:
        return "HR-positive / HER2-negative"
    if not hormone_positive and her2_positive:
        return "HR-negative / HER2-positive"
    if "negative" in er and "negative" in pr and not her2_positive:
        return "Triple-negative"
    return None


def _imaging_report_exists(db, patient_id, report_date, report_type):
    return (
        db.query(ImagingReport)
        .filter(
            ImagingReport.patient_id == patient_id,
            ImagingReport.date == report_date,
            ImagingReport.report_type == report_type,
        )
        .first()
        is not None
    )


def _treatment_exists(db, patient_id, treatment_date, cycle, drug):
    return (
        db.query(Treatment)
        .filter(
            Treatment.patient_id == patient_id,
            Treatment.date == treatment_date,
            Treatment.cycle == cycle,
            Treatment.drug == drug,
        )
        .first()
        is not None
    )


def _parse_date(value):
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _get_or_create_patient(db, patient_id, name=None, diagnosis=None):
    patient = db.get(Patient, patient_id)
    if patient:
        return patient, False

    patient = Patient(
        id=patient_id,
        name=name or f"Patient {patient_id}",
        diagnosis=diagnosis or "Breast cancer - doctor-confirmed",
    )
    db.add(patient)
    db.flush()
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
            source=str(_value(row, "source", "imported_csv")),
            source_note=_value(row, "source_note"),
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
