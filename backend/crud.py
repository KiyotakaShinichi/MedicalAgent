import pandas as pd

from backend.models import (
    BreastCancerProfile,
    ClinicalIntervention,
    CTReport,
    ImagingReport,
    LabResult,
    ChatMessage,
    MedicationLog,
    MRIFileRegistry,
    MRISeriesIndex,
    PatientUpload,
    Patient,
    SymptomReport,
    Treatment,
    TreatmentOutcome,
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


def get_clinical_interventions(db, patient_id):
    rows = (
        db.query(ClinicalIntervention)
        .filter(ClinicalIntervention.patient_id == patient_id)
        .order_by(ClinicalIntervention.date, ClinicalIntervention.id)
        .all()
    )

    return [
        {
            "id": row.id,
            "patient_id": row.patient_id,
            "date": str(row.date),
            "intervention_type": row.intervention_type,
            "reason": row.reason,
            "medication_or_product": row.medication_or_product,
            "dose": row.dose,
            "notes": row.notes,
            "source": row.source,
        }
        for row in rows
    ]


def get_treatment_outcome(db, patient_id):
    row = (
        db.query(TreatmentOutcome)
        .filter(TreatmentOutcome.patient_id == patient_id)
        .first()
    )
    if row is None:
        return None

    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "assessment_date": str(row.assessment_date),
        "response_category": row.response_category,
        "cancer_status": row.cancer_status,
        "maintenance_plan": row.maintenance_plan,
        "recurrence_risk_band": row.recurrence_risk_band,
        "notes": row.notes,
        "source": row.source,
    }


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


def get_mri_series_index(db, patient_id):
    rows = (
        db.query(MRISeriesIndex)
        .filter(MRISeriesIndex.patient_id == patient_id)
        .order_by(MRISeriesIndex.study_date, MRISeriesIndex.candidate_role, MRISeriesIndex.series_description)
        .all()
    )

    return [
        {
            "id": row.id,
            "patient_id": row.patient_id,
            "study_date": str(row.study_date) if row.study_date else None,
            "modality": row.modality,
            "series_description": row.series_description,
            "series_uid": row.series_uid,
            "folder": row.folder,
            "instance_count": row.instance_count,
            "candidate_role": row.candidate_role,
        }
        for row in rows
    ]


def get_medication_logs(db, patient_id):
    rows = (
        db.query(MedicationLog)
        .filter(MedicationLog.patient_id == patient_id)
        .order_by(MedicationLog.date, MedicationLog.id)
        .all()
    )

    return [
        {
            "id": row.id,
            "patient_id": row.patient_id,
            "date": str(row.date),
            "medication": row.medication,
            "dose": row.dose,
            "frequency": row.frequency,
            "notes": row.notes,
            "source": row.source,
        }
        for row in rows
    ]


def get_chat_messages(db, patient_id, limit=20):
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.patient_id == patient_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": row.id,
            "patient_id": row.patient_id,
            "role": row.role,
            "message": row.message,
            "intent": row.intent,
            "saved_actions_json": row.saved_actions_json,
            "created_at": str(row.created_at),
        }
        for row in reversed(rows)
    ]


def get_patient_uploads(db, patient_id, limit=20):
    rows = (
        db.query(PatientUpload)
        .filter(PatientUpload.patient_id == patient_id)
        .order_by(PatientUpload.created_at.desc(), PatientUpload.id.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": row.id,
            "patient_id": row.patient_id,
            "upload_type": row.upload_type,
            "original_filename": row.original_filename,
            "content_type": row.content_type,
            "local_path": row.local_path,
            "notes": row.notes,
            "created_at": str(row.created_at),
        }
        for row in rows
    ]
