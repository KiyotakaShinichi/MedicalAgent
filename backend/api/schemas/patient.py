"""Pydantic request contracts for patient-facing and clinician patient routes."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


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


class MySymptomCreate(BaseModel):
    date: date
    symptom: str
    severity: int
    notes: str | None = None
    duration: str | None = None
    urgent_flag: bool = False


class MyLabCreate(BaseModel):
    date: date
    wbc: float
    hemoglobin: float
    platelets: float
    anc: float | None = None
    lab_source: str | None = None
    notes: str | None = None


class MyImagingReportCreate(BaseModel):
    date: date
    modality: str
    report_type: str | None = None
    body_site: str | None = None
    findings: str | None = None
    impression: str | None = None
    notes: str | None = None


class MyMedicationCreate(BaseModel):
    medication: str
    dose: str | None = None
    frequency: str | None = None
    date: date
    side_effects: str | None = None
    notes: str | None = None


class MyTreatmentCreate(BaseModel):
    date: date
    drug: str
    cycle: int | None = None
    notes: str | None = None


class ImagingReportCreate(BaseModel):
    date: date
    modality: str
    report_type: str
    body_site: str | None = "Breast"
    findings: str
    impression: str


class CTReportCreate(BaseModel):
    date: date
    report_type: str
    findings: str
    impression: str


class MRIRegistryCreate(BaseModel):
    scan_date: date | None = None
    modality: str = "Breast MRI"
    series_description: str | None = None
    local_path: str
    notes: str | None = None


class PatientChatRequest(BaseModel):
    message: str


class AgentFeedbackRequest(BaseModel):
    chat_message_id: int | None = None
    rating: int
    thumbs_up: bool | None = None
    feedback_text: str | None = None


class FamilyHistoryCreate(BaseModel):
    relationship: str
    family_side: str | None = None
    cancer_type: str
    age_at_diagnosis: int | None = None
    relative_status: str | None = None
    multiple_relatives_affected: str | None = "unknown"
    male_breast_cancer: str | None = "unknown"
    known_familial_mutation: str | None = "unknown"
    bilateral_breast_cancer: str | None = "unknown"
    multiple_primary_cancers: str | None = "unknown"
    ancestry_ethnicity: str | None = None
    prior_breast_biopsy_atypia: str | None = "unknown"
    relation_degree: str | None = None
    notes: str | None = None


class GeneticTestRecordCreate(BaseModel):
    test_type: str
    sample_type: str | None = "unknown"
    gene: str | None = None
    variant_text: str | None = None
    classification: str | None = "unknown"
    report_date: date | None = None
    lab_provider: str | None = None
    upload_reference: str | None = None
    reviewed_by_genetic_counselor: str | None = "unknown"
    clinician_review_status: str | None = "pending"
    notes: str | None = None


class BiomarkerRecordCreate(BaseModel):
    source: str
    er_status: str | None = "unknown"
    pr_status: str | None = "unknown"
    her2_status: str | None = "unknown"
    ki67_percent: float | None = None
    grade: str | None = None
    stage: str | None = None
    report_date: date | None = None
    report_text: str | None = None
    upload_reference: str | None = None
    clinician_review_needed: str | None = "yes"


class TumorMarkerRecordCreate(BaseModel):
    marker: str
    value: float
    unit: str | None = None
    reference_range: str | None = None
    date_collected: date
    trend_direction: str | None = "unknown"
    notes: str | None = None


class GeneticReviewCreate(BaseModel):
    decision: str
    notes: str | None = None


class PatientUploadCreate(BaseModel):
    upload_type: str = "document"
    file_name: str
    content_type: str | None = None
    content_base64: str
    notes: str | None = None
    scan_date: date | None = None


__all__ = [
    "AgentFeedbackRequest",
    "BiomarkerRecordCreate",
    "CTReportCreate",
    "FamilyHistoryCreate",
    "GeneticReviewCreate",
    "GeneticTestRecordCreate",
    "ImagingReportCreate",
    "LabCreate",
    "MRIRegistryCreate",
    "MyImagingReportCreate",
    "MyLabCreate",
    "MyMedicationCreate",
    "MySymptomCreate",
    "MyTreatmentCreate",
    "PatientChatRequest",
    "PatientCreate",
    "PatientUploadCreate",
    "SymptomCreate",
    "TreatmentCreate",
    "TumorMarkerRecordCreate",
]
