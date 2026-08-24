"""Pydantic request contracts for patient-facing and clinician patient routes.

Field constraints here are the structural boundary: length, numeric range, and
type. They are enforced before any handler runs, so a request that violates one
is rejected with a 422 and never reaches business logic.

The bounds are imported from :mod:`backend.services.input_validation` rather
than written out again. That module enforces the same limits for callers who
never send a request body - the support agent constructs symptom and lab
records from a conversation - and two copies of the numbers would eventually
disagree. One declaration, two enforcement points appropriate to their layers.

What deliberately stays in that module: cross-field coherence and the
*warnings* it returns for values that are inside the accepted range but still
merit clinician review. Pydantic cannot express "accept this but flag it", and
folding clinical judgement into a request schema would put it in the wrong
layer.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator

from backend.services.input_validation import (
    CBC_LIMITS,
    CHAT_MESSAGE_MAX_LENGTH,
    NOTES_MAX_LENGTH,
    SEVERITY_MAX,
    SEVERITY_MIN,
    SYMPTOM_MAX_LENGTH,
)


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
    """Patient-entered symptom record."""

    date: date
    symptom: str = Field(
        min_length=1,
        max_length=SYMPTOM_MAX_LENGTH,
        description="Short free-text symptom label.",
    )
    severity: int = Field(
        ge=SEVERITY_MIN,
        le=SEVERITY_MAX,
        description=f"Patient-reported severity, {SEVERITY_MIN}-{SEVERITY_MAX} inclusive.",
    )
    notes: str | None = Field(default=None, max_length=NOTES_MAX_LENGTH)
    duration: str | None = Field(default=None, max_length=NOTES_MAX_LENGTH)
    urgent_flag: bool = False

    @field_validator("symptom")
    @classmethod
    def _symptom_must_not_be_blank(cls, value: str) -> str:
        """A whitespace-only label passes min_length but records nothing."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("symptom is required.")
        return cleaned


class MyLabCreate(BaseModel):
    """Patient-entered CBC record.

    The numeric bounds are the demo-data constraints from `CBC_LIMITS`, not
    clinical reference ranges. A value inside them can still be clinically
    alarming, which is what the warnings from `validate_cbc_values` are for.
    """

    date: date
    wbc: float = Field(
        ge=CBC_LIMITS["wbc"]["min"],
        le=CBC_LIMITS["wbc"]["max"],
        description=f"White blood cells, {CBC_LIMITS['wbc']['unit']}.",
    )
    hemoglobin: float = Field(
        ge=CBC_LIMITS["hemoglobin"]["min"],
        le=CBC_LIMITS["hemoglobin"]["max"],
        description=f"Haemoglobin, {CBC_LIMITS['hemoglobin']['unit']}.",
    )
    platelets: float = Field(
        ge=CBC_LIMITS["platelets"]["min"],
        le=CBC_LIMITS["platelets"]["max"],
        description=f"Platelets, {CBC_LIMITS['platelets']['unit']}.",
    )
    anc: float | None = Field(default=None, ge=0.0)
    lab_source: str | None = Field(default=None, max_length=NOTES_MAX_LENGTH)
    notes: str | None = Field(default=None, max_length=NOTES_MAX_LENGTH)


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
    """A single patient message to the support agent."""

    message: str = Field(
        min_length=1,
        max_length=CHAT_MESSAGE_MAX_LENGTH,
        description="Patient message text.",
    )

    @field_validator("message")
    @classmethod
    def _message_must_not_be_blank(cls, value: str) -> str:
        """Whitespace is not a question; reject it at the boundary."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message is required.")
        return cleaned


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
