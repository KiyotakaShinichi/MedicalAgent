"""Clinical ontology and data dictionary for NLCare.

This is not a clinical terminology server. It is a compact, versioned
engineering dictionary that keeps patient forms, agent tools, validation,
and documentation aligned on allowed values and field meanings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ONTOLOGY_VERSION = "clinical_ontology_v1_2026_05"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    allowed_values: tuple[str, ...] = ()
    required: bool = False
    description: str = ""
    patient_label: str = ""
    clinician_label: str = ""
    units: tuple[str, ...] = ()
    reference_range: str | None = None
    source_type: str = "structured_patient_or_report_field"
    allowed_use: tuple[str, ...] = ("record_organization", "clinician_review")
    blocked_claims: tuple[str, ...] = ("diagnosis", "treatment_recommendation")
    review_notes: str = ""
    claim_boundary: str = "Record organization only; not clinical interpretation."

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allowed_values": list(self.allowed_values),
            "required": self.required,
            "description": self.description,
            "patient_label": self.patient_label or self.name.replace("_", " ").title(),
            "clinician_label": self.clinician_label or self.name,
            "units": list(self.units),
            "reference_range": self.reference_range,
            "source_type": self.source_type,
            "allowed_use": list(self.allowed_use),
            "blocked_claims": list(self.blocked_claims),
            "review_notes": self.review_notes,
            "claim_boundary": self.claim_boundary,
        }


CLINICAL_ONTOLOGY: dict[str, dict[str, FieldSpec]] = {
    "lab": {
        "wbc": FieldSpec("wbc", patient_label="White blood cells", clinician_label="WBC", units=("10^9/L", "K/uL"), reference_range="population default varies by lab", review_notes="Interpret with ANC, symptoms, treatment timing, and local lab range."),
        "anc": FieldSpec("anc", patient_label="Absolute neutrophil count", clinician_label="ANC", units=("10^9/L", "K/uL"), reference_range="population default varies by lab", review_notes="Low ANC with fever is a review/escalation pattern."),
        "hemoglobin": FieldSpec("hemoglobin", patient_label="Hemoglobin", clinician_label="Hgb", units=("g/dL",), reference_range="sex/age/lab dependent", review_notes="Population default range is not personalized."),
        "platelets": FieldSpec("platelets", patient_label="Platelets", clinician_label="PLT", units=("10^9/L", "K/uL"), reference_range="lab dependent", review_notes="Low platelets plus bleeding symptoms requires clinician review."),
    },
    "symptom": {
        "symptom": FieldSpec(
            "symptom",
            allowed_values=(
                "fatigue", "nausea", "vomiting", "fever", "mouth_sores",
                "neuropathy", "diarrhea", "bleeding", "wound_discharge",
                "pain", "shortness_of_breath", "chest_pain", "cognitive_changes",
                "brain_fog", "lymphedema", "hot_flashes", "other",
            ),
            required=True,
            description="Patient-reported symptom category.",
            review_notes="Patient severity is an organizing signal, not a clinician-assigned toxicity grade.",
        ),
        "severity": FieldSpec("severity", required=True, description="0-10 patient-reported severity."),
    },
    "imaging": {
        "modality": FieldSpec(
            "modality",
            allowed_values=("mri", "ct", "ultrasound", "mammogram", "pet_ct", "dexa", "xray", "other"),
            required=True,
            description="Imaging/report modality. Images do not diagnose genetics.",
            blocked_claims=("diagnosis", "treatment_response_confirmation", "genetic_diagnosis"),
        ),
        "report_type": FieldSpec(
            "report_type",
            allowed_values=("baseline", "interim", "end_of_course", "follow_up", "unknown"),
            description="Clinical timing category for report organization.",
        ),
    },
    "biomarker": {
        "er_status": FieldSpec("er_status", allowed_values=("positive", "negative", "low_positive", "unknown")),
        "pr_status": FieldSpec("pr_status", allowed_values=("positive", "negative", "low_positive", "unknown")),
        "her2_status": FieldSpec("her2_status", allowed_values=("positive", "negative", "equivocal", "unknown")),
        "ihc_score": FieldSpec("ihc_score", allowed_values=("0", "1+", "2+", "3+", "unknown")),
        "fish_status": FieldSpec("fish_status", allowed_values=("amplified", "not_amplified", "unknown")),
        "ki67_percent": FieldSpec("ki67_percent", description="Numeric percent if shown on pathology report."),
    },
    "genetic_test": {
        "test_type": FieldSpec(
            "test_type",
            allowed_values=("germline", "somatic", "tumor_sequencing", "multigene_panel", "brca_only", "unknown"),
            required=True,
        ),
        "sample_type": FieldSpec("sample_type", allowed_values=("blood", "saliva", "tumor_tissue", "unknown")),
        "gene": FieldSpec("gene", allowed_values=("BRCA1", "BRCA2", "PALB2", "TP53", "PTEN", "CHEK2", "ATM", "other")),
        "classification": FieldSpec(
            "classification",
            allowed_values=("pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign", "unknown"),
            description="Variant classification copied from a report; not interpreted by NLCare.",
        ),
    },
    "tumor_marker": {
        "marker": FieldSpec("marker", allowed_values=("CA 15-3", "CA 27.29", "CEA", "other"), required=True),
        "value": FieldSpec("value", required=True, description="Numeric value as shown on the lab report."),
        "unit": FieldSpec("unit", description="Unit copied from the lab report."),
    },
    "supplement": {
        "name": FieldSpec(
            "name",
            allowed_values=("st_johns_wort", "turmeric_curcumin", "green_tea_extract", "ginger", "garlic", "ginkgo", "cbd_cannabis", "high_dose_vitamin_c", "antioxidant", "probiotic", "other"),
            required=True,
            description="Supplement/vitamin/herbal product reported by patient.",
            allowed_use=("interaction_safety_flag", "clinician_review", "education"),
            blocked_claims=("safe_with_chemo", "replace_treatment", "cancer_treatment_claim"),
            review_notes="Ask oncology team/pharmacist before use during active treatment.",
        ),
    },
    "medication": {
        "category": FieldSpec("category", allowed_values=("chemotherapy", "targeted_therapy", "endocrine_therapy", "immunotherapy", "supportive_care", "other")),
        "name": FieldSpec("name", description="Medication name as entered or selected."),
    },
}


def ontology_manifest() -> dict[str, Any]:
    return {
        "version": ONTOLOGY_VERSION,
        "record_types": {
            record_type: {field: spec.to_dict() for field, spec in fields.items()}
            for record_type, fields in CLINICAL_ONTOLOGY.items()
        },
        "claim_boundary": (
            "The ontology constrains structured data entry and agent extraction. "
            "It does not interpret records, diagnose, or recommend treatment."
        ),
    }


def validate_record_against_ontology(record_type: str, record: Mapping[str, Any]) -> dict[str, Any]:
    specs = CLINICAL_ONTOLOGY.get(record_type)
    if specs is None:
        return {"status": "unknown_record_type", "issues": [f"unknown_record_type:{record_type}"]}

    issues: list[str] = []
    normalized: dict[str, Any] = {}
    for field, spec in specs.items():
        value = record.get(field)
        if spec.required and (value is None or str(value).strip() == ""):
            issues.append(f"missing_required:{field}")
            continue
        if value is None:
            continue
        normalized_value = str(value).strip() if isinstance(value, str) else value
        if spec.allowed_values:
            candidate = str(normalized_value).strip()
            allowed_lower = {v.lower(): v for v in spec.allowed_values}
            if candidate.lower() not in allowed_lower:
                issues.append(f"invalid_value:{field}")
            else:
                normalized_value = allowed_lower[candidate.lower()]
        normalized[field] = normalized_value

    return {
        "status": "passed" if not issues else "needs_review",
        "record_type": record_type,
        "normalized": normalized,
        "issues": issues,
        "ontology_version": ONTOLOGY_VERSION,
    }


__all__ = [
    "CLINICAL_ONTOLOGY",
    "ONTOLOGY_VERSION",
    "ontology_manifest",
    "validate_record_against_ontology",
]
