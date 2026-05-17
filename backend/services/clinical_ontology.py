"""Clinical ontology and data dictionary for OncoTrack.

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
    claim_boundary: str = "Record organization only; not clinical interpretation."

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allowed_values": list(self.allowed_values),
            "required": self.required,
            "description": self.description,
            "claim_boundary": self.claim_boundary,
        }


CLINICAL_ONTOLOGY: dict[str, dict[str, FieldSpec]] = {
    "symptom": {
        "symptom": FieldSpec(
            "symptom",
            allowed_values=(
                "fatigue", "nausea", "vomiting", "fever", "mouth_sores",
                "neuropathy", "diarrhea", "bleeding", "wound_discharge",
                "pain", "shortness_of_breath", "chest_pain", "other",
            ),
            required=True,
            description="Patient-reported symptom category.",
        ),
        "severity": FieldSpec("severity", required=True, description="0-10 patient-reported severity."),
    },
    "imaging": {
        "modality": FieldSpec(
            "modality",
            allowed_values=("mri", "ct", "ultrasound", "pet_ct", "xray", "other"),
            required=True,
            description="Imaging/report modality. Images do not diagnose genetics.",
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
            description="Variant classification copied from a report; not interpreted by OncoTrack.",
        ),
    },
    "tumor_marker": {
        "marker": FieldSpec("marker", allowed_values=("CA 15-3", "CA 27.29", "CEA", "other"), required=True),
        "value": FieldSpec("value", required=True, description="Numeric value as shown on the lab report."),
        "unit": FieldSpec("unit", description="Unit copied from the lab report."),
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
