from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = "Data/external_bridge/canonical_oncology_schema.json"

CANONICAL_SCHEMA_VERSION = "oncotrack_canonical_oncology_v1"


FIELD_DEFINITIONS: list[dict[str, Any]] = [
    {"name": "source_dataset", "type": "string", "required": True, "meaning": "Public/synthetic source identifier."},
    {"name": "source_record_id", "type": "string", "required": True, "meaning": "Record identifier within the source dataset."},
    {"name": "patient_id", "type": "string", "required": True, "meaning": "Stable patient/case identifier after source namespacing."},
    {"name": "timepoint_index", "type": "integer", "required": False, "meaning": "Timepoint/cycle index when present."},
    {"name": "age", "type": "number", "required": False, "meaning": "Age at baseline/index where available."},
    {"name": "sex", "type": "string", "required": False, "allowed_values": ["female", "male", "unknown"]},
    {"name": "stage", "type": "string", "required": False, "meaning": "Reported stage; do not infer if absent."},
    {"name": "molecular_subtype", "type": "string", "required": False, "meaning": "Dataset subtype label."},
    {"name": "er_status", "type": "string", "required": False, "allowed_values": ["positive", "negative", "unknown"]},
    {"name": "pr_status", "type": "string", "required": False, "allowed_values": ["positive", "negative", "unknown"]},
    {"name": "her2_status", "type": "string", "required": False, "allowed_values": ["positive", "negative", "equivocal", "unknown"]},
    {"name": "ki67_percent", "type": "number", "required": False, "meaning": "Use only when explicitly reported."},
    {"name": "genetic_context_available", "type": "boolean", "required": False},
    {"name": "genetic_variant_classification", "type": "string", "required": False, "meaning": "Pathogenic/likely/VUS/benign if reported."},
    {"name": "treatment_phase", "type": "string", "required": False, "allowed_values": ["baseline", "neoadjuvant", "adjuvant", "metastatic", "survivorship", "unknown"]},
    {"name": "treatment_modalities", "type": "array[string]", "required": False, "meaning": "Structured modalities, not recommendations."},
    {"name": "treatment_combination_pattern", "type": "string", "required": False, "meaning": "Canonical combination bucket."},
    {"name": "regimen_text", "type": "string", "required": False, "meaning": "Source-reported regimen/arm text."},
    {"name": "cbc_available", "type": "boolean", "required": False},
    {"name": "symptoms_available", "type": "boolean", "required": False},
    {"name": "imaging_available", "type": "boolean", "required": False},
    {"name": "imaging_modality", "type": "string", "required": False, "allowed_values": ["MRI", "CT", "ultrasound", "mammogram", "PET/CT", "unknown"]},
    {"name": "imaging_features", "type": "object", "required": False, "meaning": "Dataset-specific numeric imaging features."},
    {"name": "tumor_marker_available", "type": "boolean", "required": False},
    {"name": "tumor_marker_context_only", "type": "boolean", "required": False},
    {"name": "outcome_label_name", "type": "string", "required": False, "meaning": "Source outcome label, e.g. pCR."},
    {"name": "outcome_label_value", "type": "number|string|boolean", "required": False},
    {"name": "source_urls", "type": "array[string]", "required": False},
    {"name": "claim_boundary", "type": "string", "required": True},
]

REQUIRED_FIELDS = {field["name"] for field in FIELD_DEFINITIONS if field.get("required")}


def build_canonical_oncology_schema(output_path: str | None = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    payload = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "fields": FIELD_DEFINITIONS,
        "minimum_dataset_roles": {
            "response_benchmark": ["patient_id", "imaging_available", "outcome_label_name", "outcome_label_value"],
            "treatment_sequence_benchmark": ["patient_id", "treatment_modalities", "treatment_combination_pattern"],
            "biomarker_mapping": ["patient_id", "er_status", "pr_status", "her2_status", "molecular_subtype"],
            "genetic_context_mapping": ["patient_id", "genetic_context_available", "genetic_variant_classification"],
        },
        "claim_boundary": (
            "Canonical schema alignment is engineering interoperability only. It does not establish "
            "clinical validation or make treatment recommendations."
        ),
    }
    if output_path:
        path = ROOT_DIR / output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_canonical_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = sorted(field for field in REQUIRED_FIELDS if row.get(field) in {None, ""})
        if missing:
            issues.append({"row_index": index, "missing_required": missing})
    return {
        "status": "passed" if not issues else "failed",
        "row_count": len(rows),
        "issue_count": len(issues),
        "issues": issues[:25],
    }
