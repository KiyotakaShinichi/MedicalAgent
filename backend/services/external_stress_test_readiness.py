from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_external_stress_test_readiness.json"

CLAIM_BOUNDARY = (
    "External stress-test readiness maps public/external-like schemas into the "
    "NLCare feature contract for engineering stress tests only. It is not "
    "clinical validation, external validation, or evidence of patient-level utility."
)

DATASET_TARGETS = [
    {
        "dataset_id": "tcga_brca",
        "label": "TCGA-BRCA",
        "source_artifact": "Data/evals/models/latest_tcga_metabric_canonical_mapping.json",
        "mapping_key": "tcga_brca_pan_can_atlas_2018",
        "expected_role": "genomic/subtype distribution stress only",
    },
    {
        "dataset_id": "metabric",
        "label": "METABRIC",
        "source_artifact": "Data/evals/models/latest_tcga_metabric_canonical_mapping.json",
        "mapping_key": "metabric",
        "expected_role": "biomarker/subtype/outcome schema stress only",
    },
    {
        "dataset_id": "breastdcedl_spy1",
        "label": "BreastDCEDL / I-SPY common-feature rows",
        "source_artifact": "Data/evals/models/latest_external_data_bridge_eval.json",
        "mapping_key": None,
        "expected_role": "imaging/pCR common-feature stress only",
    },
    {
        "dataset_id": "duke_tcia_mri",
        "label": "Duke MRI / TCIA schema candidate",
        "source_artifact": "Data/evals/models/latest_priority_dataset_bridge.json",
        "mapping_key": "duke_breast_mri",
        "expected_role": "future image/schema stress only",
    },
]

ONCOTRACK_REQUIRED_SIGNAL_FIELDS = {
    "demographics": ["age"],
    "response_context": ["er_status", "pr_status", "her2_status", "molecular_subtype"],
    "temporal_monitoring": ["treatment_cycle", "cbc_trend", "symptom_trajectory", "imaging_timeline"],
    "safety_context": ["treatment_modalities"],
}


def build_external_stress_test_readiness(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    datasets = [_dataset_report(target) for target in DATASET_TARGETS]
    total_rows = sum(int(item.get("row_count") or 0) for item in datasets)
    abstained_rows = sum(int(item.get("prediction_stress", {}).get("expected_abstained_rows") or 0) for item in datasets)
    payload = {
        "schema_version": "external_stress_test_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "datasets": datasets,
        "summary": {
            "dataset_count": len(datasets),
            "datasets_with_local_rows": sum(1 for item in datasets if int(item.get("row_count") or 0) > 0),
            "total_external_like_rows_seen": total_rows,
            "expected_abstained_rows": abstained_rows,
            "clinical_validation": False,
            "promotion_allowed": False,
        },
        "pipeline_use": {
            "allowed": [
                "schema mapping stress",
                "missing-field analysis",
                "abstention behavior audit",
                "external distribution sanity checks",
                "future data-access planning",
            ],
            "blocked": [
                "clinical validation claim",
                "patient-level treatment-response claim",
                "treatment recommendation",
                "genetic-risk interpretation",
                "tumor-marker interpretation",
                "model promotion without exact-label temporal validation",
            ],
        },
        "why_not_clinical_validation": [
            "External rows do not contain the full NLCare longitudinal CBC/symptom/medication/imaging timeline.",
            "Outcome labels such as pCR, survival, recurrence, or real-world response are not identical to the synthetic monitoring heads.",
            "Rows are not clinician-reviewed for the exact NLCare question and are not linked to patient-facing workflow safety outcomes.",
            "No IRB/ethics workflow, real patient deployment, or clinical sign-off exists in this project state.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(output_path, payload)
    return payload


def _dataset_report(target: dict[str, Any]) -> dict[str, Any]:
    source = _read_json(target["source_artifact"])
    dataset_payload = _extract_dataset_payload(source, target.get("mapping_key"))
    mapped_fields = _mapped_fields(dataset_payload, source, target["dataset_id"])
    row_count = _row_count(dataset_payload, source, target["dataset_id"])
    missing = _missing_oncotrack_fields(mapped_fields)
    expected_abstained = row_count if _should_abstain_for_full_hybrid(mapped_fields) else 0
    failure_cases = _failure_cases(mapped_fields, missing, row_count)
    return {
        "dataset_id": target["dataset_id"],
        "label": target["label"],
        "expected_role": target["expected_role"],
        "source_artifact": target["source_artifact"],
        "source_status": dataset_payload.get("status") or source.get("status") or "unknown",
        "row_count": row_count,
        "mapped_fields": sorted(mapped_fields),
        "missing_fields": missing,
        "prediction_stress": {
            "mode": "abstention_readiness_check",
            "expected_abstained_rows": expected_abstained,
            "expected_scored_rows": max(row_count - expected_abstained, 0),
            "reason": _abstention_reason(mapped_fields),
        },
        "failure_cases": failure_cases,
        "promotion_allowed": False,
        "claim_boundary": "This dataset entry is an external stress-test candidate only, not clinical validation.",
    }


def _extract_dataset_payload(source: dict[str, Any], mapping_key: str | None) -> dict[str, Any]:
    if mapping_key:
        datasets = source.get("datasets") if isinstance(source.get("datasets"), dict) else {}
        payload = datasets.get(mapping_key)
        return payload if isinstance(payload, dict) else {}
    return source


def _mapped_fields(dataset_payload: dict[str, Any], source: dict[str, Any], dataset_id: str) -> set[str]:
    fields: set[str] = set()
    available = dataset_payload.get("available_canonical_fields")
    if isinstance(available, list):
        fields.update(str(item) for item in available if item)
    field_map = dataset_payload.get("canonical_field_map")
    if isinstance(field_map, dict):
        for key, value in field_map.items():
            if value is not None and value != "" and value != []:
                fields.update(part.strip() for part in str(key).split("/") if part.strip())
    coverage = source.get("coverage") if isinstance(source.get("coverage"), dict) else {}
    if dataset_id == "breastdcedl_spy1":
        if coverage.get("modality_counts"):
            fields.add("imaging_timeline")
        if coverage.get("receptor_known_counts"):
            fields.update(["er_status", "pr_status", "her2_status"])
        if (coverage.get("roles_supported") or {}).get("external_pcr_imaging_response_benchmark"):
            fields.add("outcome_label_value")
        fields.update(["age", "molecular_subtype"])
    supported_roles = dataset_payload.get("supported_roles")
    if isinstance(supported_roles, dict):
        if supported_roles.get("treatment_history_bridge"):
            fields.add("treatment_modalities")
        if supported_roles.get("genomic_context_bridge"):
            fields.add("genetic_variant_classification")
        if supported_roles.get("mri_image_bridge"):
            fields.add("imaging_timeline")
    return fields


def _row_count(dataset_payload: dict[str, Any], source: dict[str, Any], dataset_id: str) -> int:
    if dataset_id == "breastdcedl_spy1":
        return int(source.get("row_count") or 0)
    return int(dataset_payload.get("row_count") or (dataset_payload.get("coverage") or {}).get("row_count") or 0)


def _missing_oncotrack_fields(mapped_fields: set[str]) -> list[str]:
    required = {field for values in ONCOTRACK_REQUIRED_SIGNAL_FIELDS.values() for field in values}
    return sorted(required - mapped_fields)


def _should_abstain_for_full_hybrid(mapped_fields: set[str]) -> bool:
    return not {"cbc_trend", "symptom_trajectory", "treatment_cycle"}.issubset(mapped_fields)


def _abstention_reason(mapped_fields: set[str]) -> str:
    missing = _missing_oncotrack_fields(mapped_fields)
    if missing:
        return "missing_oncotrack_longitudinal_modalities: " + ", ".join(missing[:6])
    return "full_hybrid_stress_candidate_has_required_field_names_but_still_needs exact-label temporal validation"


def _failure_cases(mapped_fields: set[str], missing: list[str], row_count: int) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if row_count == 0:
        failures.append({
            "type": "no_local_rows",
            "severity": "medium",
            "note": "Schema is mapped or planned, but no local rows are available for a pipeline stress run.",
        })
    if "imaging_timeline" not in mapped_fields:
        failures.append({
            "type": "missing_imaging_timeline",
            "severity": "medium",
            "note": "Response heads should abstain or remain context-only when imaging timeline is absent.",
        })
    if {"cbc_trend", "symptom_trajectory"} & set(missing):
        failures.append({
            "type": "missing_patient_monitoring_timeline",
            "severity": "high",
            "note": "External rows cannot exercise the full NLCare temporal monitoring pipeline.",
        })
    if "outcome_label_value" in mapped_fields:
        failures.append({
            "type": "target_semantics_mismatch",
            "severity": "high",
            "note": "External endpoints are useful stress labels but are not identical to NLCare synthetic monitoring heads.",
        })
    return failures


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    full_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    if not full_path.exists():
        return {}
    try:
        parsed = json.loads(full_path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["DEFAULT_OUTPUT_PATH", "build_external_stress_test_readiness"]
