from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import (
    ROOT_DIR,
    build_canonical_oncology_schema,
    validate_canonical_rows,
)


DEFAULT_FEATURES_CSV = "Data/breastdcedl_spy1_features.csv"
DEFAULT_METRICS_JSON = "Data/breastdcedl_spy1_baseline_metrics.json"
DEFAULT_PREDICTIONS_CSV = "Data/breastdcedl_spy1_model_predictions.csv"
DEFAULT_CANONICAL_CSV = "Data/external_bridge/canonical_breastdcedl_spy1.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_external_data_bridge_eval.json"
DEFAULT_FAILURE_GALLERY_PATH = "Data/evals/models/latest_external_failure_case_gallery.json"

BREASTDCEDL_SOURCE_URLS = [
    "https://zenodo.org/records/17578255",
    "https://www.nature.com/articles/s41597-025-05359-4",
]

CLAIM_BOUNDARY = (
    "External bridge artifacts align public benchmark fields to the NLCare schema for engineering "
    "readiness only. BreastDCEDL/I-SPY is an imaging/pCR benchmark and is not a full NLCare "
    "longitudinal treatment, CBC, symptom, medication, tumor-marker, or clinical-validation dataset."
)


def build_external_data_bridge(
    *,
    features_csv: str = DEFAULT_FEATURES_CSV,
    metrics_json: str = DEFAULT_METRICS_JSON,
    predictions_csv: str = DEFAULT_PREDICTIONS_CSV,
    canonical_csv: str = DEFAULT_CANONICAL_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    failure_gallery_path: str = DEFAULT_FAILURE_GALLERY_PATH,
) -> dict[str, Any]:
    build_canonical_oncology_schema()

    features_path = _resolve(features_csv)
    rows = _read_csv(features_path)
    canonical_rows = [_breastdcedl_to_canonical(row) for row in rows]
    validation = validate_canonical_rows(canonical_rows)
    _write_csv(_resolve(canonical_csv), canonical_rows)

    metrics = _read_json(_resolve(metrics_json))
    predictions = _read_csv(_resolve(predictions_csv)) if _resolve(predictions_csv).exists() else []
    failure_gallery = _build_failure_gallery(predictions)
    _write_json(_resolve(failure_gallery_path), failure_gallery)

    report: dict[str, Any] = {
        "schema_version": "external_data_bridge_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if canonical_rows and validation["status"] == "passed" else "needs_attention",
        "source_dataset": "breastdcedl_spy1",
        "canonical_schema_path": "Data/external_bridge/canonical_oncology_schema.json",
        "canonical_csv_path": canonical_csv,
        "row_count": len(canonical_rows),
        "validation": validation,
        "coverage": _coverage(canonical_rows),
        "external_model_snapshot": {
            "model_type": metrics.get("model_type"),
            "rows": metrics.get("rows"),
            "positive_pcr": metrics.get("positive_pcr"),
            "negative_pcr": metrics.get("negative_pcr"),
            "best_model_by_roc_auc": metrics.get("best_model_by_roc_auc"),
            "models": metrics.get("models", {}),
            "warning": metrics.get("warning", "Exploratory PoC only. Not clinically validated."),
        },
        "failure_gallery_path": failure_gallery_path,
        "failure_gallery_summary": failure_gallery["summary"],
        "candidate_comparison_matrix": _candidate_comparison_matrix(metrics),
        "next_dataset_targets": [
            {
                "dataset": "AACR GENIE BPC Breast Cancer",
                "role": "future real-world treatment/genomic/outcome benchmark after access workflow",
                "reason_not_current_training": "not locally mapped into NLCare longitudinal features yet",
            },
            {
                "dataset": "SEER / SEER-Medicare",
                "role": "future treatment-combination distribution and claims-derived therapy context",
                "reason_not_current_training": "requires agreement/application and does not provide the current CBC/symptom timeline",
            },
            {
                "dataset": "Duke Breast Cancer MRI / TCIA",
                "role": "future imaging externalization candidate",
                "reason_not_current_training": "requires image normalization and manual label/schema mapping",
            },
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), report)
    return report


def _breastdcedl_to_canonical(row: dict[str, str]) -> dict[str, Any]:
    subtype = row.get("molecular_subtype") or "unknown"
    er_status, pr_status, her2_status = _derive_receptor_status(subtype)
    imaging_features = {
        key: _to_float(row.get(key))
        for key in (
            "baseline_longest_diameter_mm",
            "tumor_voxel_count",
            "acq0_mask_mean",
            "acq1_mask_mean",
            "acq2_mask_mean",
            "early_enhancement_mean",
            "delayed_enhancement_mean",
            "washout_mean",
            "early_enhancement_p90",
            "delayed_enhancement_p90",
            "washout_p10",
        )
        if row.get(key) not in {None, ""}
    }
    source_record_id = str(row.get("patient_id") or "")
    return {
        "source_dataset": "breastdcedl_spy1",
        "source_record_id": source_record_id,
        "patient_id": f"BREASTDCEDL:{source_record_id}",
        "timepoint_index": 0,
        "age": _to_float(row.get("age")),
        "sex": "unknown",
        "stage": "unknown",
        "molecular_subtype": subtype,
        "er_status": er_status,
        "pr_status": pr_status,
        "her2_status": her2_status,
        "ki67_percent": "",
        "genetic_context_available": False,
        "genetic_variant_classification": "not_reported",
        "treatment_phase": "neoadjuvant",
        "treatment_modalities": ["chemotherapy_context", "MRI"],
        "treatment_combination_pattern": "neoadjuvant_systemic_imaging_response_context",
        "regimen_text": "I-SPY neoadjuvant trial context; regimen details not mapped in this local bridge",
        "cbc_available": False,
        "symptoms_available": False,
        "imaging_available": True,
        "imaging_modality": "MRI",
        "imaging_features": imaging_features,
        "tumor_marker_available": False,
        "tumor_marker_context_only": True,
        "outcome_label_name": "pCR",
        "outcome_label_value": _to_int_like(row.get("pcr_label")),
        "source_urls": BREASTDCEDL_SOURCE_URLS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _derive_receptor_status(subtype: str) -> tuple[str, str, str]:
    normalized = subtype.lower().replace("+", "pos").replace("-", "neg")
    if "tripleneg" in normalized or "triple" in normalized:
        return "negative", "negative", "negative"
    er = "unknown"
    pr = "unknown"
    her2 = "unknown"
    if "hrpos" in normalized or "hr/erpos" in normalized or "erpos" in normalized:
        er = "positive"
        pr = "unknown"
    if "hrneg" in normalized or "erneg" in normalized:
        er = "negative"
        pr = "unknown"
    if "her2pos" in normalized:
        her2 = "positive"
    elif "her2neg" in normalized:
        her2 = "negative"
    return er, pr, her2


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modality_counts = Counter()
    subtype_counts = Counter()
    receptor_known = Counter()
    outcome_count = 0
    for row in rows:
        subtype_counts[str(row.get("molecular_subtype") or "unknown")] += 1
        if row.get("imaging_available"):
            modality_counts["imaging"] += 1
        if row.get("cbc_available"):
            modality_counts["cbc"] += 1
        if row.get("symptoms_available"):
            modality_counts["symptoms"] += 1
        if row.get("tumor_marker_available"):
            modality_counts["tumor_marker"] += 1
        for key in ("er_status", "pr_status", "her2_status"):
            if row.get(key) not in {None, "", "unknown"}:
                receptor_known[key] += 1
        if row.get("outcome_label_name") and row.get("outcome_label_value") not in {None, ""}:
            outcome_count += 1
    return {
        "modality_counts": dict(modality_counts),
        "molecular_subtype_counts": dict(subtype_counts),
        "receptor_known_counts": dict(receptor_known),
        "outcome_label_count": outcome_count,
        "roles_supported": {
            "external_pcr_imaging_response_benchmark": bool(rows and modality_counts["imaging"] == len(rows) and outcome_count == len(rows)),
            "full_oncotrack_timeline_training": False,
            "treatment_combination_training": False,
            "tumor_marker_response_training": False,
        },
    }


def _build_failure_gallery(predictions: list[dict[str, str]], *, limit: int = 10) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    false_positive_count = 0
    false_negative_count = 0
    for row in predictions:
        actual = _to_int_like(row.get("pcr_label"))
        predicted = _to_int_like(row.get("best_model_predicted_label"))
        probability = _to_float(row.get("best_model_pcr_probability"))
        if actual is None or predicted is None or probability is None or actual == predicted:
            continue
        error_type = "false_positive" if predicted == 1 else "false_negative"
        if error_type == "false_positive":
            false_positive_count += 1
        else:
            false_negative_count += 1
        cases.append({
            "patient_id": row.get("patient_id"),
            "molecular_subtype": row.get("molecular_subtype"),
            "actual_pcr_label": actual,
            "predicted_label": predicted,
            "pcr_probability": round(float(probability), 6),
            "confidence_distance_from_threshold": round(abs(float(probability) - 0.5), 6),
            "error_type": error_type,
            "review_note": "External benchmark failure case for engineering review; not a clinical adjudication.",
        })
    cases.sort(key=lambda item: item["confidence_distance_from_threshold"], reverse=True)
    return {
        "schema_version": "external_failure_case_gallery_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "source_dataset": "breastdcedl_spy1",
        "summary": {
            "case_count": len(cases),
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
            "displayed_case_count": min(len(cases), limit),
        },
        "cases": cases[:limit],
        "claim_boundary": (
            "Failure cases expose external benchmark errors for debugging and portfolio transparency. "
            "They are not patient-level clinical errors or treatment guidance."
        ),
    }


def _candidate_comparison_matrix(external_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    synthetic_metrics = _read_json(ROOT_DIR / "Data/complete_synthetic_training/complete_synthetic_model_metrics.json")
    dl_metrics = _read_json(ROOT_DIR / "Data/evals/models/latest_deep_learning_candidate_benchmark.json")
    best_external_model = external_metrics.get("best_model_by_roc_auc")
    external_model_metrics = (external_metrics.get("models") or {}).get(best_external_model or "", {})
    return [
        {
            "candidate": "synthetic_classical_champion",
            "dataset": "complete_synthetic_breast_journeys",
            "primary_metric": _synthetic_champion_roc_auc(synthetic_metrics),
            "metric_name": "synthetic_patient_level_roc_auc",
            "role": "current simulator benchmark",
            "promotion_decision": "monitor_only_synthetic",
        },
        {
            "candidate": "synthetic_deep_learning_candidate",
            "dataset": "complete_synthetic_breast_journeys",
            "primary_metric": _dig(dl_metrics, ["best_model", "classification_auroc"]),
            "metric_name": "synthetic_holdout_roc_auc",
            "role": "experimental A/B candidate",
            "promotion_decision": "candidate_only_until_external_validation",
        },
        {
            "candidate": f"breastdcedl_{best_external_model or 'baseline'}",
            "dataset": "breastdcedl_spy1",
            "primary_metric": external_model_metrics.get("roc_auc"),
            "metric_name": "external_pcr_cv_roc_auc",
            "role": "public external imaging/pCR sanity check",
            "promotion_decision": "not_promoted_external_context_only",
        },
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT_DIR / resolved


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_like(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(round(number))


def _dig(payload: dict[str, Any], path: list[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_number(payload: dict[str, Any], *paths: list[str]) -> float | None:
    for path in paths:
        value = _dig(payload, path)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _synthetic_champion_roc_auc(metrics: dict[str, Any]) -> float | None:
    best_name = metrics.get("best_model_by_patient_level_roc_auc") or metrics.get("best_model")
    if isinstance(best_name, str):
        best_metrics = _dig(metrics, ["models", best_name]) or {}
        value = best_metrics.get("patient_level_roc_auc") or best_metrics.get("roc_auc")
        if isinstance(value, (int, float)):
            return float(value)
    model_metrics = metrics.get("models")
    if isinstance(model_metrics, dict):
        candidates = [
            values.get("patient_level_roc_auc") or values.get("roc_auc")
            for values in model_metrics.values()
            if isinstance(values, dict)
        ]
        numeric = [float(value) for value in candidates if isinstance(value, (int, float))]
        if numeric:
            return max(numeric)
    return _first_number(metrics, ["classification", "test_roc_auc"], ["roc_auc"], ["test_auc"])
