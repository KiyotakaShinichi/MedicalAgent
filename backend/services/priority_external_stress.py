from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_BRIDGE_PATH = "Data/evals/models/latest_priority_dataset_bridge.json"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_priority_external_stress.json"

COMMON_FEATURES = [
    "age",
    "er_status",
    "pr_status",
    "her2_status",
    "molecular_subtype",
    "treatment_modalities",
    "treatment_combination_pattern",
    "outcome_label_name",
    "outcome_label_value",
]

CLAIM_BOUNDARY = (
    "Priority external stress checks schema overlap and endpoint compatibility for GENIE BPC BRCA "
    "and Duke Breast MRI bridge rows. It is not clinical validation and cannot promote any model "
    "without exact-label temporal validation and clinician-reviewed endpoints."
)


def build_priority_external_stress(
    *,
    bridge_path: str = DEFAULT_BRIDGE_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    bridge = _read_json(_resolve(bridge_path))
    datasets = bridge.get("datasets") or {}
    stress_rows = {
        dataset_id: _load_dataset_rows(dataset)
        for dataset_id, dataset in datasets.items()
    }
    dataset_reports = {
        dataset_id: _dataset_stress_report(dataset_id, rows)
        for dataset_id, rows in stress_rows.items()
    }
    mapped_dataset_count = sum(1 for rows in stress_rows.values() if rows)
    endpoint_compatibility = _endpoint_compatibility(dataset_reports)
    promotion_allowed = False
    status = "strong" if mapped_dataset_count >= 2 else "ready_when_mapped"
    payload: dict[str, Any] = {
        "schema_version": "priority_external_stress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "bridge_path": bridge_path,
        "common_features": COMMON_FEATURES,
        "mapped_dataset_count": mapped_dataset_count,
        "datasets": dataset_reports,
        "endpoint_compatibility": endpoint_compatibility,
        "promotion_decision": {
            "promotion_allowed": promotion_allowed,
            "required_before_promotion": [
                "same target semantics across datasets",
                "temporal patient journeys with prior-only features",
                "clinician-reviewed labels for the exact monitoring question",
                "calibration and subgroup reliability on external rows",
            ],
            "reason": "Mapped public rows are schema/external-stress evidence only; they are not NLCare longitudinal validation rows.",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    return payload


def _dataset_stress_report(dataset_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_coverage = {feature: _coverage_rate(rows, feature) for feature in COMMON_FEATURES}
    outcomes = Counter(str(row.get("outcome_label_name") or "missing") for row in rows)
    treatment_patterns = Counter(str(row.get("treatment_combination_pattern") or "missing") for row in rows)
    return {
        "status": "mapped" if rows else "not_mapped",
        "row_count": len(rows),
        "feature_coverage": feature_coverage,
        "outcome_label_counts": dict(outcomes),
        "treatment_combination_counts": dict(treatment_patterns),
        "roles": {
            "schema_stress_ready": bool(rows),
            "common_feature_ab_ready": bool(rows and feature_coverage.get("age", 0) > 0),
            "exact_label_temporal_validation_ready": False,
        },
    }


def _endpoint_compatibility(dataset_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = {
        dataset_id: sorted(
            label for label in report["outcome_label_counts"]
            if label not in {"missing", ""}
        )
        for dataset_id, report in dataset_reports.items()
    }
    non_empty = [tuple(value) for value in labels.values() if value]
    same_endpoint = bool(non_empty) and len(set(non_empty)) == 1
    return {
        "same_endpoint_labels": same_endpoint,
        "observed_labels_by_dataset": labels,
        "exact_oncotrack_label_match": False,
        "reason": (
            "GENIE/Duke endpoints such as real-world response, pCR, recurrence, PFS, or OS are useful "
            "external context, but they are not the same as NLCare's synthetic response classification, "
            "response-score regression, or toxicity-review labels."
        ),
    }


def _load_dataset_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    path = dataset.get("canonical_csv_path")
    if not path:
        return []
    file_path = _resolve(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return rows


def _coverage_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    present = 0
    for row in rows:
        value = row.get(key)
        if value not in {None, "", "unknown", "[]", "{}"}:
            present += 1
    return round(present / len(rows), 4)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
