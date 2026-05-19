from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.external_data_bridge import DEFAULT_PREDICTIONS_CSV
from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_external_failure_case_analysis.json"

CLAIM_BOUNDARY = (
    "External failure-case analysis is for engineering review of public benchmark behavior. "
    "It is not patient-level clinical adjudication and does not establish clinical safety or utility."
)


def build_external_failure_case_analysis(
    *,
    predictions_csv: str = DEFAULT_PREDICTIONS_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = _read_csv(_resolve(predictions_csv))
    cases = [_case(row) for row in rows]
    failures = [case for case in cases if case["is_error"]]
    subtype_summary = _group_summary(failures, "molecular_subtype")
    confidence_summary = _group_summary(failures, "confidence_bucket")
    error_summary = _group_summary(failures, "error_type")
    high_confidence_failures = [
        case for case in sorted(failures, key=lambda item: item["confidence_distance_from_threshold"], reverse=True)
        if case["confidence_bucket"] in {"high_confidence", "very_high_confidence"}
    ][:25]
    payload = {
        "schema_version": "external_failure_case_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if rows else "needs_attention",
        "source_dataset": "breastdcedl_spy1",
        "summary": {
            "row_count": len(rows),
            "failure_count": len(failures),
            "failure_rate": round(len(failures) / max(len(rows), 1), 4),
            "false_positive_count": sum(1 for case in failures if case["error_type"] == "false_positive"),
            "false_negative_count": sum(1 for case in failures if case["error_type"] == "false_negative"),
            "high_confidence_failure_count": len(high_confidence_failures),
        },
        "by_molecular_subtype": subtype_summary,
        "by_confidence_bucket": confidence_summary,
        "by_error_type": error_summary,
        "high_confidence_failures": high_confidence_failures,
        "review_questions": [
            "Are false positives concentrated in a subtype?",
            "Are false negatives low-confidence or high-confidence misses?",
            "Would stricter abstention reduce high-confidence errors?",
            "Which failures would need clinician or imaging-review adjudication before any claim?",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _case(row: dict[str, str]) -> dict[str, Any]:
    actual = _to_int(row.get("pcr_label"))
    predicted = _to_int(row.get("best_model_predicted_label"))
    probability = _to_float(row.get("best_model_pcr_probability")) or 0.0
    is_error = actual is not None and predicted is not None and actual != predicted
    distance = abs(probability - 0.5)
    return {
        "patient_id": row.get("patient_id"),
        "molecular_subtype": row.get("molecular_subtype") or "unknown",
        "actual_pcr_label": actual,
        "predicted_label": predicted,
        "pcr_probability": round(probability, 6),
        "confidence_distance_from_threshold": round(distance, 6),
        "confidence_bucket": _confidence_bucket(distance),
        "is_error": is_error,
        "error_type": "false_positive" if is_error and predicted == 1 else "false_negative" if is_error else "correct",
    }


def _confidence_bucket(distance: float) -> str:
    if distance >= 0.35:
        return "very_high_confidence"
    if distance >= 0.20:
        return "high_confidence"
    if distance >= 0.10:
        return "moderate_confidence"
    return "near_threshold"


def _group_summary(cases: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case.get(key) or "unknown")].append(case)
    rows = []
    for value, group in sorted(groups.items()):
        errors = Counter(case["error_type"] for case in group)
        rows.append({
            key: value,
            "failure_count": len(group),
            "false_positive_count": errors.get("false_positive", 0),
            "false_negative_count": errors.get("false_negative", 0),
            "mean_confidence_distance": round(
                sum(case["confidence_distance_from_threshold"] for case in group) / max(len(group), 1),
                4,
            ),
        })
    rows.sort(key=lambda item: item["failure_count"], reverse=True)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(round(number))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
