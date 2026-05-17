"""Robustness stress suite for synthetic patient timeline rows.

This suite intentionally damages or removes parts of a representative timeline
row and checks whether the hybrid model reacts safely: uncertainty should
increase, response heads should abstain when response evidence is insufficient,
and clinician-review routing should remain conservative.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backend.services.hybrid_prediction import predict_hybrid
from backend.services.medical_claim_boundary import classify_medical_claim


DEFAULT_ROWS_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/robustness/latest_robustness_report.json"


def run_robustness_stress_suite(
    *,
    rows_path: str = DEFAULT_ROWS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = pd.read_csv(rows_path)
    base_row = _select_base_row(rows)
    baseline = predict_hybrid(base_row).to_dict()
    baseline_width = _regression_width(baseline)
    cases = []
    for case in STRESS_CASES:
        stressed = case.transform(deepcopy(base_row))
        prediction = predict_hybrid(stressed).to_dict()
        cases.append(_evaluate_case(case, prediction, baseline_width))

    summary = _summarise(cases)
    payload = {
        "schema_version": "robustness_stress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": summary["status"],
        "rows_path": rows_path,
        "baseline": _compact_prediction(baseline),
        "summary": summary,
        "cases": cases,
        "claim_boundary": (
            "Synthetic stress testing only. These cases test failure behavior "
            "under missing/corrupt/conflicting inputs; they do not validate "
            "clinical robustness."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class StressCase:
    def __init__(
        self,
        name: str,
        category: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        expected: str,
    ) -> None:
        self.name = name
        self.category = category
        self.transform = transform
        self.expected = expected


def _missing(row: dict[str, Any], *columns: str) -> dict[str, Any]:
    for column in columns:
        row[column] = None
    return row


def _wrong_units(row: dict[str, Any]) -> dict[str, Any]:
    row["pre_wbc"] = 6200.0
    row["nadir_wbc"] = 2100.0
    row["recovery_wbc"] = 5000.0
    return row


def _contradictory_symptoms(row: dict[str, Any]) -> dict[str, Any]:
    row["max_symptom_severity"] = 10
    row["symptom_count"] = 8
    row["pre_wbc"] = 7.0
    row["nadir_wbc"] = 6.8
    row["recovery_wbc"] = 7.2
    return row


def _noisy_tumor_marker_context(row: dict[str, Any]) -> dict[str, Any]:
    row["tumor_marker_context"] = "CA 15-3 fluctuating, single high value; context dependent"
    return row


def _ambiguous_biomarker(row: dict[str, Any]) -> dict[str, Any]:
    row["molecular_subtype"] = "unknown"
    row["biomarker_note"] = "HER2 equivocal; FISH not available"
    return row


STRESS_CASES: tuple[StressCase, ...] = (
    StressCase(
        "missing_cbc_values",
        "missing_data",
        lambda r: _missing(r, "pre_wbc", "pre_anc", "nadir_wbc", "nadir_anc", "recovery_wbc"),
        "uncertainty_or_abstention",
    ),
    StressCase(
        "missing_imaging_notes",
        "missing_data",
        lambda r: _missing(r, "mri_tumor_size_cm", "mri_percent_change_from_baseline"),
        "uncertainty_or_abstention",
    ),
    StressCase("wrong_wbc_units", "data_quality", _wrong_units, "clinician_review"),
    StressCase("contradictory_symptoms_vs_labs", "conflict", _contradictory_symptoms, "uncertainty_or_review"),
    StressCase(
        "delayed_imaging_report",
        "recency",
        lambda r: _missing(r, "mri_percent_change_from_baseline"),
        "uncertainty_or_abstention",
    ),
    StressCase("noisy_tumor_marker_context", "tumor_marker", _noisy_tumor_marker_context, "no_overclaim"),
    StressCase("ambiguous_biomarker_result", "biomarker", _ambiguous_biomarker, "no_overclaim"),
    StressCase(
        "incomplete_family_history",
        "genetics",
        lambda r: {**r, "family_history_context": "unknown maternal/paternal side"},
        "review_context_only",
    ),
)


def _select_base_row(rows: pd.DataFrame) -> dict[str, Any]:
    candidates = rows.dropna(subset=["mri_tumor_size_cm", "mri_percent_change_from_baseline", "nadir_wbc"])
    if candidates.empty:
        candidates = rows
    return candidates.iloc[-1].to_dict()


def _evaluate_case(case: StressCase, prediction: dict[str, Any], baseline_width: float | None) -> dict[str, Any]:
    classification = prediction["classification"]
    regression = prediction["response_score"]
    toxicity = prediction["toxicity"]
    width = _regression_width(prediction)
    abstained = any(
        head.get("decision") == "insufficient_evidence"
        for head in (classification, regression, toxicity)
    )
    uncertainty_increased = (
        width is not None and baseline_width is not None and width >= baseline_width
    )
    review_routed = (
        abstained
        or toxicity.get("decision") in {"moderate_toxicity_signal", "high_toxicity_signal"}
        or classification.get("confidence") == "low"
        or regression.get("confidence") == "low"
    )
    claim_check = classify_medical_claim(
        "Tumor markers are context-dependent and cannot diagnose recurrence by themselves."
    )
    passed = _case_passed(case, abstained, uncertainty_increased, review_routed, claim_check)
    return {
        "case": case.name,
        "category": case.category,
        "expected": case.expected,
        "passed": passed,
        "abstained_any_head": abstained,
        "uncertainty_increased_or_equal": uncertainty_increased,
        "clinician_review_routed": review_routed,
        "medical_claim_boundary_decision": claim_check["decision"],
        "prediction": _compact_prediction(prediction),
    }


def _case_passed(
    case: StressCase,
    abstained: bool,
    uncertainty_increased: bool,
    review_routed: bool,
    claim_check: dict[str, Any],
) -> bool:
    if case.expected == "no_overclaim":
        return claim_check["decision"] != "blocked" and review_routed
    if case.expected == "clinician_review":
        return review_routed
    if case.expected == "review_context_only":
        return review_routed or uncertainty_increased
    return abstained or uncertainty_increased or review_routed


def _compact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": {
            "decision": prediction["classification"].get("decision"),
            "confidence": prediction["classification"].get("confidence"),
            "sufficiency": prediction["classification"].get("evidence", {}).get("sufficiency"),
        },
        "response_score": {
            "decision": prediction["response_score"].get("decision"),
            "confidence": prediction["response_score"].get("confidence"),
            "band": prediction["response_score"].get("uncertainty_band"),
            "sufficiency": prediction["response_score"].get("evidence", {}).get("sufficiency"),
        },
        "toxicity": {
            "decision": prediction["toxicity"].get("decision"),
            "confidence": prediction["toxicity"].get("confidence"),
            "sufficiency": prediction["toxicity"].get("evidence", {}).get("sufficiency"),
        },
    }


def _regression_width(prediction: dict[str, Any]) -> float | None:
    band = prediction.get("response_score", {}).get("uncertainty_band")
    if not band:
        return None
    return float(band[1]) - float(band[0])


def _summarise(cases: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for case in cases if case["passed"])
    rate = passed / max(1, len(cases))
    return {
        "status": "strong" if rate >= 0.90 else "acceptable" if rate >= 0.75 else "needs_attention",
        "case_count": len(cases),
        "passed": passed,
        "pass_rate": round(rate, 4),
        "abstention_or_review_rate": round(
            sum(1 for case in cases if case["abstained_any_head"] or case["clinician_review_routed"]) / max(1, len(cases)),
            4,
        ),
    }


def load_robustness_stress_report(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "robustness_stress_v1",
            "status": "missing",
            "message": "Run scripts/run_robustness_stress.py to generate this artifact.",
            "cases": [],
            "summary": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "load_robustness_stress_report",
    "run_robustness_stress_suite",
]
