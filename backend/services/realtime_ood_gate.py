"""Inference-time OOD / drift / data-quality gate.

This is an engineering guardrail before synthetic ML heads.  It detects
obvious data-quality problems (physiologically impossible labs, unknown units,
impossible dates, modality drift) and tells callers whether to allow, lower
confidence, or abstain/route for review.  It is not clinical safety proof.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backend.services.evidence_sufficiency import detect_modalities


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_realtime_ood_eval.json"
DEFAULT_BASELINE_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"

PHYSIOLOGICAL_BOUNDS = {
    "pre_wbc": (0.1, 200.0),
    "nadir_wbc": (0.1, 200.0),
    "recovery_wbc": (0.1, 200.0),
    "pre_anc": (0.0, 100.0),
    "nadir_anc": (0.0, 100.0),
    "pre_hemoglobin": (2.0, 25.0),
    "nadir_hemoglobin": (2.0, 25.0),
    "recovery_hemoglobin": (2.0, 25.0),
    "pre_platelets": (1.0, 2000.0),
    "nadir_platelets": (1.0, 2000.0),
    "recovery_platelets": (1.0, 2000.0),
    "mri_tumor_size_cm": (0.0, 50.0),
    "mri_percent_change_from_baseline": (-100.0, 500.0),
}

EXPECTED_UNITS = {
    "wbc": {"10^9/l", "x10^9/l", "k/ul", "k/uL".lower()},
    "anc": {"10^9/l", "x10^9/l", "k/ul", "k/uL".lower()},
    "hemoglobin": {"g/dl"},
    "platelets": {"10^9/l", "x10^9/l", "k/ul", "k/uL".lower()},
}


@dataclass
class OODGateResult:
    severity: str
    action: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence_modifier: float = 1.0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "action": self.action,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "confidence_modifier": self.confidence_modifier,
            "latency_ms": self.latency_ms,
            "claim_boundary": (
                "OOD/data-quality gate is an engineering guardrail for synthetic "
                "model inference, not clinical safety validation."
            ),
        }


def assess_realtime_ood(row: Mapping[str, Any]) -> OODGateResult:
    started = time.perf_counter()
    severe: list[str] = []
    moderate: list[str] = []
    mild: list[str] = []

    # Named `field_name` rather than `field`, which shadowed dataclasses.field
    # imported at module scope.
    for field_name, (lo, hi) in PHYSIOLOGICAL_BOUNDS.items():
        value = _as_float(row.get(field_name))
        if value is None:
            continue
        if value < lo or value > hi:
            severe.append(f"physiological_range_violation:{field_name}")

    for key, value in row.items():
        if not key.endswith("_unit") or value is None:
            continue
        label = key.replace("pre_", "").replace("nadir_", "").replace("recovery_", "").replace("_unit", "").lower()
        expected = EXPECTED_UNITS.get(label)
        if expected and str(value).strip().lower() not in expected:
            moderate.append(f"unknown_or_unexpected_unit:{key}")

    date_reason = _date_quality_reason(row)
    if date_reason:
        severe.append(date_reason)

    present, missing = detect_modalities(row)
    if "demographics" not in present:
        severe.append("missing_demographic_context")
    if "imaging" in missing:
        mild.append("missing_imaging_modality")
    if len(present) <= 2:
        moderate.append("low_modality_availability")

    suspicious = _suspicious_text(row)
    if suspicious:
        moderate.append(suspicious)

    if severe:
        severity = "severe"
        action = "abstain_or_clinician_review"
        modifier = 0.0
    elif moderate:
        severity = "moderate"
        action = "lower_confidence"
        modifier = 0.65
    elif mild:
        severity = "mild"
        action = "lower_confidence"
        modifier = 0.85
    else:
        severity = "none"
        action = "allow"
        modifier = 1.0

    return OODGateResult(
        severity=severity,
        action=action,
        reasons=severe + moderate,
        warnings=mild,
        confidence_modifier=modifier,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def run_realtime_ood_eval(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    baseline_csv: str | Path = DEFAULT_BASELINE_CSV,
) -> dict[str, Any]:
    normal_rows = _normal_rows(baseline_csv)
    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(normal_rows[:20]):
        result = assess_realtime_ood(row)
        cases.append({"case_id": f"normal_{idx}", "expected_ood": False, "result": result.to_dict()})

    template = normal_rows[0] if normal_rows else _default_normal_row()
    abnormal_cases = {
        "extreme_wbc": {**template, "pre_wbc": 9999},
        "unknown_unit": {**template, "pre_wbc_unit": "bananas"},
        "impossible_date": {**template, "treatment_date": "3026-01-01"},
        "missing_imaging": {**template, "mri_tumor_size_cm": None, "mri_percent_change_from_baseline": None},
        "modality_drift": {"age": template.get("age"), "cycle": template.get("cycle"), "stage": template.get("stage")},
    }
    for case_id, row in abnormal_cases.items():
        result = assess_realtime_ood(row)
        cases.append({"case_id": case_id, "expected_ood": True, "result": result.to_dict()})

    expected_ood = [case for case in cases if case["expected_ood"]]
    normal = [case for case in cases if not case["expected_ood"]]
    detected = [case for case in expected_ood if case["result"]["severity"] != "none"]
    false_ood = [case for case in normal if case["result"]["severity"] in {"moderate", "severe"}]
    severe_cases = [case for case in expected_ood if case["result"]["severity"] == "severe"]
    severe_abstained = [case for case in severe_cases if case["result"]["action"] == "abstain_or_clinician_review"]
    unit_cases = [case for case in cases if case["case_id"] == "unknown_unit"]
    latencies = [float(case["result"]["latency_ms"]) for case in cases]

    payload = {
        "schema_version": "realtime_ood_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if not false_ood and len(detected) >= 4 and len(severe_abstained) == len(severe_cases) else "acceptable",
        "summary": {
            "case_count": len(cases),
            "ood_detection_rate": round(len(detected) / max(len(expected_ood), 1), 4),
            "false_ood_rate": round(len(false_ood) / max(len(normal), 1), 4),
            "severe_ood_abstention_rate": round(len(severe_abstained) / max(len(severe_cases), 1), 4),
            "unit_error_detection_rate": round(sum(1 for case in unit_cases if case["result"]["severity"] != "none") / max(len(unit_cases), 1), 4),
            "physiological_range_violation_count": sum(
                1 for case in cases for reason in case["result"]["reasons"] if reason.startswith("physiological_range_violation")
            ),
            "modality_drift_alert_count": sum(
                1 for case in cases for reason in case["result"]["reasons"] if reason == "low_modality_availability"
            ),
            "p50_ood_check_latency_ms": _percentile(latencies, 50),
            "p95_ood_check_latency_ms": _percentile(latencies, 95),
        },
        "cases": cases,
        "claim_boundary": (
            "Real-time OOD gate is an engineering guardrail. It does not prove "
            "clinical safety, real-world robustness, or patient benefit."
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _normal_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists():
        return [_default_normal_row()]
    frame = pd.read_csv(path).head(25)
    return frame.to_dict(orient="records")


def _default_normal_row() -> dict[str, Any]:
    return {
        "age": 52,
        "cycle": 3,
        "stage": "II",
        "molecular_subtype": "HR+/HER2-",
        "regimen": "synthetic regimen",
        "pre_wbc": 5.2,
        "pre_anc": 2.4,
        "pre_hemoglobin": 12.0,
        "pre_platelets": 220,
        "nadir_wbc": 2.1,
        "nadir_anc": 1.1,
        "nadir_hemoglobin": 10.8,
        "nadir_platelets": 160,
        "recovery_wbc": 4.8,
        "recovery_hemoglobin": 11.7,
        "recovery_platelets": 210,
        "mri_tumor_size_cm": 2.4,
        "mri_percent_change_from_baseline": -22.0,
        "max_symptom_severity": 3,
        "symptom_count": 2,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _date_quality_reason(row: Mapping[str, Any]) -> str | None:
    for field_name in ("treatment_date", "imaging_date", "report_date"):
        raw = row.get(field_name)
        if raw in (None, ""):
            continue
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return f"invalid_date:{field_name}"
        if parsed.year > datetime.now(timezone.utc).year + 5 or parsed.year < 1900:
            return f"impossible_date:{field_name}"
    return None


def _suspicious_text(row: Mapping[str, Any]) -> str | None:
    joined = " ".join(str(value).lower() for value in row.values() if isinstance(value, str))
    if any(token in joined for token in ("ignore previous", "system prompt", "jailbreak", "developer message")):
        return "suspicious_prompt_injection_pattern"
    return None


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((percentile / 100) * (len(values) - 1))))
    return round(values[index], 3)


__all__ = ["OODGateResult", "assess_realtime_ood", "run_realtime_ood_eval"]
