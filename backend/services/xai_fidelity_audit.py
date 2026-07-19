"""Mechanical fidelity and presentation-safety audit for synthetic XAI exports."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("Data/complete_synthetic_training/synthetic_xai_explanations.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_xai_fidelity_audit.json")
NEAR_OUTCOME_PROXIES = {"mri_percent_change_from_baseline", "response_score_percent", "latent_response_strength"}
ADDITIVITY_TOLERANCE_LOG_ODDS = 1e-5


def build_xai_fidelity_audit(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    source = json.loads(Path(input_path).read_text(encoding="utf-8"))
    patients = source.get("patients") or {}
    rows = list(patients.values()) if isinstance(patients, dict) else list(patients)
    finite = 0
    signed = 0
    total_contributions = 0
    duplicate_patients = 0
    multi_one_hot_patients = 0
    proxy_patients = 0
    prediction_present = 0
    base_value_present = 0
    additivity_fields_present = 0
    additivity_within_tolerance = 0
    residuals: list[float] = []
    examples = []
    for row in rows:
        positive = list(row.get("positive_contributions") or [])
        negative = list(row.get("negative_contributions") or [])
        contributions = list(row.get("all_contributions") or []) or positive + negative
        total_contributions += len(contributions)
        finite += sum(_finite(item.get("contribution")) for item in contributions)
        signed += sum(
            (_number(item.get("contribution")) >= 0 and item.get("direction") == "toward_success")
            or (_number(item.get("contribution")) <= 0 and item.get("direction") == "toward_non_success")
            for item in contributions
        )
        features = [str(item.get("feature") or "") for item in contributions]
        duplicates = sorted({feature for feature in features if feature and features.count(feature) > 1})
        duplicate_patients += int(bool(duplicates))
        stage_features = [feature for feature in features if feature.startswith("stage_")]
        multi_one_hot_patients += int(len(stage_features) > 1)
        proxies = sorted(set(features) & NEAR_OUTCOME_PROXIES)
        proxy_patients += int(bool(proxies))
        model_output = row.get("model_output") if isinstance(row.get("model_output"), dict) else {}
        prediction_value = model_output.get("mean_prediction_log_odds")
        if prediction_value is None and isinstance(row.get("prediction"), (int, float)):
            prediction_value = row.get("prediction")
        base_value = row.get("base_value", row.get("expected_value"))
        prediction_present += int(prediction_value is not None or row.get("prediction") is not None)
        base_value_present += int(base_value is not None)
        if prediction_value is not None and base_value is not None and contributions:
            reconstructed = _number(base_value) + sum(_number(item.get("contribution")) for item in contributions)
            residual = abs(reconstructed - _number(prediction_value))
            if math.isfinite(residual):
                additivity_fields_present += 1
                residuals.append(residual)
                additivity_within_tolerance += int(residual <= ADDITIVITY_TOLERANCE_LOG_ODDS)
        if (duplicates or len(stage_features) > 1 or proxies) and len(examples) < 12:
            examples.append({
                "patient_id": row.get("patient_id"), "duplicate_features": duplicates,
                "multiple_mutually_exclusive_one_hot_features": stage_features,
                "near_outcome_proxy_features": proxies,
            })
    n = len(rows)
    additivity_verifiable = n > 0 and additivity_fields_present == n
    additivity_pass_rate = _rate(additivity_within_tolerance, n)
    payload = {
        "schema_version": "xai_fidelity_audit_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "acceptable"
            if additivity_verifiable and additivity_pass_rate == 1.0 and duplicate_patients == 0
            else "needs_attention"
        ),
        "method": source.get("method"), "shap_available": bool(source.get("shap_available")),
        "patient_explanation_n": n, "contribution_n": total_contributions,
        "finite_contribution_rate": _rate(finite, total_contributions),
        "direction_sign_consistency_rate": _rate(signed, total_contributions),
        "duplicate_feature_patient_rate": _rate(duplicate_patients, n),
        "multiple_one_hot_feature_patient_rate": _rate(multi_one_hot_patients, n),
        "near_outcome_proxy_patient_rate": _rate(proxy_patients, n),
        "prediction_present_rate": _rate(prediction_present, n),
        "base_value_present_rate": _rate(base_value_present, n),
        "additivity_verifiable": additivity_verifiable,
        "additivity_output_space": "log_odds",
        "additivity_tolerance_log_odds": ADDITIVITY_TOLERANCE_LOG_ODDS,
        "additivity_pass_rate": additivity_pass_rate,
        "max_absolute_additivity_residual_log_odds": round(max(residuals), 12) if residuals else None,
        "mean_absolute_additivity_residual_log_odds": (
            round(sum(residuals) / len(residuals), 12) if residuals else None
        ),
        "rank_stability_evaluated": False,
        "causal_interpretation_allowed": False,
        "presentation_risks": [
            "Centered one-hot SHAP values can display several mutually exclusive categories and confuse users.",
            "Near-outcome proxy features can make explanations look more intelligent than the synthetic target construction warrants.",
            "Additivity verifies arithmetic in model-output space; it does not establish clinical validity or causal meaning.",
        ],
        "representative_risks": examples,
        "required_next_actions": [
            "collapse mutually exclusive one-hot groups for patient-facing display",
            "separate near-outcome proxy features from ordinary context features",
            "measure explanation rank stability across resamples before promotion",
        ],
        "synthetic_only": True,
        "clinical_validation": False,
        "claim_boundary": "This audits explanation mechanics and wording risk only; it is not clinical explainability, causality, or validation.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(value: Any) -> bool:
    return math.isfinite(_number(value))


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


__all__ = ["build_xai_fidelity_audit"]
