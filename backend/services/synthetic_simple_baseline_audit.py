"""Simple-baseline audit for the synthetic-only monitoring models."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ROWS = Path("Data/evals/models/latest_row_level_prediction_export.csv")
DEFAULT_PAIRED = Path("Data/evals/models/latest_paired_model_comparison.json")
DEFAULT_OUTPUT = Path("Data/evals/models/latest_synthetic_simple_baseline_audit.json")


def build_simple_baseline_audit(
    rows_path: str | Path = DEFAULT_ROWS,
    paired_path: str | Path = DEFAULT_PAIRED,
) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(rows_path).open(encoding="utf-8", newline="")))
    labels = [_number(row.get("actual_label")) for row in rows]
    labels = [int(value) for value in labels if value is not None]
    if not labels:
        raise ValueError("No labelled synthetic rows are available for the baseline audit.")

    prevalence = mean(labels)
    majority_probability = 1.0 if prevalence >= 0.5 else 0.0
    classifiers = {
        "constant_half": [0.5] * len(labels),
        "posthoc_test_prevalence": [prevalence] * len(labels),
        "posthoc_test_majority": [majority_probability] * len(labels),
        "logistic_regression": _column(rows, "logistic_regression_probability"),
        "gradient_boosting_calibrated_champion": _column(rows, "gradient_boosting_calibrated_probability"),
    }
    classification = {
        name: _binary_metrics(labels, probabilities)
        for name, probabilities in classifiers.items()
        if len(probabilities) == len(labels)
    }

    actual_scores = _column(rows, "actual_response_score_percent")
    regression: dict[str, Any] = {}
    if len(actual_scores) == len(labels):
        posthoc_mean = mean(actual_scores)
        regressors = {
            "zero_signal": [0.0] * len(actual_scores),
            "posthoc_test_mean": [posthoc_mean] * len(actual_scores),
            "ridge_regression": _column(rows, "ridge_regression_response_score_percent"),
            "random_forest_regressor_champion": _column(rows, "random_forest_regressor_response_score_percent"),
        }
        regression = {
            name: _regression_metrics(actual_scores, predictions)
            for name, predictions in regressors.items()
            if len(predictions) == len(actual_scores)
        }

    paired = _read_json(Path(paired_path))
    logistic_comparison = next(
        (
            item for item in paired.get("classification", [])
            if item.get("candidate_model") == "logistic_regression"
        ),
        {},
    )
    p_value = logistic_comparison.get("p_value")
    champion_accuracy_delta = logistic_comparison.get("accuracy_delta_champion_minus_candidate")
    superiority_proven_over_logistic = bool(
        p_value is not None
        and float(p_value) < 0.05
        and champion_accuracy_delta is not None
        and float(champion_accuracy_delta) > 0
    )

    return {
        "schema_version": "synthetic_simple_baseline_audit_v1_2026_07",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if classification and regression else "needs_attention",
        "total_n": len(labels),
        "data_scope": "internal synthetic held-out rows",
        "classification_prevalence": round(prevalence, 6),
        "classification": classification,
        "regression": regression,
        "paired_champion_vs_logistic": {
            "method": logistic_comparison.get("method"),
            "accuracy_delta": champion_accuracy_delta,
            "p_value": p_value,
            "superiority_proven": superiority_proven_over_logistic,
        },
        "complexity_decision": (
            "retain_for_engineering_comparison_without_superiority_claim"
            if not superiority_proven_over_logistic
            else "synthetic_test_only_improvement_detected"
        ),
        "posthoc_baseline_warning": (
            "Prevalence, majority, and mean baselines use the evaluation labels and are descriptive lower-bound checks, "
            "not deployable fitted models."
        ),
        "promotion_allowed": False,
        "synthetic_only": True,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "This audit measures behavior on simulator-built rows. It does not establish clinical validity, "
            "patient benefit, treatment utility, or generalisation to real patients."
        ),
    }


def write_simple_baseline_audit(output_path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_simple_baseline_audit()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _column(rows: list[dict[str, str]], name: str) -> list[float]:
    values = [_number(row.get(name)) for row in rows]
    return [float(value) for value in values if value is not None]


def _binary_metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    predicted = [int(probability >= 0.5) for probability in probabilities]
    accuracy = mean(int(actual == estimate) for actual, estimate in zip(labels, predicted))
    brier = mean((probability - actual) ** 2 for actual, probability in zip(labels, probabilities))
    return {
        "accuracy": round(accuracy, 6),
        "brier_score": round(brier, 6),
        "auroc": round(_auroc(labels, probabilities), 6),
    }


def _regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [estimate - target for target, estimate in zip(actual, predicted)]
    return {
        "mae": round(mean(abs(error) for error in errors), 6),
        "rmse": round(mean(error * error for error in errors) ** 0.5, 6),
    }


def _auroc(labels: list[int], probabilities: list[float]) -> float:
    positives = [score for label, score in zip(labels, probabilities) if label == 1]
    negatives = [score for label, score in zip(labels, probabilities) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _number(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["build_simple_baseline_audit", "write_simple_baseline_audit"]
