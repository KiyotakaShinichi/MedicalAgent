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
COVERAGE_LEVELS = (1.0, 0.9, 0.75, 0.5)


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
    classification_selective_risk = {
        name: _classification_selective_risk(labels, probabilities)
        for name, probabilities in classifiers.items()
        if len(probabilities) == len(labels)
    }

    actual_scores = _column(rows, "actual_response_score_percent")
    regression: dict[str, Any] = {}
    regression_disagreement_abstention: dict[str, Any] = {}
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
        ridge = regressors.get("ridge_regression", [])
        forest = regressors.get("random_forest_regressor_champion", [])
        if len(ridge) == len(actual_scores) and len(forest) == len(actual_scores):
            regression_disagreement_abstention = _regression_disagreement_curve(
                actual_scores, ridge, forest,
            )

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
        "classification_selective_risk": classification_selective_risk,
        "regression": regression,
        "regression_disagreement_abstention": regression_disagreement_abstention,
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
        "uncertainty_boundary": (
            "Classification confidence and regression model disagreement are selective-risk engineering proxies. "
            "They are not calibrated clinical uncertainty or evidence that abstention is safe for patient care."
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


def _classification_selective_risk(
    labels: list[int], probabilities: list[float]
) -> dict[str, Any]:
    ranked = sorted(
        zip(labels, probabilities),
        key=lambda item: abs(float(item[1]) - 0.5),
        reverse=True,
    )
    points = []
    for coverage in COVERAGE_LEVELS:
        keep = max(1, min(len(ranked), int(round(len(ranked) * coverage))))
        selected = ranked[:keep]
        selected_labels = [int(item[0]) for item in selected]
        selected_probabilities = [float(item[1]) for item in selected]
        metrics = _binary_metrics(selected_labels, selected_probabilities)
        points.append({
            "requested_coverage": coverage,
            "observed_coverage": round(keep / len(ranked), 6),
            "n_kept": keep,
            "accuracy": metrics["accuracy"],
            "selective_risk": round(1.0 - metrics["accuracy"], 6),
            "brier_score": metrics["brier_score"],
            "auroc": metrics["auroc"],
        })
    return {
        "ranking_signal": "absolute_distance_from_probability_0.5",
        "points": points,
        "mean_selective_risk_proxy": round(mean(point["selective_risk"] for point in points), 6),
    }


def _regression_disagreement_curve(
    actual: list[float], ridge: list[float], forest: list[float]
) -> dict[str, Any]:
    ranked = sorted(
        zip(actual, ridge, forest),
        key=lambda item: abs(float(item[1]) - float(item[2])),
    )
    points = []
    for coverage in COVERAGE_LEVELS:
        keep = max(1, min(len(ranked), int(round(len(ranked) * coverage))))
        selected = ranked[:keep]
        metrics = _regression_metrics(
            [float(item[0]) for item in selected],
            [float(item[2]) for item in selected],
        )
        points.append({
            "requested_coverage": coverage,
            "observed_coverage": round(keep / len(ranked), 6),
            "n_kept": keep,
            "random_forest_mae": metrics["mae"],
            "random_forest_rmse": metrics["rmse"],
            "max_kept_model_disagreement": round(
                max(abs(float(item[1]) - float(item[2])) for item in selected), 6
            ),
        })
    return {
        "ranking_signal": "absolute_ridge_vs_random_forest_disagreement",
        "calibrated_uncertainty": False,
        "points": points,
        "interpretation": "Lower-disagreement rows are retained first; this is a model-disagreement proxy only.",
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
