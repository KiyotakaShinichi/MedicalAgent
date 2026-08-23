"""Metric computation and aggregation for the perturbation retrain evaluation.

Everything here turns predictions into numbers: calibration error, abstention
curves, bootstrap intervals, metric deltas and distributions across repeated
splits, and the count of scenarios that breached a stress threshold.

`_train_only_constant_baseline` lives here rather than with the runner because
it trains nothing - it scores a constant predictor so a trained model has an
honest floor to be compared against. A model that cannot beat it has not
learned anything, and without that reference an AUC on saturated synthetic data
reads far better than it is.

No threshold is defined here; thresholds belong to the evaluation that calls
these functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)


def _metric_distribution(values: Any) -> dict[str, Any]:
    clean = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=float,
    )
    if not len(clean):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(clean)),
        "mean": round(float(clean.mean()), 6),
        "std": round(float(clean.std(ddof=1)), 6) if len(clean) > 1 else 0.0,
        "min": round(float(clean.min()), 6),
        "median": round(float(np.median(clean)), 6),
        "max": round(float(clean.max()), 6),
        "empirical_10_90_range": [
            round(float(np.percentile(clean, 10)), 6),
            round(float(np.percentile(clean, 90)), 6),
        ],
    }


def _metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for metric in ("classification_auroc", "classification_brier", "regression_mae"):
        left = before.get(metric)
        right = after.get(metric)
        output[metric] = (
            round(float(right) - float(left), 6)
            if left is not None and right is not None
            else None
        )
    return output


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    if len(labels) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(labels))
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        observed = float(np.mean(labels[mask]))
        predicted = float(np.mean(probabilities[mask]))
        error += float(np.sum(mask)) / total * abs(observed - predicted)
    return float(error)


def _classification_abstention_curve(
    patient: pd.DataFrame,
) -> list[dict[str, Any]]:
    ranked = patient.assign(
        confidence=(patient["probability"] - 0.5).abs()
    ).sort_values(["confidence", "patient_id"], ascending=[False, True])
    curve: list[dict[str, Any]] = []
    for coverage in (1.0, 0.9, 0.75, 0.5):
        retained_n = max(1, int(np.ceil(len(ranked) * coverage)))
        retained = ranked.head(retained_n)
        predicted = (retained["probability"] >= 0.5).astype(int)
        curve.append(
            {
                "coverage": coverage,
                "n_retained": retained_n,
                "accuracy": round(
                    float(accuracy_score(retained["label"], predicted)),
                    6,
                ),
                "balanced_accuracy": round(
                    float(balanced_accuracy_score(retained["label"], predicted)),
                    6,
                ),
                "brier": round(
                    float(
                        brier_score_loss(
                            retained["label"],
                            retained["probability"],
                        )
                    ),
                    6,
                ),
            }
        )
    return curve


def _stress_failures(
    scenarios: list[dict[str, Any]],
    generator: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    candidates = [
        (row["scenario"], row["retrained_delta_vs_guarded_clean"])
        for row in scenarios
    ]
    candidates.extend(
        [
            (
                "train_default_test_realism_v2",
                generator["default_to_realism_delta_vs_default_internal"],
            ),
            (
                "train_realism_v2_test_default",
                generator["realism_to_default_delta_vs_realism_internal"],
            ),
        ]
    )
    for scenario, deltas in candidates:
        if (
            (deltas.get("classification_auroc") or 0) < -0.05
            or (deltas.get("classification_brier") or 0) > 0.03
            or (deltas.get("regression_mae") or 0) > 5.0
        ):
            failures.append({"scenario": scenario, "metric_deltas": deltas})
    return failures


def _bootstrap_metric_intervals(
    patient: pd.DataFrame,
    *,
    seed: int,
    resamples: int = 300,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    briers: list[float] = []
    maes: list[float] = []
    size = len(patient)
    for _ in range(resamples):
        sample = patient.iloc[rng.integers(0, size, size=size)]
        if sample["label"].nunique() > 1:
            aucs.append(
                float(roc_auc_score(sample["label"], sample["probability"]))
            )
        briers.append(
            float(brier_score_loss(sample["label"], sample["probability"]))
        )
        maes.append(
            float(mean_absolute_error(sample["response"], sample["prediction"]))
        )
    return {
        "classification_auroc": _percentile_interval(aucs),
        "classification_brier": _percentile_interval(briers),
        "regression_mae": _percentile_interval(maes),
    }


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [
        round(float(np.percentile(values, 2.5)), 6),
        round(float(np.percentile(values, 97.5)), 6),
    ]


def _train_only_constant_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, Any]:
    del seed  # The baseline is intentionally deterministic.
    train_patient = (
        train.groupby("patient_id", as_index=False)
        .agg(
            label=("treatment_success_binary", "max"),
            response=("response_score_percent", "mean"),
        )
    )
    test_patient = (
        test.groupby("patient_id", as_index=False)
        .agg(
            label=("treatment_success_binary", "max"),
            response=("response_score_percent", "mean"),
        )
    )
    probability = float(train_patient["label"].mean())
    response = float(train_patient["response"].mean())
    probabilities = np.full(len(test_patient), probability, dtype=float)
    predictions = np.full(len(test_patient), response, dtype=float)
    predicted_labels = (probabilities >= 0.5).astype(int)
    return {
        "model_family": "train_only_constant",
        "patient_count": len(test_patient),
        "classification_auroc": 0.5 if test_patient["label"].nunique() > 1 else None,
        "classification_accuracy": round(
            float(accuracy_score(test_patient["label"], predicted_labels)),
            6,
        ),
        "classification_balanced_accuracy": round(
            float(
                balanced_accuracy_score(
                    test_patient["label"],
                    predicted_labels,
                )
            ),
            6,
        ),
        "classification_brier": round(
            float(brier_score_loss(test_patient["label"], probabilities)),
            6,
        ),
        "classification_ece_10_bin": round(
            _expected_calibration_error(
                test_patient["label"].to_numpy(),
                probabilities,
            ),
            6,
        ),
        "regression_mae": round(
            float(mean_absolute_error(test_patient["response"], predictions)),
            6,
        ),
        "classification_abstention_curve": [
            {
                "coverage": 1.0,
                "n_retained": len(test_patient),
                "accuracy": round(
                    float(
                        accuracy_score(
                            test_patient["label"],
                            predicted_labels,
                        )
                    ),
                    6,
                ),
                "balanced_accuracy": round(
                    float(
                        balanced_accuracy_score(
                            test_patient["label"],
                            predicted_labels,
                        )
                    ),
                    6,
                ),
                "brier": round(
                    float(
                        brier_score_loss(
                            test_patient["label"],
                            probabilities,
                        )
                    ),
                    6,
                ),
            }
        ],
        "training_only_statistics": {
            "positive_rate": round(probability, 6),
            "mean_response_score": round(response, 6),
        },
    }
