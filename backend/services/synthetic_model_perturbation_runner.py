"""Scenario execution for the perturbation retrain evaluation.

Loads the synthetic frame, applies a perturbation, splits by patient, fits the
model, and scores it - once per scenario, and repeatedly across a fixed set of
seeds for the stability check.

Two properties here are contractual and easy to break by accident:

* **Splits are grouped by patient.** Rows from one patient must never straddle
  the train/test boundary, or the model is evaluated on patients it memorised
  and every metric is optimistic.
* **Seeds are fixed and consumed in a fixed order.** `SEED` and
  `REPEATED_SPLIT_SEEDS` make the whole evaluation reproducible; reordering the
  calls that draw from them changes every number downstream without changing
  any threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.services.synthetic_model_perturbation_constants import (
    CATEGORICAL_FEATURES,
    GUARDED_NUMERIC_FEATURES,
    REPEATED_SPLIT_SEEDS,
)
from backend.services.synthetic_model_perturbation_metrics import (
    _bootstrap_metric_intervals,
    _classification_abstention_curve,
    _expected_calibration_error,
    _metric_deltas,
    _metric_distribution,
)


def _load_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "patient_id",
        "treatment_success_binary",
        "response_score_percent",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return frame


def perturb_features(
    frame: pd.DataFrame,
    *,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    if scenario not in {
        "measurement_noise",
        "modality_dropout",
        "severe_modality_dropout",
        "mnar_severity_dependent_dropout",
        "combined_noise",
    }:
        raise ValueError(f"unsupported perturbation scenario: {scenario}")
    output = frame.copy()
    rng = np.random.default_rng(seed)
    if scenario in {"measurement_noise", "combined_noise"}:
        continuous = [
            feature
            for feature in GUARDED_NUMERIC_FEATURES
            if feature not in {"cycle", "age", "dose_delayed", "dose_reduced"}
            and feature in output.columns
        ]
        for feature in continuous:
            values = pd.to_numeric(output[feature], errors="coerce")
            scale = float(values.std(skipna=True) or 0.0) * 0.10
            if scale:
                noise = rng.normal(0.0, scale, size=len(output))
                output[feature] = values + noise
    if scenario in {
        "modality_dropout",
        "severe_modality_dropout",
        "combined_noise",
    }:
        patient_ids = output["patient_id"].drop_duplicates().to_numpy()
        dropout_fraction = 0.50 if scenario == "severe_modality_dropout" else 0.25
        selected = set(
            rng.choice(
                patient_ids,
                size=max(1, round(len(patient_ids) * dropout_fraction)),
                replace=False,
            ).tolist()
        )
        mask = output["patient_id"].isin(selected)
        for feature in (
            "mri_tumor_size_cm",
            "pre_wbc",
            "pre_anc",
            "pre_hemoglobin",
            "pre_platelets",
            "max_symptom_severity",
            "symptom_count",
            "intervention_count",
            "dose_delayed",
            "dose_reduced",
        ):
            if feature in output.columns:
                output.loc[mask, feature] = np.nan
    if scenario == "mnar_severity_dependent_dropout":
        severity = pd.to_numeric(
            output.get("max_symptom_severity", pd.Series(0.0, index=output.index)),
            errors="coerce",
        ).fillna(0.0)
        severity_scale = severity / max(float(severity.max() or 1.0), 1.0)
        low_wbc = pd.to_numeric(
            output.get("pre_wbc", pd.Series(np.nan, index=output.index)),
            errors="coerce",
        )
        low_wbc_signal = (
            low_wbc < low_wbc.median(skipna=True)
        ).fillna(False).astype(float)
        missing_probability = np.clip(
            0.10 + (0.55 * severity_scale) + (0.20 * low_wbc_signal),
            0.0,
            0.90,
        )
        mask = rng.random(len(output)) < missing_probability.to_numpy()
        for feature in (
            "mri_tumor_size_cm",
            "pre_wbc",
            "pre_anc",
            "pre_hemoglobin",
            "pre_platelets",
        ):
            if feature in output.columns:
                output.loc[mask, feature] = np.nan
    return output


def perturb_training_labels(
    frame: pd.DataFrame,
    *,
    seed: int,
    fraction: float,
) -> pd.DataFrame:
    output = frame.copy()
    rng = np.random.default_rng(seed)
    patients = output["patient_id"].drop_duplicates().to_numpy()
    selected = set(
        rng.choice(
            patients,
            size=max(1, round(len(patients) * fraction)),
            replace=False,
        ).tolist()
    )
    mask = output["patient_id"].isin(selected)
    output.loc[mask, "treatment_success_binary"] = (
        1 - output.loc[mask, "treatment_success_binary"].astype(int)
    )
    return output


def _patient_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_labels = (
        frame.groupby("patient_id", as_index=False)["treatment_success_binary"]
        .first()
        .sort_values("patient_id")
    )
    rng = np.random.default_rng(seed)
    test_ids = []
    for _, group in patient_labels.groupby("treatment_success_binary"):
        ids = group["patient_id"].to_numpy().copy()
        rng.shuffle(ids)
        count = max(1, round(len(ids) * test_fraction))
        test_ids.extend(ids[:count].tolist())
    test_id_set = set(test_ids)
    train = frame[~frame["patient_id"].isin(test_id_set)].copy()
    test = frame[frame["patient_id"].isin(test_id_set)].copy()
    return train, test


def _preprocessor(
    numeric: list[str],
    categorical: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    numeric_features: list[str],
    seed: int,
    model_family: str = "gradient_boosting",
    include_bootstrap_intervals: bool = True,
) -> dict[str, Any]:
    features = [
        feature
        for feature in numeric_features + CATEGORICAL_FEATURES
        if feature in train.columns and feature in test.columns
    ]
    numeric = [feature for feature in features if feature in numeric_features]
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    if model_family == "gradient_boosting":
        classifier_model = GradientBoostingClassifier(random_state=seed)
        regressor_model = GradientBoostingRegressor(
            random_state=seed,
            loss="huber",
        )
    elif model_family == "linear":
        classifier_model = LogisticRegression(
            max_iter=2_000,
            random_state=seed,
        )
        regressor_model = Ridge(alpha=1.0)
    else:
        raise ValueError(f"unsupported model_family: {model_family}")
    classifier = Pipeline(
        [
            ("preprocessor", _preprocessor(numeric, categorical)),
            ("model", classifier_model),
        ]
    )
    regressor = Pipeline(
        [
            ("preprocessor", _preprocessor(numeric, categorical)),
            ("model", regressor_model),
        ]
    )
    classifier.fit(train[features], train["treatment_success_binary"].astype(int))
    regressor.fit(train[features], train["response_score_percent"].astype(float))
    probabilities = classifier.predict_proba(test[features])[:, 1]
    regression = regressor.predict(test[features])
    scored = pd.DataFrame(
        {
            "patient_id": test["patient_id"].to_numpy(),
            "label": test["treatment_success_binary"].astype(int).to_numpy(),
            "probability": probabilities,
            "response": test["response_score_percent"].astype(float).to_numpy(),
            "prediction": regression,
        }
    )
    patient = scored.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        probability=("probability", "mean"),
        response=("response", "last"),
        prediction=("prediction", "last"),
    )
    auc = (
        float(roc_auc_score(patient["label"], patient["probability"]))
        if patient["label"].nunique() > 1
        else None
    )
    intervals = (
        _bootstrap_metric_intervals(patient, seed=seed)
        if include_bootstrap_intervals
        else None
    )
    predicted_labels = (patient["probability"] >= 0.5).astype(int)
    return {
        "model_family": model_family,
        "patient_count": len(patient),
        "classification_auroc": round(auc, 6) if auc is not None else None,
        "classification_accuracy": round(
            float(accuracy_score(patient["label"], predicted_labels)),
            6,
        ),
        "classification_balanced_accuracy": round(
            float(balanced_accuracy_score(patient["label"], predicted_labels)),
            6,
        ),
        "classification_brier": round(
            float(brier_score_loss(patient["label"], patient["probability"])), 6
        ),
        "classification_ece_10_bin": round(
            _expected_calibration_error(
                patient["label"].to_numpy(),
                patient["probability"].to_numpy(),
            ),
            6,
        ),
        "regression_mae": round(
            float(mean_absolute_error(patient["response"], patient["prediction"])), 6
        ),
        "classification_abstention_curve": _classification_abstention_curve(
            patient
        ),
        "bootstrap_95_ci": intervals,
        "bootstrap_resamples": 300 if include_bootstrap_intervals else 0,
    }


def _repeated_patient_split_stability(
    source: pd.DataFrame,
    realism: pd.DataFrame,
    *,
    seeds: tuple[int, ...] = REPEATED_SPLIT_SEEDS,
) -> dict[str, Any]:
    """Measure split sensitivity without treating repeated splits as external data."""
    rows = []
    for split_seed in seeds:
        train, test = _patient_split(source, seed=split_seed)
        _, realism_test = _patient_split(realism, seed=split_seed)
        guarded = _fit_and_score(
            train,
            test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=split_seed,
            include_bootstrap_intervals=False,
        )
        linear = _fit_and_score(
            train,
            test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=split_seed,
            model_family="linear",
            include_bootstrap_intervals=False,
        )
        cross_generator = _fit_and_score(
            train,
            realism_test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=split_seed,
            include_bootstrap_intervals=False,
        )
        delta = _metric_deltas(linear, guarded)
        threshold_met = bool(
            (delta.get("classification_auroc") or 0.0) >= 0.02
            and (delta.get("classification_brier") or 0.0) <= 0.0
            and (delta.get("regression_mae") or 0.0) <= -1.0
        )
        rows.append(
            {
                "seed": split_seed,
                "train_patient_count": int(train["patient_id"].nunique()),
                "test_patient_count": int(test["patient_id"].nunique()),
                "patient_overlap_count": len(
                    set(train["patient_id"]) & set(test["patient_id"])
                ),
                "guarded_primary": guarded,
                "logistic_ridge": linear,
                "guarded_minus_linear": delta,
                "complex_lift_threshold_met": threshold_met,
                "train_default_test_realism_v2": cross_generator,
            }
        )
    metrics = ("classification_auroc", "classification_brier", "regression_mae")
    return {
        "seed_policy": "predeclared_fixed_seed_set",
        "seeds": list(seeds),
        "split_count": len(rows),
        "patient_overlap_count_max": max(
            (row["patient_overlap_count"] for row in rows),
            default=0,
        ),
        "guarded_primary_distributions": {
            metric: _metric_distribution(
                row["guarded_primary"].get(metric) for row in rows
            )
            for metric in metrics
        },
        "guarded_minus_linear_delta_distributions": {
            metric: _metric_distribution(
                row["guarded_minus_linear"].get(metric) for row in rows
            )
            for metric in metrics
        },
        "cross_generator_distributions": {
            metric: _metric_distribution(
                row["train_default_test_realism_v2"].get(metric) for row in rows
            )
            for metric in metrics
        },
        "complex_lift_threshold_pass_rate": round(
            sum(row["complex_lift_threshold_met"] for row in rows)
            / max(len(rows), 1),
            6,
        ),
        "rows": rows,
        "interpretation": (
            "Repeated patient-grouped splits measure sensitivity to the selected "
            "synthetic partition. Their empirical ranges are not confidence "
            "intervals for real patients or evidence of transportability."
        ),
    }
