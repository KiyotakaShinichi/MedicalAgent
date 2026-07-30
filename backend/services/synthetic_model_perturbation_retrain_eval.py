"""Patient-grouped retraining stress test across perturbations and generators."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


DEFAULT_SOURCE_PATH = Path(
    "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
)
REALISM_V2_PATH = Path(
    "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/models/latest_synthetic_model_perturbation_retrain_eval.json"
)
DIRECT_RESPONSE_PROXY = "mri_percent_change_from_baseline"
GUARDED_NUMERIC_FEATURES = [
    feature for feature in NUMERIC_FEATURES if feature != DIRECT_RESPONSE_PROXY
]
SEED = 42

CLAIM_BOUNDARY = (
    "All rows and labels in this evaluation are simulator-built. Perturbation "
    "robustness measures engineering sensitivity to synthetic assumptions; it "
    "does not establish clinical realism, external validity, treatment utility, "
    "patient benefit, or production healthcare readiness."
)


def build_synthetic_model_perturbation_retrain_eval(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    realism_v2_path: str | Path = REALISM_V2_PATH,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    source = _load_frame(source_path)
    realism = _load_frame(realism_v2_path)
    train, test = _patient_split(source, seed=seed)
    realism_train, realism_test = _patient_split(realism, seed=seed)

    full_clean = _fit_and_score(
        train, test, numeric_features=NUMERIC_FEATURES, seed=seed
    )
    guarded_clean = _fit_and_score(
        train, test, numeric_features=GUARDED_NUMERIC_FEATURES, seed=seed
    )
    scenarios = []
    for name in ("measurement_noise", "modality_dropout", "combined_noise"):
        perturbed_train = perturb_features(train, scenario=name, seed=seed)
        perturbed_test = perturb_features(test, scenario=name, seed=seed + 1)
        retrained = _fit_and_score(
            perturbed_train,
            perturbed_test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=seed,
        )
        clean_model_on_perturbed = _fit_and_score(
            train,
            perturbed_test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=seed,
        )
        scenarios.append(
            {
                "scenario": name,
                "retrained_on_perturbation": retrained,
                "clean_model_on_perturbed_test": clean_model_on_perturbed,
                "retrained_delta_vs_guarded_clean": _metric_deltas(
                    guarded_clean, retrained
                ),
            }
        )

    label_noisy_train = perturb_training_labels(train, seed=seed, fraction=0.05)
    label_noise = _fit_and_score(
        label_noisy_train,
        test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
    )
    scenarios.append(
        {
            "scenario": "five_percent_training_label_noise",
            "retrained_on_perturbation": label_noise,
            "clean_model_on_perturbed_test": None,
            "retrained_delta_vs_guarded_clean": _metric_deltas(
                guarded_clean, label_noise
            ),
        }
    )

    default_to_realism = _fit_and_score(
        train,
        realism_test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
    )
    realism_to_default = _fit_and_score(
        realism_train,
        test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
    )
    realism_internal = _fit_and_score(
        realism_train,
        realism_test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
    )
    generator_sensitivity = {
        "default_generator_internal": guarded_clean,
        "realism_v2_generator_internal": realism_internal,
        "train_default_test_realism_v2": default_to_realism,
        "train_realism_v2_test_default": realism_to_default,
        "default_to_realism_delta_vs_default_internal": _metric_deltas(
            guarded_clean, default_to_realism
        ),
        "realism_to_default_delta_vs_realism_internal": _metric_deltas(
            realism_internal, realism_to_default
        ),
    }

    stress_failures = _stress_failures(scenarios, generator_sensitivity)
    return {
        "schema_version": "synthetic_model_perturbation_retrain_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if stress_failures else "acceptable_synthetic_only",
        "clinical_validation": False,
        "production_ready": False,
        "source": {
            "default_generator_path": str(source_path).replace("\\", "/"),
            "realism_v2_path": str(realism_v2_path).replace("\\", "/"),
            "default_rows": len(source),
            "default_patients": int(source["patient_id"].nunique()),
            "realism_v2_rows": len(realism),
            "realism_v2_patients": int(realism["patient_id"].nunique()),
            "patient_overlap_between_train_and_test": 0,
        },
        "feature_policies": {
            "existing_full_feature_policy": {
                "numeric_features": NUMERIC_FEATURES,
                "metrics": full_clean,
                "direct_response_proxy_present": True,
                "risk": (
                    "mri_percent_change_from_baseline is definitionally close to "
                    "response_score_percent and can inflate regression evidence."
                ),
            },
            "guarded_primary_policy": {
                "numeric_features": GUARDED_NUMERIC_FEATURES,
                "metrics": guarded_clean,
                "direct_response_proxy_present": False,
                "delta_vs_full": _metric_deltas(full_clean, guarded_clean),
            },
        },
        "perturbation_scenarios": scenarios,
        "generator_version_sensitivity": generator_sensitivity,
        "stress_failures": stress_failures,
        "promotion_decision": "HOLD_SYNTHETIC_ONLY",
        "model_use_boundary": "monitor_only_engineering_signal",
        "limitations": [
            "Both generator versions share project assumptions and target semantics.",
            "Noise distributions are engineering stressors, not estimates of clinical measurement error.",
            "Cross-generator transfer is not external validation.",
            "Gradient boosting is a controlled benchmark, not a promoted clinical model.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_synthetic_model_perturbation_retrain_eval(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_synthetic_model_perturbation_retrain_eval(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def perturb_features(
    frame: pd.DataFrame,
    *,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    if scenario not in {"measurement_noise", "modality_dropout", "combined_noise"}:
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
    if scenario in {"modality_dropout", "combined_noise"}:
        patient_ids = output["patient_id"].drop_duplicates().to_numpy()
        selected = set(
            rng.choice(
                patient_ids,
                size=max(1, round(len(patient_ids) * 0.25)),
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


def _fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    numeric_features: list[str],
    seed: int,
) -> dict[str, Any]:
    features = [
        feature
        for feature in numeric_features + CATEGORICAL_FEATURES
        if feature in train.columns and feature in test.columns
    ]
    numeric = [feature for feature in features if feature in numeric_features]
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    classifier = Pipeline(
        [
            ("preprocessor", _preprocessor(numeric, categorical)),
            ("model", GradientBoostingClassifier(random_state=seed)),
        ]
    )
    regressor = Pipeline(
        [
            ("preprocessor", _preprocessor(numeric, categorical)),
            ("model", GradientBoostingRegressor(random_state=seed, loss="huber")),
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
    intervals = _bootstrap_metric_intervals(patient, seed=seed)
    return {
        "patient_count": len(patient),
        "classification_auroc": round(auc, 6) if auc is not None else None,
        "classification_brier": round(
            float(brier_score_loss(patient["label"], patient["probability"])), 6
        ),
        "regression_mae": round(
            float(mean_absolute_error(patient["response"], patient["prediction"])), 6
        ),
        "bootstrap_95_ci": intervals,
        "bootstrap_resamples": 300,
    }


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


__all__ = [
    "build_synthetic_model_perturbation_retrain_eval",
    "perturb_features",
    "perturb_training_labels",
    "write_synthetic_model_perturbation_retrain_eval",
]
