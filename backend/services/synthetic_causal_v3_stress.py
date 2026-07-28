"""Causal-order synthetic stress benchmark for engineering evaluation only.

The generator makes its assumptions explicit, separates latent variables from
observable measurements, and evaluates simple and nonlinear baselines across
multiple seeds and controlled distribution shifts. It is not a patient model,
does not estimate treatment effects, and cannot provide clinical evidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUTPUT_PATH = Path("Data/evals/models/latest_synthetic_causal_v3_stress.json")
NUMERIC_FEATURES = [
    "baseline_reserve",
    "observed_burden",
    "cbc_signal",
    "symptom_signal",
    "imaging_signal",
    "cycle_index",
]
CATEGORICAL_FEATURES = ["synthetic_subgroup", "treatment_context"]
SCENARIOS = ("clean", "mar_missingness", "mnar_missingness", "measurement_noise", "subgroup_shift")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def _generate_frame(seed: int, n_rows: int, *, subgroup_probability: float = 0.30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subgroup = rng.binomial(1, subgroup_probability, n_rows)
    baseline_reserve = rng.normal(0.0 - 0.15 * subgroup, 1.0, n_rows)
    latent_burden = rng.normal(0.20 + 0.35 * subgroup, 1.0, n_rows)
    treatment_context = rng.choice(["context_a", "context_b", "context_c"], n_rows, p=[0.45, 0.35, 0.20])
    cycle_index = rng.integers(1, 9, n_rows)

    # The treatment field is context, not an intervention target. The synthetic
    # mechanism intentionally excludes it from the latent outcome equation.
    latent_response = (
        0.65 * baseline_reserve
        - 0.55 * latent_burden
        + 0.12 * np.log1p(cycle_index)
        + rng.normal(0.0, 0.85, n_rows)
    )
    observed_burden = latent_burden + rng.normal(0.0, 0.45, n_rows)
    cbc_signal = 0.45 * baseline_reserve - 0.35 * latent_burden + rng.normal(0.0, 0.75, n_rows)
    symptom_signal = 0.55 * latent_burden - 0.20 * latent_response + rng.normal(0.0, 0.85, n_rows)
    imaging_signal = 0.65 * latent_response - 0.25 * latent_burden + rng.normal(0.0, 0.70, n_rows)

    class_probability = _sigmoid(0.95 * latent_response - 0.15 * subgroup + rng.normal(0.0, 0.25, n_rows))
    class_target = rng.binomial(1, class_probability)
    regression_target = np.clip(50.0 + 14.0 * latent_response + rng.normal(0.0, 8.0, n_rows), 0.0, 100.0)
    return pd.DataFrame(
        {
            "synthetic_patient_id": [f"syn-{seed:03d}-{index:05d}" for index in range(n_rows)],
            "synthetic_subgroup": np.where(subgroup == 1, "shifted_group", "reference_group"),
            "treatment_context": treatment_context,
            "baseline_reserve": baseline_reserve,
            "observed_burden": observed_burden,
            "cbc_signal": cbc_signal,
            "symptom_signal": symptom_signal,
            "imaging_signal": imaging_signal,
            "cycle_index": cycle_index.astype(float),
            "response_pattern_label": class_target,
            "response_score_target": regression_target,
        }
    )


def _apply_scenario(frame: pd.DataFrame, scenario: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    if scenario == "clean" or scenario == "subgroup_shift":
        return out
    if scenario == "mar_missingness":
        for column in ("cbc_signal", "symptom_signal", "imaging_signal"):
            missing = rng.random(len(out)) < 0.20
            out.loc[missing, column] = np.nan
        return out
    if scenario == "mnar_missingness":
        severity = _sigmoid(out["symptom_signal"].to_numpy(dtype=float))
        imaging_missing = rng.random(len(out)) < (0.08 + 0.42 * severity)
        cbc_missing = rng.random(len(out)) < (0.06 + 0.30 * severity)
        out.loc[imaging_missing, "imaging_signal"] = np.nan
        out.loc[cbc_missing, "cbc_signal"] = np.nan
        return out
    if scenario == "measurement_noise":
        for column in ("observed_burden", "cbc_signal", "symptom_signal", "imaging_signal"):
            out[column] = out[column].to_numpy(dtype=float) + rng.normal(0.0, 0.65, len(out))
        return out
    raise ValueError(f"Unknown scenario: {scenario}")


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])


def _classification_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("features", _preprocessor()), ("model", LogisticRegression(max_iter=800, random_state=seed))]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("features", _preprocessor()),
                ("model", HistGradientBoostingClassifier(max_iter=40, learning_rate=0.08, max_depth=4, random_state=seed)),
            ]
        ),
    }


def _regression_models() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([("features", _preprocessor()), ("model", Ridge(alpha=2.0))]),
        "hist_gradient_boosting": Pipeline(
            [
                ("features", _preprocessor()),
                ("model", HistGradientBoostingRegressor(max_iter=40, learning_rate=0.08, max_depth=4, random_state=0)),
            ]
        ),
    }


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if mask.any():
            error += mask.mean() * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return float(error if total else np.nan)


def _bootstrap_mean_interval(values: list[float], seed: int = 20260722, draws: int = 2000) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(data, size=(draws, len(data)), replace=True), axis=1)
    return {
        "mean": round(float(data.mean()), 6),
        "ci95_low": round(float(np.quantile(means, 0.025)), 6),
        "ci95_high": round(float(np.quantile(means, 0.975)), 6),
    }


def _fit_seed(seed: int, n_train: int, n_test: int) -> list[dict[str, Any]]:
    train = _generate_frame(seed * 100 + 1, n_train)
    test_frames = {
        "clean": _generate_frame(seed * 100 + 2, n_test),
        "subgroup_shift": _generate_frame(seed * 100 + 3, n_test, subgroup_probability=0.70),
    }
    for scenario in ("mar_missingness", "mnar_missingness", "measurement_noise"):
        test_frames[scenario] = _apply_scenario(test_frames["clean"], scenario, seed * 1000 + len(scenario))

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    class_models = _classification_models(seed)
    regression_models = _regression_models()
    for model in class_models.values():
        model.fit(train[features], train["response_pattern_label"])
    for model in regression_models.values():
        model.fit(train[features], train["response_score_target"])

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        test = test_frames[scenario]
        labels = test["response_pattern_label"].to_numpy(dtype=int)
        targets = test["response_score_target"].to_numpy(dtype=float)
        class_metrics: dict[str, Any] = {}
        for name, model in class_models.items():
            probability = model.predict_proba(test[features])[:, 1]
            class_metrics[name] = {
                "auroc": float(roc_auc_score(labels, probability)),
                "brier": float(brier_score_loss(labels, probability)),
                "ece10": _ece(labels, probability),
            }
        regression_metrics: dict[str, Any] = {}
        for name, model in regression_models.items():
            prediction = model.predict(test[features])
            regression_metrics[name] = {"mae": float(mean_absolute_error(targets, prediction))}
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "classification": class_metrics,
                "regression": regression_metrics,
                "missing_fraction": float(test[features].isna().mean().mean()),
                "shifted_group_fraction": float((test["synthetic_subgroup"] == "shifted_group").mean()),
            }
        )
    return rows


def build_synthetic_causal_v3_stress(
    output_path: Path = OUTPUT_PATH,
    *,
    seeds: int = 30,
    n_train: int = 800,
    n_test: int = 400,
) -> dict[str, Any]:
    rows = [row for seed in range(seeds) for row in _fit_seed(seed, n_train, n_test)]
    summaries: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        auroc_delta = [
            row["classification"]["hist_gradient_boosting"]["auroc"]
            - row["classification"]["logistic_regression"]["auroc"]
            for row in scenario_rows
        ]
        brier_delta = [
            row["classification"]["hist_gradient_boosting"]["brier"]
            - row["classification"]["logistic_regression"]["brier"]
            for row in scenario_rows
        ]
        mae_delta = [
            row["regression"]["hist_gradient_boosting"]["mae"] - row["regression"]["ridge"]["mae"]
            for row in scenario_rows
        ]
        summaries[scenario] = {
            "nonlinear_minus_simple_auroc": _bootstrap_mean_interval(auroc_delta),
            "nonlinear_minus_simple_brier": _bootstrap_mean_interval(brier_delta),
            "nonlinear_minus_simple_mae": _bootstrap_mean_interval(mae_delta),
        }

    report = {
        "schema_version": "synthetic_causal_v3_stress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_internal_stress_test",
        "clinical_validation": False,
        "realism_claim": False,
        "model_promotion_decision": "HOLD",
        "seed_count": seeds,
        "n_train_per_seed": n_train,
        "n_test_per_scenario_per_seed": n_test,
        "scenarios": list(SCENARIOS),
        "causal_order": [
            "synthetic subgroup -> baseline reserve and latent burden",
            "baseline reserve and latent burden -> latent response",
            "latent state -> noisy observed CBC, symptom, and imaging signals",
            "latent response -> noisy synthetic classification and regression targets",
            "missingness and measurement processes -> observed evaluation frame",
        ],
        "explicit_non_causal_context": ["treatment_context"],
        "blocked_uses": [
            "treatment effect estimation",
            "clinical outcome prediction",
            "patient-level decision support",
            "clinical realism or external validity claim",
        ],
        "paired_seed_summaries": summaries,
        "seed_level_rows": rows,
        "claim_boundary": (
            "This benchmark tests engineering behavior under an explicit synthetic mechanism. "
            "It uses no real patient data, provides no clinical validation, and cannot support diagnosis, "
            "treatment, prognosis, patient benefit, or real-world readiness claims."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["SCENARIOS", "build_synthetic_causal_v3_stress"]
