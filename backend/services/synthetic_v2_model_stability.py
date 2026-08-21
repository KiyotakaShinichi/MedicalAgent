"""Repeated patient-grouped stability evaluation on synthetic temporal data.

This benchmark measures engineering behavior under controlled perturbations.
It cannot establish clinical performance, realism, or patient benefit and its
promotion policy always remains HOLD while evidence is synthetic-only.
"""

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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REALISM_V2_INPUT_PATH = Path("Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv")
LEGACY_INPUT_PATH = Path("Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
DEFAULT_INPUT_PATH = REALISM_V2_INPUT_PATH if REALISM_V2_INPUT_PATH.exists() else LEGACY_INPUT_PATH
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_synthetic_v2_model_stability.json")
SEEDS = (17, 29, 43, 71, 101)
SCENARIOS = ("clean", "mar_missingness", "mnar_missingness", "measurement_noise", "label_noise", "subgroup_shift")

TARGET_OR_PROXY_COLUMNS = {
    "patient_id", "treatment_date", "treatment_success_binary", "response_score_percent",
    "mri_percent_change_from_baseline", "latent_response_strength", "toxicity_risk_binary",
    "urgent_intervention_needed", "support_intervention_needed", "cycle_response_trend_class",
    "final_response_category", "final_cancer_status", "maintenance_needed", "final_response_multiclass",
}


def run_synthetic_v2_model_stability(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    required = {"patient_id", "treatment_success_binary", "response_score_percent"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["patient_id", "treatment_success_binary", "response_score_percent"]).copy()
    feature_columns = [column for column in frame.columns if column not in TARGET_OR_PROXY_COLUMNS]
    groups = frame["patient_id"].astype(str)
    y_class = frame["treatment_success_binary"].astype(int)
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_index, test_index = next(splitter.split(frame, y_class, groups))
        if set(groups.iloc[train_index]) & set(groups.iloc[test_index]):
            raise AssertionError("patient overlap detected")
        base_train = frame.iloc[train_index].reset_index(drop=True)
        base_test = frame.iloc[test_index].reset_index(drop=True)
        for scenario in SCENARIOS:
            rng = np.random.default_rng(seed * 100 + SCENARIOS.index(scenario))
            train, train_y_class, train_y_reg = _perturb(
                base_train, scenario, rng, training=True,
            )
            test, test_y_class, test_y_reg = _perturb(
                base_test, scenario, rng, training=False,
            )
            X_train = train[feature_columns]
            X_test = test[feature_columns]
            for model_name, model in _classification_models(seed).items():
                pipeline = _pipeline(X_train, model, scale=model_name == "logistic_regression")
                pipeline.fit(X_train, train_y_class)
                probability = pipeline.predict_proba(X_test)[:, 1]
                rows.append({
                    "seed": seed, "scenario": scenario, "task": "classification", "model": model_name,
                    "n_train": len(X_train), "n_test": len(X_test),
                    "patient_overlap_count": 0,
                    "auroc": _safe_auc(test_y_class, probability),
                    "brier": round(float(brier_score_loss(test_y_class, probability)), 6),
                    "ece": _ece(test_y_class.to_numpy(), probability),
                })
            for model_name, model in _regression_models(seed).items():
                pipeline = _pipeline(X_train, model, scale=model_name == "ridge")
                pipeline.fit(X_train, train_y_reg)
                prediction = pipeline.predict(X_test)
                rows.append({
                    "seed": seed, "scenario": scenario, "task": "regression", "model": model_name,
                    "n_train": len(X_train), "n_test": len(X_test),
                    "patient_overlap_count": 0,
                    "mae": round(float(mean_absolute_error(test_y_reg, prediction)), 6),
                })

    aggregate = _aggregate(rows)
    decision = _promotion_decision(rows)
    payload = {
        "schema_version": "synthetic_v2_model_stability_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if rows and all(row["patient_overlap_count"] == 0 for row in rows) else "needs_attention",
        "input_path": Path(input_path).as_posix(),
        "n_rows": len(frame), "n_patients": int(groups.nunique()),
        "seeds": list(SEEDS), "scenarios": list(SCENARIOS),
        "feature_columns": feature_columns,
        "excluded_target_or_proxy_columns": sorted(TARGET_OR_PROXY_COLUMNS & set(frame.columns)),
        "runs": rows,
        "aggregate": aggregate,
        "paired_comparisons": _paired_comparisons(rows),
        "promotion_decision": decision,
        "synthetic_only": True,
        "clinical_validation": False,
        "claim_boundary": (
            "Repeated grouped stress tests are synthetic engineering evidence only. They do not show clinical "
            "accuracy, realism, patient benefit, treatment utility, or production healthcare readiness."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _perturb(
    frame: pd.DataFrame,
    scenario: str,
    rng: np.random.Generator,
    *,
    training: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = frame.copy()
    y_class = data["treatment_success_binary"].astype(int).copy()
    y_reg = data["response_score_percent"].astype(float).copy()
    numeric_candidates = [
        column for column in (
            "pre_wbc", "pre_anc", "pre_hemoglobin", "pre_platelets", "nadir_wbc", "nadir_anc",
            "nadir_hemoglobin", "nadir_platelets", "recovery_wbc", "recovery_hemoglobin",
            "recovery_platelets", "mri_tumor_size_cm", "max_symptom_severity",
        ) if column in data.columns
    ]
    if scenario == "mar_missingness":
        cycle = pd.to_numeric(data.get("cycle", 1), errors="coerce").fillna(1).to_numpy()
        probability = np.clip(0.08 + 0.025 * cycle, 0.08, 0.30)
        for column in numeric_candidates:
            data.loc[rng.random(len(data)) < probability, column] = np.nan
    elif scenario == "mnar_missingness":
        severity = pd.to_numeric(data.get("max_symptom_severity", 0), errors="coerce").fillna(0).to_numpy()
        probability = np.clip(0.05 + 0.035 * severity, 0.05, 0.38)
        for column in numeric_candidates:
            data.loc[rng.random(len(data)) < probability, column] = np.nan
    elif scenario == "measurement_noise":
        for column in numeric_candidates:
            values = pd.to_numeric(data[column], errors="coerce")
            scale = float(values.std(skipna=True) or 0) * 0.12
            if scale:
                data[column] = values + rng.normal(0, scale, len(data))
    elif scenario == "label_noise" and training:
        flip = rng.random(len(data)) < 0.10
        y_class.loc[flip] = 1 - y_class.loc[flip]
        y_reg = y_reg + rng.normal(0, max(float(y_reg.std()) * 0.10, 1.0), len(data))
    elif scenario == "subgroup_shift":
        subgroup = data.get("molecular_subtype", pd.Series("unknown", index=data.index)).astype(str).str.contains("HER2", case=False, na=False)
        if subgroup.any():
            for column in numeric_candidates:
                mask = subgroup.to_numpy() & (rng.random(len(data)) < (0.18 if training else 0.32))
                data.loc[mask, column] = np.nan
    return data, y_class, y_reg


def _pipeline(X: pd.DataFrame, model: Any, *, scale: bool) -> Pipeline:
    numeric = list(X.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in X.columns if column not in numeric]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _classification_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed, n_estimators=80, max_depth=2),
    }


def _regression_models(seed: int) -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=seed, n_estimators=80, max_depth=2, loss="huber"),
    }


def _safe_auc(y_true: pd.Series, probability: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return round(float(roc_auc_score(y_true, probability)), 6)


def _ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            error += mask.mean() * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return round(error, 6)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["task"], row["model"]), []).append(row)
    output = []
    for (scenario, task, model), items in sorted(grouped.items()):
        metrics = ("auroc", "brier", "ece") if task == "classification" else ("mae",)
        summary: dict[str, Any] = {"scenario": scenario, "task": task, "model": model, "n_seeds": len(items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            summary[f"mean_{metric}"] = round(float(np.mean(values)), 6) if values else None
            summary[f"min_{metric}"] = round(float(np.min(values)), 6) if values else None
            summary[f"max_{metric}"] = round(float(np.max(values)), 6) if values else None
        output.append(summary)
    return output


def _promotion_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [row for row in rows if row["scenario"] == "clean"]
    class_by_seed = {seed: {} for seed in SEEDS}
    reg_by_seed = {seed: {} for seed in SEEDS}
    for row in clean:
        target = class_by_seed if row["task"] == "classification" else reg_by_seed
        target[row["seed"]][row["model"]] = row
    class_wins = sum(
        bucket.get("gradient_boosting", {}).get("auroc", -1) > bucket.get("logistic_regression", {}).get("auroc", -1)
        for bucket in class_by_seed.values()
    )
    reg_wins = sum(
        bucket.get("gradient_boosting_regressor", {}).get("mae", float("inf")) < bucket.get("ridge", {}).get("mae", float("inf"))
        for bucket in reg_by_seed.values()
    )
    return {
        "decision": "HOLD",
        "promotion_allowed": False,
        "complex_classifier_clean_seed_wins": class_wins,
        "complex_regressor_clean_seed_wins": reg_wins,
        "required_seed_wins_for_engineering_candidate": 4,
        "reason": "Synthetic-only repeated stress evidence cannot authorize clinical or patient-facing promotion.",
        "current_safe_use": "monitor_only_or_review_hint_only",
        "future_requirements": [
            "externally authored evaluation design", "clinician-reviewed labels", "external temporal validation",
            "subgroup and calibration review", "failure-case review", "clinical and governance approval",
        ],
    }


def refresh_synthetic_v2_summary(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Refresh derived summaries without retraining the already-recorded runs."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    rows = list(payload.get("runs") or [])
    payload["aggregate"] = _aggregate(rows)
    payload["paired_comparisons"] = _paired_comparisons(rows)
    payload["promotion_decision"] = _promotion_decision(rows)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    rng = np.random.default_rng(20260718)
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for task in ("classification", "regression"):
            selected = [row for row in scenario_rows if row["task"] == task]
            by_seed: dict[int, dict[str, dict[str, Any]]] = {}
            for row in selected:
                by_seed.setdefault(int(row["seed"]), {})[row["model"]] = row
            deltas = []
            for bucket in by_seed.values():
                if task == "classification":
                    simple = bucket.get("logistic_regression")
                    complex_row = bucket.get("gradient_boosting")
                    if simple and complex_row and simple.get("auroc") is not None and complex_row.get("auroc") is not None:
                        deltas.append(float(complex_row["auroc"]) - float(simple["auroc"]))
                else:
                    simple = bucket.get("ridge")
                    complex_row = bucket.get("gradient_boosting_regressor")
                    if simple and complex_row:
                        # Positive means the complex model has lower MAE.
                        deltas.append(float(simple["mae"]) - float(complex_row["mae"]))
            if not deltas:
                continue
            values = np.asarray(deltas, dtype=float)
            samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
            comparisons.append({
                "scenario": scenario,
                "task": task,
                "metric": "auroc_delta_complex_minus_simple" if task == "classification" else "mae_reduction_simple_minus_complex",
                "n_paired_seeds": len(values),
                "mean_delta": round(float(values.mean()), 6),
                "bootstrap_95_ci": [round(float(np.quantile(samples, 0.025)), 6), round(float(np.quantile(samples, 0.975)), 6)],
                "complex_seed_win_count": int((values > 0).sum()),
                "uncertainty_note": "Bootstrap over five synthetic grouped splits; narrow or positive intervals do not imply external or clinical generalization.",
            })
    return comparisons


__all__ = ["refresh_synthetic_v2_summary", "run_synthetic_v2_model_stability"]
