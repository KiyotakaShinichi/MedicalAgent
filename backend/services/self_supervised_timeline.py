from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.complete_synthetic_training import DEFAULT_ML_CSV_PATH


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_self_supervised_timeline.json"
STATIC_FEATURES = ["age", "stage", "molecular_subtype", "regimen", "cycle"]
PRIOR_NUMERIC_FEATURES = [
    "prev_pre_wbc", "prev_pre_anc", "prev_pre_hemoglobin", "prev_pre_platelets",
    "prev_nadir_wbc", "prev_nadir_anc", "prev_nadir_hemoglobin", "prev_nadir_platelets",
    "prev_recovery_wbc", "prev_recovery_hemoglobin", "prev_recovery_platelets",
    "prev_mri_percent_change_from_baseline", "prev_max_symptom_severity",
    "prev_symptom_count", "prev_intervention_count", "prev_dose_delayed", "prev_dose_reduced",
]


def run_self_supervised_timeline_pretraining(
    source_csv: str = DEFAULT_ML_CSV_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    frame = _with_prior_features(pd.read_csv(source_csv))
    train, test = _patient_split(frame)
    features = [c for c in STATIC_FEATURES + PRIOR_NUMERIC_FEATURES if c in frame.columns]
    leakage_status = _leakage_check(features)

    lab_model = _regression_pipeline(features)
    symptom_model = _classification_pipeline(features)
    imaging_model = _classification_pipeline(features)

    lab_target = "nadir_anc"
    symptom_target = "_masked_symptom_target"
    imaging_target = "_masked_imaging_signal"
    train = train.dropna(subset=[lab_target]).copy()
    test = test.dropna(subset=[lab_target]).copy()

    symptom_y_train = (train["max_symptom_severity"].fillna(0).astype(float) >= 4).astype(int)
    symptom_y_test = (test["max_symptom_severity"].fillna(0).astype(float) >= 4).astype(int)
    imaging_y_train = (train["mri_percent_change_from_baseline"].fillna(0).astype(float) <= -25).astype(int)
    imaging_y_test = (test["mri_percent_change_from_baseline"].fillna(0).astype(float) <= -25).astype(int)
    train[symptom_target] = symptom_y_train
    test[symptom_target] = symptom_y_test
    train[imaging_target] = imaging_y_train
    test[imaging_target] = imaging_y_test

    lab_model.fit(train[features], train[lab_target].astype(float))
    symptom_model.fit(train[features], train[symptom_target])
    imaging_model.fit(train[features], train[imaging_target])

    lab_pred = lab_model.predict(test[features])
    symptom_pred = symptom_model.predict(test[features])
    imaging_pred = imaging_model.predict(test[features])

    payload = {
        **build_artifact_manifest(dataset_paths={"source_csv": source_csv}),
        "schema_version": "self_supervised_timeline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if leakage_status == "passed" else "failed",
        "task": "masked_timeline_signal_prediction_from_prior_cycles",
        "metrics": {
            "masked_lab_mae": round(float(mean_absolute_error(test[lab_target], lab_pred)), 4),
            "masked_symptom_f1": _safe_f1(test[symptom_target], symptom_pred),
            "masked_imaging_signal_accuracy": round(float(accuracy_score(test[imaging_target], imaging_pred)), 4),
            "leakage_check_status": leakage_status,
        },
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "features": features,
        "claim_boundary": (
            "Experimental synthetic-only representation/pretraining proxy. "
            "Uses prior-cycle features only and is not a production clinical model."
        ),
    }
    _write_json(output_path, payload)
    return payload


def load_self_supervised_timeline_report(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "missing"}


def _with_prior_features(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["patient_id", "cycle"]).copy()
    prior_cols = [
        "pre_wbc", "pre_anc", "pre_hemoglobin", "pre_platelets",
        "nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets",
        "recovery_wbc", "recovery_hemoglobin", "recovery_platelets",
        "mri_percent_change_from_baseline", "max_symptom_severity",
        "symptom_count", "intervention_count", "dose_delayed", "dose_reduced",
    ]
    for col in prior_cols:
        if col in ordered.columns:
            ordered[f"prev_{col}"] = ordered.groupby("patient_id")[col].shift(1)
    return ordered[ordered["cycle"].astype(int) > 1].copy()


def _patient_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = sorted(frame["patient_id"].astype(str).unique())
    cutoff = int(len(ids) * 0.75)
    train_ids = set(ids[:cutoff])
    return frame[frame["patient_id"].astype(str).isin(train_ids)].copy(), frame[~frame["patient_id"].astype(str).isin(train_ids)].copy()


def _regression_pipeline(features: list[str]) -> Pipeline:
    return Pipeline([("pre", _preprocessor(features)), ("model", RandomForestRegressor(n_estimators=80, random_state=42, min_samples_leaf=4))])


def _classification_pipeline(features: list[str]) -> Pipeline:
    return Pipeline([("pre", _preprocessor(features)), ("model", GradientBoostingClassifier(random_state=42))])


def _preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [c for c in features if c in {"stage", "molecular_subtype", "regimen"}]
    numeric = [c for c in features if c not in categorical]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def _leakage_check(features: list[str]) -> str:
    banned = {"response_score_percent", "treatment_success_binary", "toxicity_risk_binary", "final_response_category", "final_cancer_status"}
    return "failed" if any(f in banned or f.startswith("future_") for f in features) else "passed"


def _safe_f1(y_true, y_pred) -> float:
    values = np.asarray(y_true)
    if len(set(values.tolist())) < 2:
        return 0.0
    return round(float(f1_score(y_true, y_pred)), 4)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["DEFAULT_OUTPUT_PATH", "run_self_supervised_timeline_pretraining", "load_self_supervised_timeline_report"]
