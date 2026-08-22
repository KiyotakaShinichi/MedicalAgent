"""Input validation, patient-level splitting, and the shared preprocessor.

The split is **patient-level, not row-level**: every row for a patient lands
wholly in train or wholly in test. Splitting rows would leak one patient's
trajectory across the boundary and inflate every downstream metric.

``_patient_split`` and ``_preprocessor`` are imported directly by eight and
five other modules respectively, so despite the underscore their behaviour is
a repository-wide contract rather than an internal detail.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.services.complete_synthetic_training.feature_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RESPONSE_REGRESSION_TARGET,
    ROW_LEVEL_TARGETS,
)


def _ensure_response_regression_columns(rows):
    rows = rows.copy()
    if RESPONSE_REGRESSION_TARGET not in rows.columns and "mri_percent_change_from_baseline" in rows.columns:
        rows[RESPONSE_REGRESSION_TARGET] = -pd.to_numeric(rows["mri_percent_change_from_baseline"], errors="coerce")
    if RESPONSE_REGRESSION_TARGET in rows.columns:
        rows[RESPONSE_REGRESSION_TARGET] = pd.to_numeric(rows[RESPONSE_REGRESSION_TARGET], errors="coerce")
        rows[RESPONSE_REGRESSION_TARGET] = (
            rows.groupby("patient_id")[RESPONSE_REGRESSION_TARGET]
            .transform(lambda series: series.ffill().bfill())
            .clip(-100, 100)
        )
    return rows

def _validate_training_frame(rows, target):
    missing = [col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["patient_id", target] if col not in rows.columns]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    if target in ROW_LEVEL_TARGETS:
        if rows[target].nunique() < 2:
            raise ValueError(f"Target {target} needs at least two classes")
        return
    patient_labels = rows.groupby("patient_id")[target].max()
    if patient_labels.nunique() < 2:
        raise ValueError(f"Target {target} needs at least two classes")

def _patient_split(rows, target, test_size, seed):
    if target in ROW_LEVEL_TARGETS:
        patient_labels = (
            rows.groupby("patient_id", as_index=False)[target]
            .mean()
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        median_rate = patient_labels[target].median()
        patient_labels["split_label"] = (patient_labels[target] >= median_rate).astype(int)
        if patient_labels["split_label"].nunique() < 2:
            patient_labels["split_label"] = (patient_labels[target] > 0).astype(int)
    else:
        patient_labels = (
            rows.groupby("patient_id", as_index=False)[target]
            .max()
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        patient_labels["split_label"] = patient_labels[target].astype(int)

    patient_labels = (
        patient_labels[["patient_id", "split_label"]]
    )
    train_patients, test_patients = train_test_split(
        patient_labels["patient_id"],
        test_size=test_size,
        random_state=seed,
        stratify=patient_labels["split_label"].astype(int),
    )
    return set(train_patients), set(test_patients)

def _preprocessor(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)
