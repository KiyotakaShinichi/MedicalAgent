"""Modality-dropout retraining for the response-score regression head.

Parallel to ``modality_dropout_training`` (which retrained the
classification champion), this service retrains the **regression** head
on augmented rows where random modality groups are masked out.  The
classifier work shipped a +8.3pp accuracy improvement on `no_imaging`;
this is the same intervention for the continuous response-strength
score: teach the regressor that simultaneous-median patterns in
correlated features (all of cbc_nadir, or both imaging columns) are a
missingness signature, so the model becomes intrinsically more robust
on partial-evidence rows rather than relying solely on the inference-
time confidence-modifier shrinkage.

Artifacts
---------
  - ``Data/complete_synthetic_training/modality_robust_response_score_percent.joblib``
  - ``Data/evals/models/latest_modality_robust_regression_training.json``

Engineering provenance only.  A passing artifact means the augmented
training did not collapse and produced sensible test-split metrics; it
does not establish clinical regression validity.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _patient_split,
    _preprocessor,
)
from backend.services.modality_dropout_training import _augment_with_modality_dropout


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_MODEL_PATH = (
    "Data/complete_synthetic_training/modality_robust_response_score_percent.joblib"
)
DEFAULT_METADATA_PATH = (
    "Data/evals/models/latest_modality_robust_regression_training.json"
)
DEFAULT_TARGET = "response_score_percent"

# Same augmentation knobs as the classification trainer for consistency.
DEFAULT_N_AUG_PER_ROW = 3
DEFAULT_P_DROP_PER_MODALITY = 0.30
DEFAULT_MAX_SIMULTANEOUS_DROPOUTS = 3
DEFAULT_PROTECTED_MODALITIES = ("demographics",)


def train_modality_robust_regressor(
    *,
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    model_output_path: str = DEFAULT_MODEL_PATH,
    metadata_output_path: str = DEFAULT_METADATA_PATH,
    target: str = DEFAULT_TARGET,
    n_aug_per_row: int = DEFAULT_N_AUG_PER_ROW,
    p_drop_per_modality: float = DEFAULT_P_DROP_PER_MODALITY,
    max_simultaneous_dropouts: int = DEFAULT_MAX_SIMULTANEOUS_DROPOUTS,
    protected_modalities: tuple[str, ...] = DEFAULT_PROTECTED_MODALITIES,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Train + persist the modality-robust regressor.  Returns metadata."""
    rng = random.Random(seed)
    rows = pd.read_csv(ml_csv_path)
    if target not in rows.columns:
        raise ValueError(f"Target column '{target}' missing from {ml_csv_path}")

    # Use the standard patient-aware split (stratified on the classification
    # target) so this trainer's test patients match the classifier's — the
    # comparison eval can then compare like-for-like.
    train_patients, test_patients = _patient_split(rows, "treatment_success_binary", test_size, seed)
    train_rows = rows[rows["patient_id"].isin(train_patients)].dropna(subset=[target]).copy()
    test_rows = rows[rows["patient_id"].isin(test_patients)].dropna(subset=[target]).copy()

    augmented_train, dropout_stats = _augment_with_modality_dropout(
        train_rows,
        rng=rng,
        n_aug_per_row=n_aug_per_row,
        p_drop_per_modality=p_drop_per_modality,
        max_simultaneous_dropouts=max_simultaneous_dropouts,
        protected_modalities=protected_modalities,
    )

    model = Pipeline([
        ("preprocess", _preprocessor(scale_numeric=False)),
        ("regressor", GradientBoostingRegressor(
            random_state=seed,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
        )),
    ])
    X_train = augmented_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = augmented_train[target].astype(float)
    model.fit(X_train, y_train)

    # Evaluate on the UNAUGMENTED test split — the honest baseline.
    X_test = test_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_rows[target].astype(float).to_numpy()
    test_preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, test_preds)),
        "rmse": float(np.sqrt(np.mean((y_test - test_preds) ** 2))),
        "mean_prediction": float(np.mean(test_preds)),
        "test_rows": int(len(test_rows)),
    }

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    metadata: dict[str, Any] = {
        "schema_version": "modality_robust_regression_training_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _overall_status(metrics["mae"]),
        "target": target,
        "ml_csv_path": ml_csv_path,
        "model_path": model_output_path,
        "seed": seed,
        "training_config": {
            "n_aug_per_row": n_aug_per_row,
            "p_drop_per_modality": p_drop_per_modality,
            "max_simultaneous_dropouts": max_simultaneous_dropouts,
            "protected_modalities": list(protected_modalities),
            "test_size": test_size,
        },
        "patient_split": {
            "train_patient_count": len(train_patients),
            "test_patient_count": len(test_patients),
            "split_disjoint": len(train_patients & test_patients) == 0,
        },
        "augmentation_stats": dropout_stats,
        "test_metrics": metrics,
        "claim_boundary": (
            "Engineering artifact only.  Modality-dropout augmentation "
            "teaches the regressor to handle synthetic missingness; it "
            "does not establish clinical regression robustness."
        ),
    }
    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _overall_status(mae: float) -> str:
    """Synthetic response_score_percent is in 0-100. MAE under 15 is the
    'training didn't collapse' floor we care about for a CI gate."""
    if mae < 10:
        return "strong"
    if mae < 15:
        return "acceptable"
    return "needs_attention"


def load_modality_robust_regression_metadata(
    path: str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "modality_robust_regression_training_v1",
            "status": "missing",
            "message": (
                "Modality-robust regression training has not been run yet. "
                "Execute `scripts/run_modality_dropout_regression_training.py`."
            ),
            "training_config": {},
            "test_metrics": {},
            "augmentation_stats": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_METADATA_PATH",
    "DEFAULT_TARGET",
    "load_modality_robust_regression_metadata",
    "train_modality_robust_regressor",
]
