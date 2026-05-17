"""Modality-dropout retraining for the synthetic treatment-response classifier.

What this does
--------------
The inference-time abstention layer refuses to score rows that lack key
modalities — that's the correct safety behavior, but it doesn't make the
*model itself* better at handling missing data.  This module retrains the
gradient-boosting champion on **augmented** training rows where random
modality groups have been masked out, so the model learns the structure of
its own missingness.

The augmentation is principled: per training row we generate `n_aug` copies.
Each copy randomly drops 0–`max_simultaneous_dropouts` modality groups
(Bernoulli with `p_drop_per_modality` per group), with `demographics` always
preserved because the abstention layer requires it to even consider scoring.

The downstream preprocessor uses median imputation for numerics and OHE
with `handle_unknown="ignore"` for categoricals, so masked rows hit the
imputer (numerics) or the all-zero OHE bucket (categoricals).  The model
learns that simultaneous-median patterns in correlated features (all of
cbc_nadir, or both imaging columns) are a missingness signature and adjusts
its outputs accordingly.

Artifact: `Data/complete_synthetic_training/modality_robust_treatment_success_binary.joblib`
Metadata: `Data/evals/models/latest_modality_robust_training.json`
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _patient_split,
    _preprocessor,
)
from backend.services.evidence_sufficiency import MODALITY_GROUPS


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_MODEL_PATH = (
    "Data/complete_synthetic_training/modality_robust_treatment_success_binary.joblib"
)
DEFAULT_METADATA_PATH = "Data/evals/models/latest_modality_robust_training.json"
DEFAULT_TARGET = "treatment_success_binary"

# Augmentation knobs — tuned so a meaningful fraction of training rows
# carry a missing-modality pattern, but most rows still keep the full
# feature set so the model doesn't forget how full-evidence rows look.
DEFAULT_N_AUG_PER_ROW = 3                    # 1 original + 2 augmented copies → 3x
DEFAULT_P_DROP_PER_MODALITY = 0.30
DEFAULT_MAX_SIMULTANEOUS_DROPOUTS = 3
DEFAULT_PROTECTED_MODALITIES = ("demographics",)


def train_modality_robust_classifier(
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
    """Train and persist the modality-robust classifier.

    Returns the training metadata dict (also written to disk).
    """
    rng = random.Random(seed)
    rows = pd.read_csv(ml_csv_path)
    if target not in rows.columns:
        raise ValueError(f"Target column '{target}' missing from {ml_csv_path}")

    train_patients, test_patients = _patient_split(rows, target, test_size, seed)
    train_rows = rows[rows["patient_id"].isin(train_patients)].copy()
    test_rows = rows[rows["patient_id"].isin(test_patients)].copy()

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
        ("classifier", GradientBoostingClassifier(random_state=seed)),
    ])
    X_train = augmented_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = augmented_train[target].astype(int)
    model.fit(X_train, y_train)

    # Evaluate on the *unaugmented* test split — that is the honest baseline
    # number.  The augmented training distribution is for learning robustness;
    # the test split should still be the original synthetic data.
    X_test = test_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_rows[target].astype(int)
    test_probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "brier": float(brier_score_loss(y_test, test_probs)),
        "test_rows": int(len(test_rows)),
        "test_positive_rate": float(y_test.mean()),
    }

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    metadata: dict[str, Any] = {
        "schema_version": "modality_robust_training_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
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
            "Engineering artifact only.  Modality-dropout augmentation teaches "
            "the model to handle synthetic missingness patterns; it does not "
            "establish clinical robustness on real-world data."
        ),
    }
    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


# ─── Augmentation ────────────────────────────────────────────────────────────


def _augment_with_modality_dropout(
    train_rows: pd.DataFrame,
    *,
    rng: random.Random,
    n_aug_per_row: int,
    p_drop_per_modality: float,
    max_simultaneous_dropouts: int,
    protected_modalities: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Produce augmented training rows.  Returns the new frame + a stats dict
    summarising how many copies were generated and how many modalities were
    dropped on average per copy.

    Each input row contributes ``1 + (n_aug_per_row - 1)`` rows to the output:
    the unmodified original plus N-1 augmented copies.  The original is
    always retained so the model never forgets full-evidence rows.
    """
    if n_aug_per_row < 1:
        raise ValueError("n_aug_per_row must be >= 1 (1 means no augmentation)")

    droppable = [g for g in MODALITY_GROUPS if g not in protected_modalities]
    numeric_set = frozenset(NUMERIC_FEATURES)
    categorical_set = frozenset(CATEGORICAL_FEATURES)

    augmented_frames: list[pd.DataFrame] = [train_rows.copy()]
    dropout_counts_by_modality: dict[str, int] = {g: 0 for g in droppable}
    total_augmented_rows = 0
    total_dropouts_applied = 0

    for _ in range(max(0, n_aug_per_row - 1)):
        copy = train_rows.copy()
        # Per row, choose which modalities to drop.
        per_row_dropouts = _draw_dropouts(
            n=len(copy),
            droppable=droppable,
            p=p_drop_per_modality,
            max_simultaneous=max_simultaneous_dropouts,
            rng=rng,
        )
        for idx, modalities in enumerate(per_row_dropouts):
            for modality in modalities:
                dropout_counts_by_modality[modality] += 1
                total_dropouts_applied += 1
                for column in MODALITY_GROUPS[modality]:
                    if column in numeric_set:
                        copy.iat[idx, copy.columns.get_loc(column)] = np.nan
                    elif column in categorical_set:
                        copy.iat[idx, copy.columns.get_loc(column)] = ""
                    # Non-feature columns (label, patient_id) are untouched.
        augmented_frames.append(copy)
        total_augmented_rows += len(copy)

    out = pd.concat(augmented_frames, ignore_index=True)
    stats = {
        "input_rows": int(len(train_rows)),
        "output_rows": int(len(out)),
        "augmented_rows_added": int(total_augmented_rows),
        "total_dropouts_applied": int(total_dropouts_applied),
        "mean_dropouts_per_augmented_row": (
            round(total_dropouts_applied / max(1, total_augmented_rows), 4)
        ),
        "dropouts_by_modality": dropout_counts_by_modality,
        "protected_modalities": list(protected_modalities),
        "droppable_modalities": droppable,
    }
    return out, stats


def _draw_dropouts(
    *,
    n: int,
    droppable: list[str],
    p: float,
    max_simultaneous: int,
    rng: random.Random,
) -> list[list[str]]:
    """For each of n rows, return the list of modality groups to drop."""
    out: list[list[str]] = []
    for _ in range(n):
        chosen = [g for g in droppable if rng.random() < p]
        if len(chosen) > max_simultaneous:
            rng.shuffle(chosen)
            chosen = chosen[:max_simultaneous]
        out.append(chosen)
    return out


def load_modality_robust_training_metadata(
    path: str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Read the cached training metadata.  Returns a 'missing' shell when
    the training has never been run."""
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "modality_robust_training_v1",
            "status": "missing",
            "message": (
                "Modality-robust training has not been run yet.  Execute "
                "`scripts/run_modality_dropout_training.py` to produce the "
                "model + metadata artifact."
            ),
            "training_config": {},
            "augmentation_stats": {},
            "test_metrics": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_METADATA_PATH",
    "DEFAULT_TARGET",
    "load_modality_robust_training_metadata",
    "train_modality_robust_classifier",
]
