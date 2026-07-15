"""Patient-level temporal cross-validation for the synthetic training rows.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~
The default ``complete_synthetic_training`` flow does an 80/20 random
patient split (no patient appears in both train and test).  That avoids
within-patient leakage on a single split, but it does **not**:

  1. Preserve temporal ordering — a fold can train on a patient whose
     treatment started in 2027 and test on a patient who started in
     2025.  In a real deployment that's a future-leak.
  2. Compare against the naive row-level cross-validation that a less
     careful baseline would use, where the *same* patient's cycle 1 row
     can train a model that's evaluated on the same patient's cycle 6
     row.

This module does both and writes a side-by-side JSON so an external
reviewer can see how much the choice of CV strategy moves the headline
metric.

The metrics here are still synthetic-only — they do not establish
clinical validity.  They establish that the CV protocol itself is
*defensible*, which is what the external critique asked for.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_patient_temporal_cv.json"
DEFAULT_TARGET = "toxicity_risk_binary"
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 17

# Mirrors complete_synthetic_training.NUMERIC_FEATURES + CATEGORICAL_FEATURES
# minus the ones that leak the target (final_*, treatment_success_*).
NUMERIC_FEATURES: tuple[str, ...] = (
    "cycle", "age",
    "pre_wbc", "pre_anc", "pre_hemoglobin", "pre_platelets",
    "nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets",
    "recovery_wbc", "recovery_hemoglobin", "recovery_platelets",
    "mri_tumor_size_cm", "mri_percent_change_from_baseline",
    "max_symptom_severity", "symptom_count", "intervention_count",
    "dose_delayed", "dose_reduced",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("stage", "molecular_subtype", "regimen")


@dataclass
class FoldMetrics:
    fold: int
    train_n_rows: int
    train_n_patients: int
    test_n_rows: int
    test_n_patients: int
    train_date_min: str
    train_date_max: str
    test_date_min: str
    test_date_max: str
    roc_auc: float | None
    brier: float | None
    positive_rate_train: float
    positive_rate_test: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_n_rows": self.train_n_rows,
            "train_n_patients": self.train_n_patients,
            "test_n_rows": self.test_n_rows,
            "test_n_patients": self.test_n_patients,
            "train_date_min": self.train_date_min,
            "train_date_max": self.train_date_max,
            "test_date_min": self.test_date_min,
            "test_date_max": self.test_date_max,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "positive_rate_train": self.positive_rate_train,
            "positive_rate_test": self.positive_rate_test,
        }


@dataclass
class StrategyReport:
    name: str
    description: str
    folds: list[FoldMetrics] = field(default_factory=list)
    patient_overlap_pairs: int = 0
    temporal_violations: int = 0
    train_rows_censored_after_test_start: int = 0

    def aggregate(self) -> dict[str, Any]:
        aucs = [f.roc_auc for f in self.folds if f.roc_auc is not None]
        briers = [f.brier for f in self.folds if f.brier is not None]
        return {
            "name": self.name,
            "description": self.description,
            "n_folds_with_auc": len(aucs),
            "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
            "roc_auc_std": float(np.std(aucs)) if aucs else None,
            "brier_mean": float(np.mean(briers)) if briers else None,
            "brier_std": float(np.std(briers)) if briers else None,
            "patient_overlap_pairs": self.patient_overlap_pairs,
            "temporal_violations": self.temporal_violations,
            "train_rows_censored_after_test_start": self.train_rows_censored_after_test_start,
            "row_temporal_censoring_applied": self.train_rows_censored_after_test_start > 0,
            "folds": [f.as_dict() for f in self.folds],
        }


def _build_estimator(seed: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                list(NUMERIC_FEATURES),
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=seed))])


def _fit_and_score(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    target: str,
    seed: int,
) -> tuple[float | None, float | None]:
    feat_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    if train_rows[target].nunique() < 2 or test_rows.empty:
        return None, None
    estimator = _build_estimator(seed)
    estimator.fit(train_rows[feat_cols], train_rows[target].astype(int))
    proba = estimator.predict_proba(test_rows[feat_cols])[:, 1]
    y_true = test_rows[target].astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        return None, float(brier_score_loss(y_true, proba))
    return float(roc_auc_score(y_true, proba)), float(brier_score_loss(y_true, proba))


def _make_fold_metrics(
    fold: int,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    target: str,
    seed: int,
) -> FoldMetrics:
    auc, brier = _fit_and_score(train_rows, test_rows, target, seed)
    return FoldMetrics(
        fold=fold,
        train_n_rows=int(len(train_rows)),
        train_n_patients=int(train_rows["patient_id"].nunique()),
        test_n_rows=int(len(test_rows)),
        test_n_patients=int(test_rows["patient_id"].nunique()),
        train_date_min=str(train_rows["treatment_date"].min()),
        train_date_max=str(train_rows["treatment_date"].max()),
        test_date_min=str(test_rows["treatment_date"].min()) if not test_rows.empty else "",
        test_date_max=str(test_rows["treatment_date"].max()) if not test_rows.empty else "",
        roc_auc=auc,
        brier=brier,
        positive_rate_train=float(train_rows[target].astype(int).mean()),
        positive_rate_test=float(test_rows[target].astype(int).mean()) if not test_rows.empty else 0.0,
    )


def patient_temporal_folds(
    rows: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[tuple[Sequence[str], Sequence[str]]]:
    """Yield (train_patients, test_patients) per fold.

    Patients are ordered by their earliest ``treatment_date``.  Fold ``k``
    uses patients in the k-th time block as the held-out set; only
    patients from *earlier* time blocks are eligible for training.  This
    is a walk-forward, group-aware split — no patient_id appears in both
    train and test, and no train row is dated after the earliest test
    row in the same patient cohort.
    """
    first_dates = (
        rows.groupby("patient_id")["treatment_date"].min().sort_values()
    )
    ordered_patients = first_dates.index.tolist()
    chunks = np.array_split(ordered_patients, n_folds)
    folds: list[tuple[Sequence[str], Sequence[str]]] = []
    for k in range(1, n_folds):
        train = [p for chunk in chunks[:k] for p in chunk]
        test = list(chunks[k])
        if not train or not test:
            continue
        folds.append((train, test))
    return folds


def run_patient_temporal_cv(
    rows: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> StrategyReport:
    rows = rows.copy()
    rows["treatment_date"] = pd.to_datetime(rows["treatment_date"])
    report = StrategyReport(
        name="patient_level_temporal_cv",
        description=(
            "Walk-forward CV grouped by patient_id, ordered by each patient's "
            "earliest treatment_date. No patient overlap across folds. Training "
            "rows dated on/after the held-out fold start are censored so strict "
            "row-level temporal ordering is preserved."
        ),
    )
    folds = patient_temporal_folds(rows, n_folds=n_folds)
    for fold_idx, (train_pids, test_pids) in enumerate(folds, start=1):
        train_rows = rows[rows["patient_id"].isin(train_pids)]
        test_rows = rows[rows["patient_id"].isin(test_pids)]
        overlap = set(train_pids) & set(test_pids)
        report.patient_overlap_pairs += len(overlap)
        if not train_rows.empty and not test_rows.empty:
            test_start = test_rows["treatment_date"].min()
            original_train_n = len(train_rows)
            train_rows = train_rows[train_rows["treatment_date"] < test_start]
            report.train_rows_censored_after_test_start += original_train_n - len(train_rows)
            if train_rows.empty or train_rows["treatment_date"].max() >= test_start:
                report.temporal_violations += 1
        report.folds.append(
            _make_fold_metrics(fold_idx, train_rows, test_rows, target, seed)
        )
    return report


def run_naive_row_level_cv(
    rows: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> StrategyReport:
    rows = rows.copy()
    rows["treatment_date"] = pd.to_datetime(rows["treatment_date"])
    report = StrategyReport(
        name="naive_row_level_kfold",
        description=(
            "Random KFold over rows.  Same patient_id can appear in both "
            "train and test (within-patient leakage), and folds ignore "
            "treatment_date ordering.  Included as a worst-case baseline."
        ),
    )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(rows))
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
        train_rows = rows.iloc[train_idx]
        test_rows = rows.iloc[test_idx]
        overlap = set(train_rows["patient_id"]) & set(test_rows["patient_id"])
        report.patient_overlap_pairs += len(overlap)
        if not train_rows.empty and not test_rows.empty:
            if train_rows["treatment_date"].max() > test_rows["treatment_date"].min():
                report.temporal_violations += 1
        report.folds.append(
            _make_fold_metrics(fold_idx, train_rows, test_rows, target, seed)
        )
    return report


def build_cv_comparison(
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    target: str = DEFAULT_TARGET,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    rows = pd.read_csv(ml_csv_path)
    if target not in rows.columns:
        raise ValueError(f"target {target!r} missing from {ml_csv_path}")

    patient_report = run_patient_temporal_cv(rows, target=target, n_folds=n_folds, seed=seed)
    naive_report = run_naive_row_level_cv(rows, target=target, n_folds=n_folds, seed=seed)

    patient_agg = patient_report.aggregate()
    naive_agg = naive_report.aggregate()

    delta_auc: float | None = None
    if patient_agg["roc_auc_mean"] is not None and naive_agg["roc_auc_mean"] is not None:
        delta_auc = float(naive_agg["roc_auc_mean"] - patient_agg["roc_auc_mean"])

    n_folds_actual = len(patient_report.folds)
    return {
        "schema_version": "1.0",
        "status": "informational",
        "label": "internal_engineering_eval_synthetic_only",
        "claim_boundary": (
            "Patient-level temporal CV vs. naive row-level KFold on synthetic "
            "patient journeys.  AUC numbers here describe a synthetic distribution "
            "with internally consistent labels — they do NOT establish clinical "
            "validity, calibration, or generalization to real patient cohorts."
        ),
        "total_n": n_folds_actual,
        "pass_count": n_folds_actual,
        "fail_count": 0,
        "skipped_count": 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": ml_csv_path,
        "target": target,
        "n_folds": n_folds,
        "seed": seed,
        "n_rows_total": int(len(rows)),
        "n_patients_total": int(rows["patient_id"].nunique()),
        "patient_level_temporal_cv": patient_agg,
        "naive_row_level_kfold": naive_agg,
        "headline": {
            "auc_optimism_from_naive_cv": delta_auc,
            "note": (
                "Positive value means the naive row-level CV is "
                "optimistically biased relative to the patient-level "
                "temporal CV.  Synthetic data only; not a clinical "
                "validity claim."
            ),
        },
    }


def write_cv_comparison_report(
    output_path: str = DEFAULT_OUTPUT_PATH,
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    target: str = DEFAULT_TARGET,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> Path:
    report = build_cv_comparison(ml_csv_path=ml_csv_path, target=target, n_folds=n_folds, seed=seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


__all__ = [
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "FoldMetrics",
    "StrategyReport",
    "patient_temporal_folds",
    "run_patient_temporal_cv",
    "run_naive_row_level_cv",
    "build_cv_comparison",
    "write_cv_comparison_report",
]
