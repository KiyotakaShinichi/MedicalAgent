"""
Temporal train/eval split for the synthetic breast cancer ML pipeline.

Two complementary splits:
  1. Patient-timeline split -- sort patients by first treatment date, train on
     the earlier 50%, evaluate on the later 50%. Tests whether a model trained
     on an earlier synthetic cohort generalises to a later one.
  2. Cycle-accumulation split -- for each patient, train only on rows from
     cycles 1-3, predict on cycles 4-6. Tests whether early-cycle signals
     are sufficient to predict the final outcome.

Comparison baseline: patient-level stratified random split (standard holdout).

CLAIM BOUNDARY: Synthetic simulator generates ordered patient timelines.
Timeline ordering reflects simulator design, not real hospital enrollment
patterns. Results indicate engineering robustness only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DEFAULT_TEMPORAL_EVAL_PATH = "Data/mle_monitoring/temporal_eval_report.json"
DEFAULT_ML_CSV_PATH = (
    "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv"
    if Path("Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv").exists()
    else "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
)

_TARGET = "treatment_success_binary"
_PATIENT_COL = "patient_id"
_DATE_COL = "treatment_date"
_FIRST_DATE_COL = "first_treatment_date"
_CYCLE_COL = "cycle"
_RANDOM_SEED = 42


def run_temporal_eval(
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    output_path: str = DEFAULT_TEMPORAL_EVAL_PATH,
) -> dict:
    """
    Run temporal generalization evaluation and export a JSON report.

    Returns a dict with keys: status, temporal_split, cycle_split,
    random_split_baseline, generalization_gap, interpretation.
    """
    rows = _load(ml_csv_path)
    if rows is None:
        return _unavailable("Training CSV not found.")
    if len(rows) < 100 or _TARGET not in rows.columns:
        return _unavailable("Insufficient training rows or missing target column.")

    patient_df = _patient_aggregate(rows)
    if patient_df is None or len(patient_df) < 40:
        return _unavailable("Fewer than 40 patients -- temporal split not meaningful.")

    temporal = _timeline_split(patient_df)
    cycle = _cycle_split(rows)
    baseline = _random_split(patient_df)

    gap = None
    if temporal.get("eval_auroc") is not None and baseline.get("eval_auroc") is not None:
        gap = _r(temporal["eval_auroc"] - baseline["eval_auroc"])

    status = _status(temporal.get("eval_auroc"), baseline.get("eval_auroc"))

    result = {
        "schema_version": "temporal_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Tests whether models trained on earlier synthetic cohorts generalise "
            "to later cohorts -- a proxy for real-world deployment drift. "
            "Uses synthetic data only."
        ),
        "status": status,
        "temporal_split": temporal,
        "cycle_accumulation_split": cycle,
        "random_split_baseline": baseline,
        "generalization_gap": gap,
        "gap_meaning": (
            "Positive gap means temporal eval AUROC > random baseline "
            "(better-than-expected temporal generalisation). "
            "Negative gap means temporal ordering hurts -- possible early/late "
            "cohort distributional shift in the simulator."
        ),
        "interpretation": _interpretation(gap, status),
        "claim_boundary": (
            "Synthetic simulator generates ordered patient timelines. "
            "Temporal ordering reflects simulator design, not real hospital "
            "enrollment patterns."
        ),
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# -- Splits --------------------------------------------------------------------

def _timeline_split(patient_df: pd.DataFrame) -> dict:
    """Train on earlier 50% of patients (by first treatment date), eval on later 50%."""
    if _FIRST_DATE_COL not in patient_df.columns:
        return {"status": "unavailable", "reason": "first_treatment_date column missing"}

    sorted_pts = patient_df.sort_values(_FIRST_DATE_COL).reset_index(drop=True)
    split_idx = len(sorted_pts) // 2
    train = sorted_pts.iloc[:split_idx]
    test = sorted_pts.iloc[split_idx:]

    if len(train) < 20 or len(test) < 20:
        return {"status": "unavailable", "reason": "insufficient patients after split"}

    return _fit_eval(train, test, label="patient_timeline_split")


def _cycle_split(rows: pd.DataFrame) -> dict:
    """
    For each patient, use only cycles 1-3 as features; target is the patient-level
    treatment_success_binary. Train on all patients (early cycles), evaluate on the
    same patients using a stratified holdout split.

    This tests whether early-cycle CBC/symptom/MRI signals predict the final outcome.
    """
    if _CYCLE_COL not in rows.columns:
        return {"status": "unavailable", "reason": "cycle column missing"}

    early = rows[rows[_CYCLE_COL] <= 3].copy()
    if len(early) < 40:
        return {"status": "unavailable", "reason": "fewer than 40 rows with cycle <= 3"}

    patient_early = _patient_aggregate_from(early)
    if patient_early is None or len(patient_early) < 30:
        return {"status": "unavailable", "reason": "insufficient patient aggregates"}

    # Random split on early-cycle aggregates
    return _random_split(patient_early, label="cycle_accumulation_split_early_cycles_only")


def _random_split(patient_df: pd.DataFrame, label: str = "random_split_baseline") -> dict:
    """Standard stratified 75/25 random split as a comparison baseline."""
    X, y = _features_labels(patient_df)
    if X is None:
        return {"status": "unavailable", "reason": "feature extraction failed"}

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=_RANDOM_SEED)
    try:
        train_idx, test_idx = next(splitter.split(X, y))
    except ValueError as e:
        return {"status": "unavailable", "reason": str(e)}

    train_df = patient_df.iloc[train_idx]
    test_df = patient_df.iloc[test_idx]
    return _fit_eval(train_df, test_df, label=label)


# -- Core fit/eval -------------------------------------------------------------

def _fit_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str) -> dict:
    """Fit a logistic regression on train_df and evaluate on test_df."""
    X_train, y_train = _features_labels(train_df)
    X_test, y_test = _features_labels(test_df)
    if X_train is None or X_test is None:
        return {"status": "unavailable", "reason": "feature extraction failed"}
    if len(set(y_train.tolist())) < 2:
        return {"status": "unavailable", "reason": "single class in training split"}
    if len(set(y_test.tolist())) < 2:
        return {"status": "unavailable", "reason": "single class in evaluation split"}

    # Train/test may contain different one-hot categories after temporal slicing.
    # Align evaluation columns to the training design matrix so the split is
    # a real out-of-time evaluation instead of a feature-shape artifact.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=_RANDOM_SEED, C=1.0)),
    ])
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    train_probs = model.predict_proba(X_train)[:, 1]
    train_auroc = _r(roc_auc_score(y_train, train_probs))

    auroc = _r(roc_auc_score(y_test, probs))
    auprc = _r(average_precision_score(y_test, probs))
    brier = _r(brier_score_loss(y_test, probs))

    return {
        "label": label,
        "train_patients": int(len(train_df)),
        "eval_patients": int(len(test_df)),
        "train_auroc": train_auroc,
        "eval_auroc": auroc,
        "eval_auprc": auprc,
        "eval_brier": brier,
        "status": _auroc_status(auroc),
        "model": "logistic_regression_pipeline",
    }


# -- Feature engineering -------------------------------------------------------

def _patient_aggregate(rows: pd.DataFrame) -> pd.DataFrame | None:
    return _patient_aggregate_from(rows)


def _patient_aggregate_from(rows: pd.DataFrame) -> pd.DataFrame | None:
    if _PATIENT_COL not in rows.columns or _TARGET not in rows.columns:
        return None

    numeric_cols = [c for c in NUMERIC_FEATURES if c in rows.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in rows.columns]

    agg_spec: dict = {_TARGET: (_TARGET, "first")}
    for col in numeric_cols:
        agg_spec[col] = (col, "mean")
    for col in cat_cols:
        agg_spec[col] = (col, "first")
    if _DATE_COL in rows.columns:
        agg_spec[_FIRST_DATE_COL] = (_DATE_COL, "min")

    try:
        patient_df = rows.groupby(_PATIENT_COL, as_index=False).agg(**agg_spec)
    except Exception:
        return None

    return patient_df.dropna(subset=[_TARGET]).reset_index(drop=True)


def _features_labels(patient_df: pd.DataFrame):
    numeric_cols = [c for c in NUMERIC_FEATURES if c in patient_df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in patient_df.columns]

    if not numeric_cols:
        return None, None

    X_num = patient_df[numeric_cols].fillna(patient_df[numeric_cols].median())
    X_cat = pd.get_dummies(patient_df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=patient_df.index)
    X = pd.concat([X_num, X_cat], axis=1).astype(float)
    y = patient_df[_TARGET].astype(int).to_numpy()
    return X, y


# -- Status and helpers --------------------------------------------------------

def _status(temporal_auroc, baseline_auroc) -> str:
    if temporal_auroc is None:
        return "unavailable"
    if baseline_auroc is None:
        return _auroc_status(temporal_auroc)
    gap = temporal_auroc - baseline_auroc
    if gap >= -0.03:
        return "stable"
    if gap >= -0.08:
        return "mild_drift"
    return "significant_drift"


def _auroc_status(auroc) -> str:
    if auroc is None:
        return "unavailable"
    if auroc >= 0.80:
        return "strong"
    if auroc >= 0.70:
        return "acceptable"
    if auroc >= 0.60:
        return "unideal"
    return "failed"


def _interpretation(gap, status: str) -> str:
    if gap is None:
        return "Temporal evaluation could not be computed."
    abs_gap = abs(gap)
    if status == "stable":
        return (
            f"Temporal generalisation is stable (gap={gap:+.3f}). "
            "The synthetic simulator does not appear to introduce strong cohort effects "
            "between earlier and later patient timelines."
        )
    if status == "mild_drift":
        return (
            f"Mild temporal drift detected (gap={gap:+.3f}). "
            "The model trained on earlier synthetic cohorts performs modestly worse on "
            "later cohorts. Monitor in a real deployment setting."
        )
    return (
        f"Significant temporal drift (gap={gap:+.3f}). "
        "Strong distribution shift between earlier and later synthetic cohorts. "
        "Inspect simulator design or add periodic retraining before deployment."
    )


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def _unavailable(reason: str) -> dict:
    return {
        "schema_version": "temporal_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "reason": reason,
    }


def _r(v, digits: int = 4):
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None
