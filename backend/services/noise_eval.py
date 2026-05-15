"""
High-noise synthetic evaluation for the breast cancer ML pipeline.

Applies five realistic noise perturbations to the held-out test set and
measures AUROC/Brier/sensitivity degradation vs the clean baseline. This tests
model robustness to the kinds of data quality issues that appear in real EHR
deployments without requiring real clinical data.

Noise modes applied:
  1. lab_missingness    -- randomly null 20 % of CBC numeric values
  2. lab_jitter         -- Gaussian noise (mean=0, sd=0.5*feature-std) on all lab values
  3. unit_entry_error   -- multiply 5 % of WBC readings by 10 (K/uL vs /uL mix-up)
  4. site_batch_effect  -- systematic +0.5*std shift to all labs for a random 25 % cohort
  5. contradictory_sxs  -- set symptom_count=0 where intervention_count >= 3 (contradictory record)

CLAIM BOUNDARY: Synthetic perturbations applied to a synthetic dataset.
Results quantify model brittleness to simulator-level data quality issues,
not real-world deployment robustness.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DEFAULT_NOISE_EVAL_PATH = "Data/mle_monitoring/noise_eval_report.json"
DEFAULT_MODEL_PATH = (
    "Data/complete_synthetic_training_realism_v2/logistic_regression_treatment_success_binary.joblib"
    if Path("Data/complete_synthetic_training_realism_v2/logistic_regression_treatment_success_binary.joblib").exists()
    else "Data/complete_synthetic_training/logistic_regression_treatment_success_binary.joblib"
)
DEFAULT_ML_CSV_PATH = (
    "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv"
    if Path("Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv").exists()
    else "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
)
DEFAULT_PREDICTIONS_CSV_PATH = (
    "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_predictions.csv"
    if Path("Data/complete_synthetic_training_realism_v2/complete_synthetic_model_predictions.csv").exists()
    else "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
)

_TARGET = "treatment_success_binary"
_PATIENT_COL = "patient_id"
_RANDOM_SEED = 42

_LAB_COLS = [
    "pre_wbc", "pre_anc", "pre_hemoglobin", "pre_platelets",
    "nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets",
    "recovery_wbc", "recovery_hemoglobin", "recovery_platelets",
]
_WBC_COLS = ["pre_wbc", "nadir_wbc", "recovery_wbc"]
_SYMPTOM_COLS = ["symptom_count", "max_symptom_severity"]
_INTERVENTION_COL = "intervention_count"


def run_noise_eval(
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    predictions_csv_path: str = DEFAULT_PREDICTIONS_CSV_PATH,
    output_path: str = DEFAULT_NOISE_EVAL_PATH,
) -> dict:
    """
    Load saved LR model, apply 5 noise modes to held-out rows, report degradation.

    Returns a dict with keys: status, clean_baseline, noise_results, summary.
    """
    rows = _load_csv(ml_csv_path)
    if rows is None:
        return _unavailable("Training CSV not found.")

    model = _load_model(model_path)
    if model is None:
        return _unavailable("Saved LR model not found. Run training first.")

    # Identify test patient IDs from predictions CSV (these were the holdout)
    test_patients = _test_patient_ids(predictions_csv_path)
    if test_patients is None:
        # Fall back: random 25 % holdout
        rng = np.random.default_rng(_RANDOM_SEED)
        all_patients = rows[_PATIENT_COL].unique()
        test_patients = set(rng.choice(all_patients, size=max(1, len(all_patients) // 4), replace=False).tolist())

    test_rows = rows[rows[_PATIENT_COL].isin(test_patients)].copy()
    train_rows = rows[~rows[_PATIENT_COL].isin(test_patients)].copy()

    if len(test_rows) < 10:
        return _unavailable("Fewer than 10 test rows available.")

    # Clean baseline
    clean_metrics = _score(model, test_rows)
    if clean_metrics is None:
        return _unavailable("Could not score model on clean test set.")

    noise_results = {}
    for mode, fn in _NOISE_MODES:
        noisy = fn(test_rows.copy(), train_rows)
        m = _score(model, noisy)
        if m is not None:
            noise_results[mode] = {
                **m,
                "auroc_drop": _r((clean_metrics["auroc"] or 0) - (m["auroc"] or 0)),
                "brier_increase": _r((m["brier"] or 0) - (clean_metrics["brier"] or 0)),
                "sensitivity_drop": _r(
                    (clean_metrics["sensitivity"] or 0) - (m["sensitivity"] or 0)
                ),
            }
        else:
            noise_results[mode] = {"status": "unavailable"}

    status = _aggregate_status(noise_results, clean_metrics)
    summary = _summary(clean_metrics, noise_results)
    max_auroc_drop = _max_auroc_drop(summary)

    result = {
        "schema_version": "noise_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Measures AUROC/Brier/sensitivity degradation under five synthetic noise "
            "modes. Quantifies model brittleness to EHR data quality issues."
        ),
        "status": status,
        "model": "logistic_regression",
        "test_patients": int(len(test_patients)),
        "test_rows": int(len(test_rows)),
        "clean_baseline": clean_metrics,
        "noise_modes": _NOISE_MODE_DESCRIPTIONS,
        "noise_results": noise_results,
        "summary": summary,
        "max_auroc_drop": max_auroc_drop,
        "interpretation": _interpretation(status, max_auroc_drop),
        "claim_boundary": (
            "Synthetic perturbations on synthetic data. Real EHR data quality "
            "issues may differ in type, frequency, and correlation structure."
        ),
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# -- Noise modes ---------------------------------------------------------------

def _noise_lab_missingness(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Randomly null 20 % of CBC lab values."""
    rng = np.random.default_rng(_RANDOM_SEED + 1)
    lab_cols = [c for c in _LAB_COLS if c in df.columns]
    for col in lab_cols:
        mask = rng.random(len(df)) < 0.20
        df.loc[mask, col] = np.nan
    return df


def _noise_lab_jitter(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Add Gaussian noise (sd = 0.5 * training std) to all numeric lab values."""
    rng = np.random.default_rng(_RANDOM_SEED + 2)
    lab_cols = [c for c in _LAB_COLS if c in df.columns and c in train.columns]
    for col in lab_cols:
        std = float(train[col].std(ddof=1) or 1.0)
        noise = rng.normal(0, 0.5 * std, size=len(df))
        df[col] = df[col] + noise
        df[col] = df[col].clip(lower=0)
    return df


def _noise_unit_entry_error(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Multiply 5 % of WBC readings by 10 (K/uL vs /uL transcription error)."""
    rng = np.random.default_rng(_RANDOM_SEED + 3)
    wbc_cols = [c for c in _WBC_COLS if c in df.columns]
    for col in wbc_cols:
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, col] = df.loc[mask, col] * 10.0
    return df


def _noise_site_batch_effect(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Add +0.5*std systematic shift to all labs for a random 25 % cohort."""
    rng = np.random.default_rng(_RANDOM_SEED + 4)
    affected_idx = rng.choice(df.index, size=max(1, len(df) // 4), replace=False)
    lab_cols = [c for c in _LAB_COLS if c in df.columns and c in train.columns]
    for col in lab_cols:
        std = float(train[col].std(ddof=1) or 1.0)
        df.loc[affected_idx, col] = df.loc[affected_idx, col] + 0.5 * std
    return df


def _noise_contradictory_symptoms(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Set symptom_count=0 and max_symptom_severity=0 where intervention_count >= 3."""
    if _INTERVENTION_COL not in df.columns:
        return df
    mask = df[_INTERVENTION_COL] >= 3
    for col in _SYMPTOM_COLS:
        if col in df.columns:
            df.loc[mask, col] = 0
    return df


_NOISE_MODES = [
    ("lab_missingness", _noise_lab_missingness),
    ("lab_jitter", _noise_lab_jitter),
    ("unit_entry_error", _noise_unit_entry_error),
    ("site_batch_effect", _noise_site_batch_effect),
    ("contradictory_symptoms", _noise_contradictory_symptoms),
]

_NOISE_MODE_DESCRIPTIONS = {
    "lab_missingness": "20 % of CBC lab values randomly set to null (filled with training median at inference).",
    "lab_jitter": "Gaussian noise (sd=0.5*training-std) added to all numeric lab values.",
    "unit_entry_error": "5 % of WBC readings multiplied by 10 (K/uL vs /uL transcription error).",
    "site_batch_effect": "Systematic +0.5*std shift on all labs for a random 25 % patient cohort.",
    "contradictory_symptoms": "symptom_count and severity zeroed where intervention_count >= 3.",
}


# -- Scoring -------------------------------------------------------------------

def _score(model, df: pd.DataFrame) -> dict | None:
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    if not numeric_cols or _TARGET not in df.columns:
        return None

    # The saved classifier is a sklearn Pipeline with a ColumnTransformer.
    # Pass the raw feature frame through that same inference path so the
    # perturbation test measures data-quality brittleness, not feature schema
    # mismatch from manually one-hot encoding at eval time.
    feature_cols = numeric_cols + cat_cols
    X = df[feature_cols].copy()
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna("unknown")

    labels = df[_TARGET].fillna(0).astype(int).to_numpy()

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        return None

    if len(set(labels.tolist())) < 2:
        return None

    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else None

    return {
        "auroc": _r(roc_auc_score(labels, probs)),
        "auprc": _r(average_precision_score(labels, probs)),
        "brier": _r(brier_score_loss(labels, probs)),
        "sensitivity": _r(sensitivity),
        "n": int(len(df)),
    }


# -- Helpers -------------------------------------------------------------------

def _test_patient_ids(predictions_csv_path: str) -> set | None:
    p = Path(predictions_csv_path)
    if not p.exists():
        return None
    preds = pd.read_csv(p)
    if _PATIENT_COL not in preds.columns:
        return None
    return set(preds[_PATIENT_COL].unique().tolist())


def _load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def _load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def _aggregate_status(noise_results: dict, clean: dict) -> str:
    drops = [
        v.get("auroc_drop") for v in noise_results.values()
        if isinstance(v.get("auroc_drop"), (int, float))
    ]
    if not drops:
        return "unavailable"
    max_drop = max(drops)
    if max_drop <= 0.05:
        return "robust"
    if max_drop <= 0.12:
        return "mild_degradation"
    return "significant_degradation"


def _summary(clean: dict, noise_results: dict) -> list:
    rows = []
    for mode, res in noise_results.items():
        drop = res.get("auroc_drop")
        rows.append({
            "noise_mode": mode,
            "clean_auroc": clean.get("auroc"),
            "noisy_auroc": res.get("auroc"),
            "auroc_drop": drop,
            "brier_increase": res.get("brier_increase"),
            "sensitivity_drop": res.get("sensitivity_drop"),
            "severity": (
                "low" if drop is not None and drop <= 0.05
                else "medium" if drop is not None and drop <= 0.12
                else "high"
            ),
        })
    return sorted(rows, key=lambda r: -(r.get("auroc_drop") or 0))


def _max_auroc_drop(summary: list) -> float | None:
    drops = [row.get("auroc_drop") for row in summary if row.get("auroc_drop") is not None]
    return _r(max(drops)) if drops else None


def _interpretation(status: str, max_auroc_drop) -> str:
    if max_auroc_drop is None:
        return "Noise robustness could not be computed."
    if status == "robust":
        return (
            f"Robust to the synthetic perturbation set (max AUROC drop={max_auroc_drop:.4f}). "
            "Continue to frame this as simulator stress testing, not real-world EHR validation."
        )
    if status == "mild_degradation":
        return (
            f"Mild degradation under synthetic noise (max AUROC drop={max_auroc_drop:.4f}). "
            "Inspect the worst perturbation and consider stronger imputation or clipping."
        )
    return (
        f"Significant degradation under synthetic noise (max AUROC drop={max_auroc_drop:.4f}). "
        "Do not promote probability claims until robustness improves."
    )


def _unavailable(reason: str) -> dict:
    return {
        "schema_version": "noise_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "reason": reason,
    }


def _r(v, digits: int = 4):
    try:
        f = float(v)
        return round(f, digits)
    except (TypeError, ValueError):
        return None
