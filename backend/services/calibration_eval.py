"""
Calibration comparison: raw, isotonic, Platt scaling, and temperature scaling.

Exports a reliability-diagram JSON and recommends the safest probability source
for the current synthetic champion model.

CLAIM BOUNDARY: Calibration fitted on synthetic holdout data. Probability
reliability claims require real-world validation before clinical use.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_CALIBRATION_EVAL_PATH = "Data/mle_monitoring/calibration_eval_report.json"
DEFAULT_PREDICTIONS_CSV_PATH = (
    "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_predictions.csv"
    if Path("Data/complete_synthetic_training_realism_v2/complete_synthetic_model_predictions.csv").exists()
    else "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
)
DEFAULT_METRICS_JSON_PATH = (
    "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_metrics.json"
    if Path("Data/complete_synthetic_training_realism_v2/complete_synthetic_model_metrics.json").exists()
    else "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
)

_TARGET = "actual_label"
_RANDOM_SEED = 42
_BINS = 10
_TEMP_GRID = list(np.linspace(0.3, 4.0, 38))


def run_calibration_eval(
    predictions_csv_path: str = DEFAULT_PREDICTIONS_CSV_PATH,
    metrics_json_path: str = DEFAULT_METRICS_JSON_PATH,
    output_path: str = DEFAULT_CALIBRATION_EVAL_PATH,
) -> dict:
    """
    Compare calibration methods and export a reliability-diagram JSON.

    Returns dict with keys: status, raw, isotonic, platt, temperature,
    recommended_source, reliability_diagram, bins.
    """
    preds, best_model = _load(predictions_csv_path, metrics_json_path)
    if preds is None:
        return _unavailable("Predictions CSV or metrics JSON not found.")

    labels = preds[_TARGET].astype(int).to_numpy()
    raw_col = f"{best_model}_probability"
    cal_col = f"{best_model}_calibrated_probability"

    if raw_col not in preds.columns:
        return _unavailable(f"Raw probability column '{raw_col}' not found.")

    raw_probs = preds[raw_col].astype(float).to_numpy()

    # Split into calibration (50 %) and validation (50 %) to evaluate each method
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=_RANDOM_SEED)
    try:
        cal_idx, val_idx = next(splitter.split(raw_probs.reshape(-1, 1), labels))
    except ValueError as e:
        return _unavailable(f"Could not create calibration split: {e}")

    cal_probs = raw_probs[cal_idx]
    val_probs = raw_probs[val_idx]
    cal_labels = labels[cal_idx]
    val_labels = labels[val_idx]

    # -- Raw (no calibration)
    raw_metrics = _metrics(val_labels, val_probs, "raw")

    # -- Isotonic regression
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(cal_probs, cal_labels)
    iso_val = np.clip(iso_model.transform(val_probs), 0.0, 1.0)
    iso_metrics = _metrics(val_labels, iso_val, "isotonic_regression")

    # -- Platt scaling (logistic regression on probabilities)
    platt_model = LogisticRegression(solver="lbfgs", random_state=_RANDOM_SEED, max_iter=500)
    try:
        platt_model.fit(cal_probs.reshape(-1, 1), cal_labels)
        platt_val = platt_model.predict_proba(val_probs.reshape(-1, 1))[:, 1]
        platt_metrics = _metrics(val_labels, platt_val, "platt_scaling")
    except Exception as e:
        platt_metrics = {"method": "platt_scaling", "status": "unavailable", "error": str(e)}
        platt_val = val_probs

    # -- Temperature scaling (grid search on logit-space)
    best_temp, temp_val = _temperature_scale(cal_probs, cal_labels, val_probs)
    temp_metrics = _metrics(val_labels, temp_val, "temperature_scaling")
    temp_metrics["best_temperature"] = _r(best_temp)

    # -- Existing calibrated column (isotonic from training pipeline, if present)
    existing_metrics = None
    if cal_col in preds.columns:
        existing_probs = preds[cal_col].astype(float).to_numpy()
        existing_val = existing_probs[val_idx]
        existing_metrics = _metrics(val_labels, existing_val, "training_pipeline_isotonic")

    # -- Reliability diagram (raw probabilities, all patients)
    reliability = _reliability_diagram(labels, raw_probs)

    # -- Recommendation
    candidates = [m for m in [raw_metrics, iso_metrics, platt_metrics, temp_metrics]
                  if m.get("ece") is not None]
    best = min(candidates, key=lambda m: (m.get("ece") or 1.0))
    recommendation = _recommend(best, raw_metrics, existing_metrics)

    result = {
        "schema_version": "calibration_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Compares calibration methods on the synthetic champion model holdout. "
            "Informs which probability source to surface in the admin dashboard."
        ),
        "status": _status(best.get("ece")),
        "champion_model": best_model,
        "calibration_patients": int(len(cal_idx)),
        "validation_patients": int(len(val_idx)),
        "raw": raw_metrics,
        "isotonic": iso_metrics,
        "platt": platt_metrics,
        "temperature": temp_metrics,
        "training_pipeline_isotonic": existing_metrics,
        "best_method": best.get("method"),
        "best_ece": best.get("ece"),
        "best_brier": best.get("brier"),
        "recommended_source": recommendation["source"],
        "recommendation_reason": recommendation["reason"],
        "reliability_diagram": reliability,
        "claim_boundary": (
            "Calibration fitted on synthetic holdout data only. "
            "Real-world probability reliability requires external validation."
        ),
        "display_hint": (
            "Reliability diagram: each bin shows mean predicted probability vs "
            "observed positive rate. Perfect calibration = diagonal. "
            "ECE <= 0.06 is acceptable for PoC monitoring language with caveats."
        ),
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# -- Core helpers --------------------------------------------------------------

def _metrics(labels: np.ndarray, probs: np.ndarray, method: str) -> dict:
    ece = _ece(labels, probs)
    brier = _r(brier_score_loss(labels, probs))
    return {
        "method": method,
        "ece": ece,
        "brier": brier,
        "status": _status(ece),
    }


def _reliability_diagram(labels: np.ndarray, probs: np.ndarray) -> list:
    bins = []
    n = len(labels)
    edges = np.linspace(0.0, 1.0, _BINS + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi if hi == 1.0 else probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = _r(float(probs[mask].mean()))
        obs_rate = _r(float(labels[mask].mean()))
        bins.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "count": count,
            "fraction_of_total": _r(count / n),
            "mean_predicted_probability": mean_pred,
            "observed_positive_rate": obs_rate,
            "gap": _r(abs((obs_rate or 0) - (mean_pred or 0))),
        })
    return bins


def _temperature_scale(cal_probs, cal_labels, val_probs):
    """Grid-search temperature T minimising NLL on calibration split."""
    cal_probs = np.clip(cal_probs, 1e-6, 1 - 1e-6)
    val_probs = np.clip(val_probs, 1e-6, 1 - 1e-6)
    cal_logits = np.log(cal_probs / (1 - cal_probs))
    val_logits = np.log(val_probs / (1 - val_probs))

    best_T = 1.0
    best_nll = _nll(cal_labels, 1 / (1 + np.exp(-cal_logits)))
    for T in _TEMP_GRID:
        scaled = 1 / (1 + np.exp(-cal_logits / T))
        nll = _nll(cal_labels, scaled)
        if nll < best_nll:
            best_nll = nll
            best_T = T

    scaled_val = 1 / (1 + np.exp(-val_logits / best_T))
    return best_T, scaled_val


def _nll(labels, probs):
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = _BINS) -> float | None:
    if len(labels) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(labels)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi if hi == 1.0 else probs < hi)
        if not np.any(mask):
            continue
        gap = abs(float(labels[mask].mean()) - float(probs[mask].mean()))
        ece += (float(mask.sum()) / n) * gap
    return _r(ece)


def _status(ece) -> str:
    if ece is None:
        return "unavailable"
    if ece <= 0.03:
        return "passed"
    if ece <= 0.06:
        return "strong"
    if ece <= 0.10:
        return "acceptable"
    if ece <= 0.15:
        return "unideal"
    return "failed"


def _recommend(best: dict, raw: dict, existing: dict | None) -> dict:
    best_method = best.get("method", "raw")
    best_ece = best.get("ece") or 1.0
    raw_ece = raw.get("ece") or 1.0

    if existing and (existing.get("ece") or 1.0) <= best_ece:
        return {
            "source": "training_pipeline_isotonic",
            "reason": (
                "The isotonic calibrator from the training pipeline has the lowest or "
                "equal ECE. Prefer the artifact already serialised during training to "
                "avoid extra hold-out leakage."
            ),
        }

    improvement = raw_ece - best_ece
    if improvement < 0.005:
        return {
            "source": "raw",
            "reason": (
                f"Post-hoc calibration offers minimal improvement ({improvement:.4f} ECE delta). "
                "Raw probabilities are adequate; keep probability language hedged."
            ),
        }

    return {
        "source": best_method,
        "reason": (
            f"{best_method} reduced ECE by {improvement:.4f} vs raw. "
            "Fit a locked calibration split before surfacing calibrated probabilities "
            "in any patient-facing context."
        ),
    }


def _load(predictions_path: str, metrics_path: str):
    pp = Path(predictions_path)
    mp = Path(metrics_path)
    if not pp.exists() or not mp.exists():
        return None, None
    preds = pd.read_csv(pp)
    metrics = json.loads(mp.read_text(encoding="utf-8"))
    best = metrics.get("best_model_by_patient_level_roc_auc")
    if not best or _TARGET not in preds.columns:
        return None, None
    return preds, best


def _unavailable(reason: str) -> dict:
    return {
        "schema_version": "calibration_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "reason": reason,
    }


def _r(v, digits: int = 4):
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None
