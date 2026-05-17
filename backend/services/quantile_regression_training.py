"""Quantile-regression head for the response-score signal.

What this is
------------
The existing regression head uses a single point estimator (random forest)
and the inference layer wraps its output in a *heuristic* uncertainty band
scaled to the evidence-sufficiency confidence modifier.  That band is
defensible but it is not derived from the model — it is a UI hint.

This service trains **three quantile gradient-boosting regressors** —
p10, p50, p90 — so the inference layer can emit a *genuine* 80% prediction
interval drawn from the model's own residual distribution.  p50 becomes
the central estimate; (p10, p90) becomes the prediction band.

When the band is wider, the model is genuinely less sure about that
patient's response strength.  When it's narrow, the model is sharp.  This
is the property the heuristic band could not provide.

Artifacts
---------
  - ``Data/complete_synthetic_training/quantile_gbm_p10_response_score_percent.joblib``
  - ``Data/complete_synthetic_training/quantile_gbm_p50_response_score_percent.joblib``
  - ``Data/complete_synthetic_training/quantile_gbm_p90_response_score_percent.joblib``

Metadata
--------
  ``Data/evals/models/latest_quantile_regression_training.json`` carries:
    - quantile config (alphas + seed),
    - per-quantile test pinball loss + MAE on the unaugmented test split,
    - **interval coverage** (the fraction of test rows where the true value
      lands inside [p10, p90] — should be ~0.80 if calibrated),
    - patient-split disjointness (mirrors the leakage-audit contract).

Engineering provenance only.  A passing artifact means the quantile heads
trained without crashing and produced a coverage rate near the nominal
80%; it does **not** establish clinical interval validity.
"""

from __future__ import annotations

import json
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


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_TARGET = "response_score_percent"
DEFAULT_METADATA_PATH = "Data/evals/models/latest_quantile_regression_training.json"

# Triplet of quantiles we train.  10/50/90 is the standard 80% interval choice
# for monitoring signals — narrow enough to be informative, wide enough that
# the residual distribution can actually cover it on synthetic data.
DEFAULT_QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)


def _model_path_for(quantile: float) -> str:
    """Stable artifact filename per quantile (p10 / p50 / p90 etc.)."""
    pct = int(round(quantile * 100))
    return f"Data/complete_synthetic_training/quantile_gbm_p{pct:02d}_response_score_percent.joblib"


def train_quantile_regression_heads(
    *,
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    target: str = DEFAULT_TARGET,
    metadata_output_path: str = DEFAULT_METADATA_PATH,
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the three quantile heads + write the metadata artifact.

    Returns the metadata payload (also written to ``metadata_output_path``).
    """
    rows = pd.read_csv(ml_csv_path)
    if target not in rows.columns:
        raise ValueError(f"Target column '{target}' missing from {ml_csv_path}")

    # Use the project's standard patient-aware split so leakage_audit + this
    # trainer can never disagree about who's in train vs. test.
    train_patients, test_patients = _patient_split(rows, "treatment_success_binary", test_size, seed)
    train_rows = rows[rows["patient_id"].isin(train_patients)].copy()
    test_rows = rows[rows["patient_id"].isin(test_patients)].copy()
    train_rows = train_rows.dropna(subset=[target])
    test_rows = test_rows.dropna(subset=[target])

    X_train = train_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_rows[target].astype(float)
    X_test = test_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_rows[target].astype(float).to_numpy()

    per_quantile_metrics: dict[str, dict[str, Any]] = {}
    quantile_predictions: dict[float, np.ndarray] = {}
    artifact_paths: dict[str, str] = {}

    for q in quantiles:
        model = Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                random_state=seed,
                # Modest depth/n_estimators — synthetic dataset doesn't need
                # heavy capacity to fit the quantile loss cleanly.
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
            )),
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        quantile_predictions[q] = preds

        path = _model_path_for(q)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        pct_key = f"p{int(round(q * 100)):02d}"
        artifact_paths[pct_key] = path

        per_quantile_metrics[pct_key] = {
            "quantile": q,
            "pinball_loss": float(_pinball_loss(y_test, preds, q)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "mean_prediction": float(np.mean(preds)),
        }

    # Interval coverage: with p10/p90 we expect ~80% of true values inside
    # the predicted band.  Coverage materially below that means the band
    # is too tight (underconfident → overclaiming); materially above means
    # it's too wide (overconfident → not informative).
    p_lo = min(quantiles)
    p_hi = max(quantiles)
    lo_preds = quantile_predictions[p_lo]
    hi_preds = quantile_predictions[p_hi]
    sorted_lo_preds = np.minimum(lo_preds, hi_preds)
    sorted_hi_preds = np.maximum(lo_preds, hi_preds)
    nominal_coverage = round(p_hi - p_lo, 4)
    raw_inside = (y_test >= lo_preds) & (y_test <= hi_preds)
    sorted_inside = (y_test >= sorted_lo_preds) & (y_test <= sorted_hi_preds)
    raw_empirical_coverage = float(np.mean(raw_inside)) if len(raw_inside) > 0 else None
    empirical_coverage = float(np.mean(sorted_inside)) if len(sorted_inside) > 0 else None
    raw_median_band_width = float(np.median(hi_preds - lo_preds)) if len(hi_preds) > 0 else None
    median_band_width = (
        float(np.median(sorted_hi_preds - sorted_lo_preds))
        if len(sorted_hi_preds) > 0
        else None
    )

    # Monotonicity sanity: p10 ≤ p50 ≤ p90 should hold for nearly every
    # test row.  Tree-based quantile regressors trained independently can
    # cross occasionally — track the rate so a regression is visible.
    sorted_qs = sorted(quantiles)
    monotonic_rate = None
    if len(sorted_qs) == 3:
        a, b, c = (quantile_predictions[q] for q in sorted_qs)
        monotonic_rate = float(np.mean((a <= b) & (b <= c)))

    metadata: dict[str, Any] = {
        "schema_version": "quantile_regression_training_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _overall_status(empirical_coverage, monotonic_rate, nominal_coverage),
        "target": target,
        "ml_csv_path": ml_csv_path,
        "seed": seed,
        "quantiles": list(quantiles),
        "artifact_paths": artifact_paths,
        "patient_split": {
            "train_patient_count": len(train_patients),
            "test_patient_count": len(test_patients),
            "split_disjoint": len(train_patients & test_patients) == 0,
        },
        "test_rows": int(len(test_rows)),
        "per_quantile_metrics": per_quantile_metrics,
        "interval": {
            "lower_quantile": p_lo,
            "upper_quantile": p_hi,
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": empirical_coverage,
            "empirical_coverage_sorted": empirical_coverage,
            "empirical_coverage_raw": raw_empirical_coverage,
            "median_band_width": median_band_width,
            "median_band_width_raw": raw_median_band_width,
            "coverage_method": "per-row sorted p10/p90 interval, matching inference-time crossing guard",
        },
        "monotonic_rate_p10_p50_p90": monotonic_rate,
        "claim_boundary": (
            "Engineering artifact only.  Coverage near nominal means the "
            "quantile heads produced a calibrated 80% band on the synthetic "
            "test split — it does not establish clinical interval validity."
        ),
    }
    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss — the natural training-time + eval-time
    metric for a single quantile head."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def _overall_status(
    empirical_coverage: float | None,
    monotonic_rate: float | None,
    nominal_coverage: float,
) -> str:
    """Coverage is the primary calibration signal.  Independent quantile
    heads can cross occasionally — that's why the inference layer sorts
    the trio at read time — so raw monotonicity is reported as a
    diagnostic but does not gate the status.

    Strong   = coverage within 5pp of nominal
    Acceptable = coverage within 10pp of nominal
    Otherwise needs_attention.
    """
    if empirical_coverage is None:
        return "missing"
    coverage_gap = abs(empirical_coverage - nominal_coverage)
    if coverage_gap <= 0.05:
        return "strong"
    if coverage_gap <= 0.10:
        return "acceptable"
    return "needs_attention"


def load_quantile_regression_training_metadata(
    path: str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "quantile_regression_training_v1",
            "status": "missing",
            "message": (
                "Quantile regression training has not been run yet. Execute "
                "`scripts/run_quantile_regression_training.py` to produce "
                "the artifacts."
            ),
            "quantiles": list(DEFAULT_QUANTILES),
            "artifact_paths": {},
            "per_quantile_metrics": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_QUANTILES",
    "DEFAULT_TARGET",
    "load_quantile_regression_training_metadata",
    "train_quantile_regression_heads",
]
