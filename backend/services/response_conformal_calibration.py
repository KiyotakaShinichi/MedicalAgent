"""Conformal calibration for response-score prediction intervals.

The robust quantile regression head gives an 80% interval, but the latest
artifact under-covers slightly on the synthetic holdout. This service computes
a simple split-conformal adjustment over the held-out patient split:

    residual = max(lower - y, y - upper, 0)

The selected residual quantile is then added to both sides of the interval at
inference. This is still synthetic-only engineering calibration, not clinical
validity.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _patient_split,
)
from backend.services.modality_dropout_quantile_regression_training import (
    robust_quantile_model_path_for,
)
from backend.services.quantile_regression_training import DEFAULT_QUANTILES


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_TARGET = "response_score_percent"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_response_conformal_calibration.json"
DEFAULT_NOMINAL_COVERAGE = 0.80


def build_response_conformal_calibration(
    *,
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    target: str = DEFAULT_TARGET,
    nominal_coverage: float = DEFAULT_NOMINAL_COVERAGE,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    rows = pd.read_csv(ml_csv_path)
    if target not in rows.columns:
        raise ValueError(f"Target column '{target}' missing from {ml_csv_path}")

    _, test_patients = _patient_split(rows, "treatment_success_binary", test_size, seed)
    calib_rows = rows[rows["patient_id"].isin(test_patients)].dropna(subset=[target]).copy()
    if calib_rows.empty:
        raise ValueError("No calibration rows available.")

    models = {}
    for q in DEFAULT_QUANTILES:
        path = Path(robust_quantile_model_path_for(q))
        if not path.exists():
            raise FileNotFoundError(f"Missing robust quantile model: {path}")
        models[q] = joblib.load(path)

    X = calib_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = calib_rows[target].astype(float).to_numpy()
    lo_raw = models[min(DEFAULT_QUANTILES)].predict(X)
    mid_raw = models[0.50].predict(X)
    hi_raw = models[max(DEFAULT_QUANTILES)].predict(X)
    lo = np.minimum(lo_raw, hi_raw)
    hi = np.maximum(lo_raw, hi_raw)
    residuals = np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])

    q_level = math.ceil((len(residuals) + 1) * nominal_coverage) / len(residuals)
    q_level = min(1.0, max(0.0, q_level))
    qhat = float(np.quantile(residuals, q_level, method="higher"))
    adjusted_lo = lo - qhat
    adjusted_hi = hi + qhat
    raw_coverage = float(np.mean((y >= lo) & (y <= hi)))
    adjusted_coverage = float(np.mean((y >= adjusted_lo) & (y <= adjusted_hi)))

    payload = {
        "schema_version": "response_conformal_calibration_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status(adjusted_coverage, nominal_coverage),
        "target": target,
        "ml_csv_path": ml_csv_path,
        "model_family": "modality_dropout_quantile_gbm",
        "nominal_coverage": nominal_coverage,
        "calibration_rows": int(len(calib_rows)),
        "q_level": q_level,
        "qhat_percent": qhat,
        "qhat_normalized": qhat / 100.0,
        "raw_coverage": raw_coverage,
        "adjusted_coverage": adjusted_coverage,
        "raw_median_band_width": float(np.median(hi - lo)),
        "adjusted_median_band_width": float(np.median(adjusted_hi - adjusted_lo)),
        "median_p50_mae": float(np.mean(np.abs(mid_raw - y))),
        "claim_boundary": (
            "Synthetic split-conformal interval calibration only. It improves "
            "coverage on a held-out synthetic patient split and does not prove "
            "clinical response-score validity."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _load_response_conformal_calibration.cache_clear()
    return payload


def _status(adjusted_coverage: float, nominal_coverage: float) -> str:
    gap = abs(adjusted_coverage - nominal_coverage)
    if gap <= 0.05:
        return "strong"
    if gap <= 0.10:
        return "acceptable"
    return "needs_attention"


@lru_cache(maxsize=1)
def _load_response_conformal_calibration(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "response_conformal_calibration_v1",
            "status": "missing",
            "qhat_normalized": 0.0,
            "message": "Run scripts/run_response_conformal_calibration.py to generate this artifact.",
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


def conformal_adjustment(path: str = DEFAULT_OUTPUT_PATH) -> float:
    payload = _load_response_conformal_calibration(path)
    if payload.get("status") not in {"strong", "acceptable"}:
        return 0.0
    return float(payload.get("qhat_normalized") or 0.0)


def load_response_conformal_calibration(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    return _load_response_conformal_calibration(path)


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "build_response_conformal_calibration",
    "conformal_adjustment",
    "load_response_conformal_calibration",
]
