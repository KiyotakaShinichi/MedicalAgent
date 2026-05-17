"""Modality-dropout quantile regression for response-strength intervals.

This combines the two regression upgrades already in the project:

* quantile heads (p10/p50/p90) for an 80% interval, and
* modality-dropout augmentation so missing CBC/imaging/symptom groups are
  learned as normal synthetic missingness patterns rather than surprises.

The output is still synthetic engineering evidence only.  It improves the
response-score head's behavior under missing modalities; it does not validate
clinical response prediction.
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
from backend.services.evidence_abstention_eval import SCENARIOS, _strip_modalities
from backend.services.evidence_sufficiency import assess_evidence
from backend.services.modality_dropout_training import _augment_with_modality_dropout
from backend.services.quantile_regression_training import (
    DEFAULT_QUANTILES,
    _pinball_loss,
)


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_TARGET = "response_score_percent"
DEFAULT_METADATA_PATH = "Data/evals/models/latest_modality_dropout_quantile_regression_training.json"
DEFAULT_N_AUG_PER_ROW = 3
DEFAULT_P_DROP_PER_MODALITY = 0.30
DEFAULT_MAX_SIMULTANEOUS_DROPOUTS = 3
DEFAULT_PROTECTED_MODALITIES = ("demographics",)


def robust_quantile_model_path_for(quantile: float) -> str:
    pct = int(round(quantile * 100))
    return f"Data/complete_synthetic_training/modality_dropout_quantile_gbm_p{pct:02d}_response_score_percent.joblib"


def train_modality_dropout_quantile_regression_heads(
    *,
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    target: str = DEFAULT_TARGET,
    metadata_output_path: str = DEFAULT_METADATA_PATH,
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    n_aug_per_row: int = DEFAULT_N_AUG_PER_ROW,
    p_drop_per_modality: float = DEFAULT_P_DROP_PER_MODALITY,
    max_simultaneous_dropouts: int = DEFAULT_MAX_SIMULTANEOUS_DROPOUTS,
    protected_modalities: tuple[str, ...] = DEFAULT_PROTECTED_MODALITIES,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = pd.read_csv(ml_csv_path)
    if target not in rows.columns:
        raise ValueError(f"Target column '{target}' missing from {ml_csv_path}")

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

    X_train = augmented_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = augmented_train[target].astype(float)
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
                n_estimators=240,
                max_depth=4,
                learning_rate=0.05,
            )),
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        quantile_predictions[q] = preds

        path = robust_quantile_model_path_for(q)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        key = f"p{int(round(q * 100)):02d}"
        artifact_paths[key] = path
        per_quantile_metrics[key] = {
            "quantile": q,
            "pinball_loss": float(_pinball_loss(y_test, preds, q)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "mean_prediction": float(np.mean(preds)),
        }

    interval = _interval_metrics(y_test, quantile_predictions, quantiles)
    scenario_comparison = _scenario_comparison(rows.dropna(subset=[target]).copy(), quantiles, artifact_paths, target)
    status = _overall_status(interval, scenario_comparison)

    metadata: dict[str, Any] = {
        "schema_version": "modality_dropout_quantile_regression_training_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
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
        "training_config": {
            "n_aug_per_row": n_aug_per_row,
            "p_drop_per_modality": p_drop_per_modality,
            "max_simultaneous_dropouts": max_simultaneous_dropouts,
            "protected_modalities": list(protected_modalities),
            "test_size": test_size,
        },
        "augmentation_stats": dropout_stats,
        "test_rows": int(len(test_rows)),
        "per_quantile_metrics": per_quantile_metrics,
        "interval": interval,
        "scenario_comparison": scenario_comparison,
        "claim_boundary": (
            "Synthetic engineering artifact only. Modality-dropout quantile "
            "heads improve interval behavior under simulated missingness; "
            "they do not establish clinical treatment-response validity."
        ),
    }
    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _interval_metrics(
    y_true: np.ndarray,
    predictions: dict[float, np.ndarray],
    quantiles: tuple[float, ...],
) -> dict[str, Any]:
    p_lo = min(quantiles)
    p_hi = max(quantiles)
    lo = predictions[p_lo]
    hi = predictions[p_hi]
    sorted_lo = np.minimum(lo, hi)
    sorted_hi = np.maximum(lo, hi)
    inside = (y_true >= sorted_lo) & (y_true <= sorted_hi)
    sorted_qs = sorted(quantiles)
    monotonic_rate = None
    if len(sorted_qs) == 3:
        a, b, c = (predictions[q] for q in sorted_qs)
        monotonic_rate = float(np.mean((a <= b) & (b <= c)))
    return {
        "lower_quantile": p_lo,
        "upper_quantile": p_hi,
        "nominal_coverage": round(p_hi - p_lo, 4),
        "empirical_coverage": float(np.mean(inside)) if len(inside) else None,
        "median_band_width": float(np.median(sorted_hi - sorted_lo)) if len(sorted_hi) else None,
        "monotonic_rate_p10_p50_p90": monotonic_rate,
        "coverage_method": "per-row sorted p10/p90 interval, matching inference-time crossing guard",
    }


def _scenario_comparison(
    rows: pd.DataFrame,
    quantiles: tuple[float, ...],
    artifact_paths: dict[str, str],
    target: str,
) -> dict[str, Any]:
    robust_models = {
        q: joblib.load(artifact_paths[f"p{int(round(q * 100)):02d}"])
        for q in quantiles
    }
    from backend.services.quantile_regression_training import _model_path_for

    legacy_models: dict[float, Any] = {}
    legacy_available = True
    for q in quantiles:
        path = Path(_model_path_for(q))
        if not path.exists():
            legacy_available = False
            break
        legacy_models[q] = joblib.load(path)

    scenario_rows: list[dict[str, Any]] = []
    for scenario, stripped in SCENARIOS.items():
        masked = _strip_modalities(rows, stripped)
        y_true = masked[target].astype(float).to_numpy()
        covered_indices = [
            idx for idx, row in enumerate(masked.to_dict("records"))
            if not assess_evidence(row, question="response_regression").abstain
        ]
        scenario_rows.append({
            "scenario": scenario,
            "stripped_modalities": list(stripped),
            "rows_evaluated": int(len(masked)),
            "coverage_rate": round(len(covered_indices) / max(1, len(masked)), 4),
            "robust": _score_quantile_models(masked, y_true, robust_models, quantiles),
            "legacy": (
                _score_quantile_models(masked, y_true, legacy_models, quantiles)
                if legacy_available
                else {"status": "missing"}
            ),
        })
        if legacy_available:
            scenario_rows[-1]["deltas"] = {
                "p50_mae_robust_minus_legacy": _round(
                    scenario_rows[-1]["robust"]["p50_mae"] - scenario_rows[-1]["legacy"]["p50_mae"],
                ),
                "coverage_gap_robust_minus_legacy": _round(
                    scenario_rows[-1]["robust"]["coverage_gap"] - scenario_rows[-1]["legacy"]["coverage_gap"],
                ),
            }

    wins = sum(
        1 for row in scenario_rows
        if row.get("deltas") and row["deltas"]["p50_mae_robust_minus_legacy"] < -0.5
    )
    losses = sum(
        1 for row in scenario_rows
        if row.get("deltas") and row["deltas"]["p50_mae_robust_minus_legacy"] > 0.5
    )
    return {
        "legacy_available": legacy_available,
        "scenario_count": len(scenario_rows),
        "robust_mae_wins": wins,
        "robust_mae_losses": losses,
        "scenarios": scenario_rows,
    }


def _score_quantile_models(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    models: dict[float, Any],
    quantiles: tuple[float, ...],
) -> dict[str, Any]:
    X = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds = {q: models[q].predict(X) for q in quantiles}
    p_lo = min(quantiles)
    p_hi = max(quantiles)
    lo = np.minimum(preds[p_lo], preds[p_hi])
    hi = np.maximum(preds[p_lo], preds[p_hi])
    inside = (y_true >= lo) & (y_true <= hi)
    p50 = preds[0.50] if 0.50 in preds else preds[sorted(quantiles)[len(quantiles) // 2]]
    nominal = p_hi - p_lo
    empirical = float(np.mean(inside)) if len(inside) else None
    return {
        "p50_mae": _round(float(mean_absolute_error(y_true, p50))),
        "p50_rmse": _round(float(np.sqrt(np.mean((y_true - p50) ** 2)))),
        "empirical_coverage": _round(empirical),
        "coverage_gap": _round(abs((empirical or 0.0) - nominal)),
        "median_band_width": _round(float(np.median(hi - lo))),
    }


def _overall_status(interval: dict[str, Any], scenario_comparison: dict[str, Any]) -> str:
    empirical = interval.get("empirical_coverage")
    nominal = interval.get("nominal_coverage")
    if empirical is None or nominal is None:
        return "missing"
    coverage_gap = abs(float(empirical) - float(nominal))
    if coverage_gap > 0.12:
        return "needs_attention"
    if scenario_comparison.get("legacy_available") and (
        scenario_comparison.get("robust_mae_losses", 0) > scenario_comparison.get("robust_mae_wins", 0) + 2
    ):
        return "acceptable"
    return "strong" if coverage_gap <= 0.06 else "acceptable"


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def load_modality_dropout_quantile_regression_metadata(
    path: str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "modality_dropout_quantile_regression_training_v1",
            "status": "missing",
            "message": (
                "Modality-dropout quantile regression has not been run yet. "
                "Execute `scripts/run_modality_dropout_quantile_regression_training.py`."
            ),
            "artifact_paths": {},
            "interval": {},
            "scenario_comparison": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_METADATA_PATH",
    "load_modality_dropout_quantile_regression_metadata",
    "robust_quantile_model_path_for",
    "train_modality_dropout_quantile_regression_heads",
]
