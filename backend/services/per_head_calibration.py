from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, DEFAULT_ML_CSV_PATH, NUMERIC_FEATURES


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_per_head_calibration.json"
DENYLIST = {
    "patient_id", "treatment_date", "latent_response_strength", "response_score_percent",
    "treatment_success_binary", "toxicity_risk_binary", "urgent_intervention_needed",
    "support_intervention_needed", "cycle_response_trend_class", "final_response_category",
    "final_cancer_status", "maintenance_needed", "final_response_multiclass",
}


def run_per_head_calibration(source_csv: str = DEFAULT_ML_CSV_PATH, output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    frame = pd.read_csv(source_csv)
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in frame.columns and c not in DENYLIST]
    train, test = _patient_split(frame)
    class_model = _calibrated_classifier(features).fit(train[features], train["treatment_success_binary"].astype(int))
    tox_model = _calibrated_classifier(features).fit(train[features], train["toxicity_risk_binary"].astype(int))
    reg_model = _regressor(features).fit(train[features], train["response_score_percent"].astype(float))

    class_prob = class_model.predict_proba(test[features])[:, 1]
    tox_prob = tox_model.predict_proba(test[features])[:, 1]
    reg_pred = reg_model.predict(test[features])
    residual = np.abs(reg_pred - test["response_score_percent"].astype(float).to_numpy())
    q80 = float(np.quantile(residual, 0.80))

    payload = {
        **build_artifact_manifest(dataset_paths={"source_csv": source_csv}),
        "schema_version": "per_head_calibration_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "heads": {
            "response_classification": _classification_metrics(test["treatment_success_binary"].astype(int), class_prob),
            "toxicity": _classification_metrics(test["toxicity_risk_binary"].astype(int), tox_prob),
            "response_regression": {
                "mae": round(float(mean_absolute_error(test["response_score_percent"], reg_pred)), 4),
                "interval_method": "absolute_residual_q80_synthetic_proxy",
                "q80_absolute_error": round(q80, 4),
                "empirical_coverage_q80": round(float((residual <= q80).mean()), 4),
                "uncertainty_story": "Prediction interval proxy from held-out synthetic residuals.",
            },
            "abstention": {
                "coverage_vs_error_tradeoff": [
                    _coverage_tradeoff(class_prob, test["treatment_success_binary"].astype(int).to_numpy(), threshold)
                    for threshold in (0.55, 0.65, 0.75, 0.85)
                ],
                "uncertainty_story": "Higher confidence threshold lowers coverage and should raise covered accuracy.",
            },
        },
        "claim_boundary": (
            "Per-head calibration is estimated on synthetic holdout rows only. "
            "It compares engineering uncertainty behavior, not clinical calibration."
        ),
    }
    _write_json(output_path, payload)
    return payload


def load_per_head_calibration(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "missing"}


def _classification_metrics(y_true, prob) -> dict[str, Any]:
    return {
        "auroc": _safe_auc(y_true, prob),
        "brier": round(float(brier_score_loss(y_true, prob)), 4),
        "ece": _ece(y_true, prob),
        "reliability_bins": _reliability_bins(y_true, prob),
        "uncertainty_story": "Isotonic-calibrated probability with Brier/ECE/reliability bins.",
    }


def _patient_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = sorted(frame["patient_id"].astype(str).unique())
    train_ids = set(ids[: int(len(ids) * 0.75)])
    return frame[frame["patient_id"].astype(str).isin(train_ids)].copy(), frame[~frame["patient_id"].astype(str).isin(train_ids)].copy()


def _preprocessor(features: list[str]) -> ColumnTransformer:
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    num = [c for c in features if c not in cat]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def _calibrated_classifier(features: list[str]) -> Pipeline:
    base = GradientBoostingClassifier(random_state=42)
    return Pipeline([("pre", _preprocessor(features)), ("model", CalibratedClassifierCV(base, method="isotonic", cv=3))])


def _regressor(features: list[str]) -> Pipeline:
    return Pipeline([("pre", _preprocessor(features)), ("model", GradientBoostingRegressor(random_state=42))])


def _safe_auc(y_true, prob) -> float:
    values = np.asarray(y_true)
    if len(set(values.tolist())) < 2:
        return 0.0
    return round(float(roc_auc_score(y_true, prob)), 4)


def _ece(y_true, prob, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(p)
    score = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if not mask.any():
            continue
        score += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return round(float(score), 4)


def _reliability_bins(y_true, prob, bins: int = 5) -> list[dict[str, Any]]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    rows = []
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        rows.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "count": int(mask.sum()),
            "mean_probability": round(float(p[mask].mean()), 4) if mask.any() else None,
            "empirical_rate": round(float(y[mask].mean()), 4) if mask.any() else None,
        })
    return rows


def _coverage_tradeoff(prob, y_true, threshold: float) -> dict[str, Any]:
    confidence = np.maximum(prob, 1 - prob)
    cover = confidence >= threshold
    pred = prob >= 0.5
    return {
        "confidence_threshold": threshold,
        "coverage": round(float(cover.mean()), 4),
        "covered_accuracy": round(float((pred[cover] == y_true[cover]).mean()), 4) if cover.any() else None,
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["DEFAULT_OUTPUT_PATH", "run_per_head_calibration", "load_per_head_calibration"]
