import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RESPONSE_REGRESSION_TARGET,
    _aggregate_patient_predictions,
    _aggregate_patient_regression_predictions,
    _binary_metrics,
    _regression_metrics,
)
from backend.services.detailed_training_report import _add_response_uncertainty_columns


DEFAULT_MODEL_DIR = "Data/complete_synthetic_training"
DEFAULT_METRICS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
DEFAULT_LOCKED_HOLDOUT_ROWS_PATH = "Data/complete_synthetic_training/locked_holdout/locked_holdout_rows.csv"
DEFAULT_OUTPUT_DIR = "Data/complete_synthetic_training/locked_holdout_eval"


def evaluate_locked_holdout(
    model_dir=DEFAULT_MODEL_DIR,
    metrics_path=DEFAULT_METRICS_PATH,
    locked_holdout_rows_path=DEFAULT_LOCKED_HOLDOUT_ROWS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(locked_holdout_rows_path)
    metrics = _load_json(metrics_path)
    target = metrics.get("task") or "treatment_success_binary"
    best_classifier = metrics.get("best_model_by_patient_level_roc_auc")
    best_regressor = (
        metrics.get("best_response_regressor_by_patient_level_mae")
        or (metrics.get("response_regression") or {}).get("best_model_by_patient_level_mae")
    )
    if not best_classifier:
        raise ValueError("No best classifier found in metrics artifact.")

    classification = _evaluate_classifier(rows, target, best_classifier, model_path)
    regression = _evaluate_regressor(rows, best_regressor, model_path) if best_regressor else {
        "metrics": {"status": "unavailable", "reason": "No best regressor in metrics artifact."},
        "predictions": pd.DataFrame(),
    }
    detailed = classification["predictions"]
    if not regression["predictions"].empty:
        detailed = detailed.merge(regression["predictions"], on="patient_id", how="left")
        detailed = _add_response_uncertainty_columns(detailed, best_regressor)

    files = {
        "locked_holdout_predictions_csv": str(output_path / "locked_holdout_predictions.csv"),
        "locked_holdout_summary_json": str(output_path / "locked_holdout_summary.json"),
    }
    detailed.to_csv(files["locked_holdout_predictions_csv"], index=False)
    summary = {
        "schema_version": "locked_holdout_evaluation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_path),
        "metrics_path": metrics_path,
        "locked_holdout_rows_path": locked_holdout_rows_path,
        "patients": int(rows["patient_id"].nunique()),
        "rows": int(len(rows)),
        "best_classifier": best_classifier,
        "best_regressor": best_regressor,
        "classification": classification["metrics"],
        "regression": regression["metrics"],
        "response_uncertainty": _uncertainty_summary(detailed),
        "files": files,
        "claim_boundary": (
            "This is a locked synthetic holdout evaluation. It is stronger engineering discipline than "
            "an ordinary internal split, but it is not external clinical validation."
        ),
    }
    Path(files["locked_holdout_summary_json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _evaluate_classifier(rows, target, best_classifier, model_path):
    artifact = model_path / f"{best_classifier}_{target}.joblib"
    model = joblib.load(artifact)
    X = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    probabilities = model.predict_proba(X)[:, 1]
    patient_predictions = _aggregate_patient_predictions(rows, target, probabilities, best_classifier)
    raw_metrics = _binary_metrics(
        patient_predictions["actual_label"].astype(int),
        patient_predictions[f"{best_classifier}_probability"].astype(float),
        prefix="patient_level_",
    )

    calibrator_path = model_path / f"{best_classifier}_isotonic_calibrator_treatment_success_binary.joblib"
    calibrated_metrics = None
    if calibrator_path.exists():
        payload = joblib.load(calibrator_path)
        calibrator = payload.get("calibrator")
        calibrated_col = payload.get("output_probability_column") or f"{best_classifier}_calibrated_probability"
        raw_col = f"{best_classifier}_probability"
        patient_predictions[calibrated_col] = np.clip(
            calibrator.transform(patient_predictions[raw_col].astype(float).to_numpy()),
            0,
            1,
        ).round(6)
        patient_predictions[f"{best_classifier}_calibrated_predicted_label"] = (
            patient_predictions[calibrated_col] >= 0.5
        ).astype(int)
        calibrated_metrics = _binary_metrics(
            patient_predictions["actual_label"].astype(int),
            patient_predictions[calibrated_col].astype(float),
            prefix="patient_level_",
        )

    return {
        "metrics": {
            "status": "evaluated",
            "raw_probability": raw_metrics,
            "calibrated_probability": calibrated_metrics,
            "probability_used_for_routing": "calibrated_probability" if calibrated_metrics else "raw_probability",
        },
        "predictions": patient_predictions,
    }


def _evaluate_regressor(rows, best_regressor, model_path):
    target = RESPONSE_REGRESSION_TARGET
    if target not in rows.columns:
        return {
            "metrics": {"status": "unavailable", "reason": f"{target} column is missing from holdout rows."},
            "predictions": pd.DataFrame(),
        }
    X = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predictions = _regressor_row_predictions(X, best_regressor, model_path)
    patient_predictions = _aggregate_patient_regression_predictions(rows, target, predictions, best_regressor)

    member_predictions = {}
    for member in _robust_members():
        if member == best_regressor:
            continue
        artifact = model_path / f"{member}_{target}.joblib"
        if artifact.exists():
            member_predictions[member] = _aggregate_patient_regression_predictions(
                rows,
                target,
                joblib.load(artifact).predict(X),
                member,
            )
    for member, frame in member_predictions.items():
        member_col = f"{member}_response_score_percent"
        if member_col in patient_predictions.columns:
            continue
        patient_predictions = patient_predictions.merge(
            frame.drop(columns=["actual_response_score_percent"]),
            on="patient_id",
            how="left",
        )

    pred_col = f"{best_regressor}_response_score_percent"
    return {
        "metrics": {
            "status": "evaluated",
            **_regression_metrics(
                patient_predictions["actual_response_score_percent"],
                patient_predictions[pred_col],
                prefix="patient_level_",
            ),
        },
        "predictions": patient_predictions,
    }


def _regressor_row_predictions(X, best_regressor, model_path):
    target = RESPONSE_REGRESSION_TARGET
    if best_regressor == "robust_response_ensemble":
        matrices = []
        for member in _robust_members():
            artifact = model_path / f"{member}_{target}.joblib"
            if artifact.exists():
                matrices.append(joblib.load(artifact).predict(X))
        if len(matrices) < 2:
            raise ValueError("Robust response ensemble needs at least two member artifacts.")
        return np.median(np.column_stack(matrices), axis=1)
    artifact = model_path / f"{best_regressor}_{target}.joblib"
    return joblib.load(artifact).predict(X)


def _robust_members():
    return [
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
        "gradient_boosting_huber_regressor",
        "huber_regressor",
    ]


def _uncertainty_summary(frame):
    if frame.empty or "response_uncertainty_band" not in frame.columns:
        return {"status": "unavailable"}
    counts = frame["response_uncertainty_band"].value_counts(dropna=False).to_dict()
    return {
        "status": "available",
        "band_counts": {str(key): int(value) for key, value in counts.items()},
        "mean_width": round(float(pd.to_numeric(frame["response_uncertainty_width"], errors="coerce").mean()), 3),
        "interpretation": "Width is the 10th-to-90th percentile spread across response-regression model family predictions.",
    }


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
