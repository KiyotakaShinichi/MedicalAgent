"""Champion-model probability calibration.

Fits an isotonic calibrator on a held-out half of the test split and persists
it beside the champion model. The 0.5 split fraction and its literal
``random_state=42`` are preserved exactly: they decide which rows calibrate
versus evaluate, so changing either would move published calibration numbers.

Calibration *measurement* (ECE, reliability bins) lives in :mod:`metrics`.
"""

import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedShuffleSplit

from backend.services.complete_synthetic_training.metrics import _binary_metrics


def _attach_calibrated_champion(predictions, best_model, output_path):
    probability_col = f"{best_model}_probability"
    label_col = "actual_label"
    calibrated_col = f"{best_model}_calibrated_probability"
    if probability_col not in predictions.columns or label_col not in predictions.columns:
        return predictions, {
            "metrics": {
                "status": "unavailable",
                "reason": f"{probability_col} or {label_col} is missing.",
            },
            "artifacts": {},
        }

    frame = predictions[["patient_id", label_col, probability_col]].dropna().copy()
    frame[label_col] = frame[label_col].astype(int)
    frame[probability_col] = frame[probability_col].astype(float).clip(0, 1)
    if len(frame) < 30 or frame[label_col].nunique() < 2:
        return predictions, {
            "metrics": {
                "status": "unavailable",
                "reason": "At least 30 patient-level predictions with both classes are required for calibration.",
            },
            "artifacts": {},
        }

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    calibration_idx, validation_idx = next(splitter.split(frame[[probability_col]], frame[label_col]))
    calibration = frame.iloc[calibration_idx].copy()
    validation = frame.iloc[validation_idx].copy()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration[probability_col].to_numpy(), calibration[label_col].to_numpy())
    frame[calibrated_col] = calibrator.transform(frame[probability_col].to_numpy()).clip(0, 1)
    validation_calibrated = calibrator.transform(validation[probability_col].to_numpy()).clip(0, 1)

    output = predictions.merge(
        frame[["patient_id", calibrated_col]],
        on="patient_id",
        how="left",
    )
    output[calibrated_col] = output[calibrated_col].round(6)
    output[f"{best_model}_calibrated_predicted_label"] = (output[calibrated_col] >= 0.5).astype("Int64")

    artifact_path = output_path / f"{best_model}_isotonic_calibrator_treatment_success_binary.joblib"
    joblib.dump({
        "calibrator": calibrator,
        "model_name": best_model,
        "input_probability_column": probability_col,
        "output_probability_column": calibrated_col,
        "method": "isotonic_regression",
        "calibration_patient_ids": calibration["patient_id"].tolist(),
        "validation_patient_ids": validation["patient_id"].tolist(),
        "warning": "Synthetic-data calibrator. Needs locked/external validation before stronger probability claims.",
    }, artifact_path)

    metrics = {
        "status": "trained",
        "model_name": best_model,
        "method": "isotonic_regression",
        "probability_column": calibrated_col,
        "calibration_patients": int(len(calibration)),
        "validation_patients": int(len(validation)),
        "raw_validation": _binary_metrics(
            validation[label_col].to_numpy(),
            validation[probability_col].to_numpy(),
            prefix="validation_",
        ),
        "calibrated_validation": _binary_metrics(
            validation[label_col].to_numpy(),
            validation_calibrated,
            prefix="validation_",
        ),
        "all_holdout_calibrated": _binary_metrics(
            frame[label_col].to_numpy(),
            frame[calibrated_col].to_numpy(),
            prefix="patient_level_",
        ),
        "claim_boundary": (
            "This calibrated probability head is trained on a synthetic holdout calibration split. "
            "It improves engineering probability behavior but is not clinical validation."
        ),
    }
    return output, {
        "metrics": metrics,
        "artifacts": {
            f"{best_model}_isotonic_calibrator": str(artifact_path),
        },
    }
