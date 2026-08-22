"""Training orchestration: the one public entrypoint.

``train_complete_synthetic_models`` reads the frame, splits at patient level,
dispatches to the classical, regression, and sequence trainers, attaches the
calibrated champion, and writes the artifacts and metrics manifest.

It owns sequencing only. Every computation lives in the focused module it
calls, which is what makes each of those independently testable.
"""

import json
from pathlib import Path

import pandas as pd

from backend.services.complete_synthetic_training.calibration import (
    _attach_calibrated_champion,
)
from backend.services.complete_synthetic_training.classical_models import (
    _train_classical_models,
)
from backend.services.complete_synthetic_training.data_preparation import (
    _ensure_response_regression_columns,
    _patient_split,
    _validate_training_frame,
)
from backend.services.complete_synthetic_training.feature_schema import (
    CATEGORICAL_FEATURES,
    DEFAULT_ML_CSV_PATH,
    DEFAULT_OUTPUT_DIR,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
    RESPONSE_REGRESSION_TARGET,
    ROW_LEVEL_TARGETS,
)
from backend.services.complete_synthetic_training.regression_models import (
    _train_response_regression,
)
from backend.services.complete_synthetic_training.sequence_models import (
    _dl_experiment_report,
    _train_sequence_cnn,
    _train_sequence_cnn_baseline,
    _train_sequence_gru,
)


def train_complete_synthetic_models(
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    target: str = "treatment_success_binary",
    test_size: float = 0.25,
    seed: int = 42,
    cnn_epochs: int = 20,
    cnn_batch_size: int = 16,
):
    rows = _ensure_response_regression_columns(pd.read_csv(ml_csv_path))
    _validate_training_frame(rows, target)
    train_patients, test_patients = _patient_split(rows, target, test_size, seed)
    train_rows = rows[rows["patient_id"].isin(train_patients)].copy()
    test_rows = rows[rows["patient_id"].isin(test_patients)].copy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    classical_results = _train_classical_models(train_rows, test_rows, target, output_path, seed)
    all_models = {**classical_results["models"]}
    artifacts = {**classical_results["artifacts"]}
    predictions = classical_results["predictions"]
    sequence_note = None
    if target not in ROW_LEVEL_TARGETS:
        baseline_cnn_results = _train_sequence_cnn_baseline(
            train_rows=train_rows,
            test_rows=test_rows,
            target=target,
            output_path=output_path,
            seed=seed,
            epochs=cnn_epochs,
            batch_size=cnn_batch_size,
        )
        cnn_results = _train_sequence_cnn(
            train_rows=train_rows,
            test_rows=test_rows,
            target=target,
            output_path=output_path,
            seed=seed,
            epochs=cnn_epochs,
            batch_size=cnn_batch_size,
        )
        gru_results = _train_sequence_gru(
            train_rows=train_rows,
            test_rows=test_rows,
            target=target,
            output_path=output_path,
            seed=seed,
            epochs=cnn_epochs,
            batch_size=cnn_batch_size,
        )
        all_models["temporal_baseline_cnn"] = baseline_cnn_results["metrics"]
        all_models["temporal_1d_cnn"] = cnn_results["metrics"]
        all_models["temporal_gru"] = gru_results["metrics"]
        artifacts["temporal_baseline_cnn"] = baseline_cnn_results["artifact_path"]
        artifacts["temporal_1d_cnn"] = cnn_results["artifact_path"]
        artifacts["temporal_gru"] = gru_results["artifact_path"]
        predictions = predictions.merge(
            baseline_cnn_results["predictions"],
            on=["patient_id", "actual_label"],
            how="outer",
        ).merge(
            cnn_results["predictions"],
            on=["patient_id", "actual_label"],
            how="outer",
        ).merge(
            gru_results["predictions"],
            on=["patient_id", "actual_label"],
            how="outer",
        )
    else:
        sequence_note = "Sequence CNN/GRU skipped because this is a cycle-level monitoring target."

    response_regression = _train_response_regression(train_rows, test_rows, output_path, seed)
    best_model = max(
        all_models,
        key=lambda name: (
            all_models[name].get("patient_level_roc_auc")
            if all_models[name].get("patient_level_roc_auc") is not None
            else all_models[name].get("roc_auc", -1)
        ),
    )
    predictions, calibrated_champion = _attach_calibrated_champion(predictions, best_model, output_path)

    metrics = {
        "task": target,
        "source_csv": ml_csv_path,
        "rows": int(len(rows)),
        "patients": int(rows["patient_id"].nunique()),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded": sorted(EXCLUDED_COLUMNS),
            "response_regression_target": RESPONSE_REGRESSION_TARGET,
        },
        "models": all_models,
        "response_regression": response_regression["metrics"],
        "best_response_regressor_by_patient_level_mae": response_regression["best_model"],
        "calibrated_champion": calibrated_champion["metrics"],
        "dl_experiment_report": _dl_experiment_report(all_models),
        "best_model_by_patient_level_roc_auc": best_model,
        "artifacts": {
            **artifacts,
            **response_regression["artifacts"],
            **calibrated_champion["artifacts"],
            "predictions_csv": str(output_path / "complete_synthetic_model_predictions.csv"),
            "response_regression_predictions_csv": response_regression["predictions_csv"],
        },
        "sequence_note": sequence_note,
        "warning": (
            "Models were trained only on synthetic data. Results measure ability to learn the simulator, "
            "not clinical performance."
        ),
    }

    predictions.to_csv(output_path / "complete_synthetic_model_predictions.csv", index=False)
    (output_path / "complete_synthetic_model_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics
