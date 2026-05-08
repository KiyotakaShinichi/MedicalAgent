import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_DIR = "Data/complete_synthetic_training"
RESPONSE_REGRESSION_TARGET = "response_score_percent"

NUMERIC_FEATURES = [
    "cycle",
    "age",
    "pre_wbc",
    "pre_anc",
    "pre_hemoglobin",
    "pre_platelets",
    "nadir_wbc",
    "nadir_anc",
    "nadir_hemoglobin",
    "nadir_platelets",
    "recovery_wbc",
    "recovery_hemoglobin",
    "recovery_platelets",
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
    "max_symptom_severity",
    "symptom_count",
    "intervention_count",
    "dose_delayed",
    "dose_reduced",
]

CATEGORICAL_FEATURES = ["stage", "molecular_subtype", "regimen"]
ROW_LEVEL_TARGETS = {
    "toxicity_risk_binary",
    "support_intervention_needed",
    "urgent_intervention_needed",
}
EXCLUDED_COLUMNS = {
    "patient_id",
    "treatment_date",
    "latent_response_strength",
    "response_score_percent",
    "final_response_category",
    "final_cancer_status",
    "final_response_multiclass",
    "treatment_success_binary",
    "maintenance_needed",
    "toxicity_risk_binary",
    "support_intervention_needed",
    "urgent_intervention_needed",
    "cycle_response_trend_class",
}


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


def _ensure_response_regression_columns(rows):
    rows = rows.copy()
    if RESPONSE_REGRESSION_TARGET not in rows.columns and "mri_percent_change_from_baseline" in rows.columns:
        rows[RESPONSE_REGRESSION_TARGET] = -pd.to_numeric(rows["mri_percent_change_from_baseline"], errors="coerce")
    if RESPONSE_REGRESSION_TARGET in rows.columns:
        rows[RESPONSE_REGRESSION_TARGET] = pd.to_numeric(rows[RESPONSE_REGRESSION_TARGET], errors="coerce")
        rows[RESPONSE_REGRESSION_TARGET] = (
            rows.groupby("patient_id")[RESPONSE_REGRESSION_TARGET]
            .transform(lambda series: series.ffill().bfill())
            .clip(-100, 100)
        )
    return rows


def _validate_training_frame(rows, target):
    missing = [col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["patient_id", target] if col not in rows.columns]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    if target in ROW_LEVEL_TARGETS:
        if rows[target].nunique() < 2:
            raise ValueError(f"Target {target} needs at least two classes")
        return
    patient_labels = rows.groupby("patient_id")[target].max()
    if patient_labels.nunique() < 2:
        raise ValueError(f"Target {target} needs at least two classes")


def _patient_split(rows, target, test_size, seed):
    if target in ROW_LEVEL_TARGETS:
        patient_labels = (
            rows.groupby("patient_id", as_index=False)[target]
            .mean()
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        median_rate = patient_labels[target].median()
        patient_labels["split_label"] = (patient_labels[target] >= median_rate).astype(int)
        if patient_labels["split_label"].nunique() < 2:
            patient_labels["split_label"] = (patient_labels[target] > 0).astype(int)
    else:
        patient_labels = (
            rows.groupby("patient_id", as_index=False)[target]
            .max()
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        patient_labels["split_label"] = patient_labels[target].astype(int)

    patient_labels = (
        patient_labels[["patient_id", "split_label"]]
    )
    train_patients, test_patients = train_test_split(
        patient_labels["patient_id"],
        test_size=test_size,
        random_state=seed,
        stratify=patient_labels["split_label"].astype(int),
    )
    return set(train_patients), set(test_patients)


def _train_classical_models(train_rows, test_rows, target, output_path, seed):
    X_train = train_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_rows[target].astype(int)
    X_test = test_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_rows[target].astype(int)

    models = {
        "logistic_regression": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=350,
                max_depth=7,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "extra_trees": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("classifier", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("classifier", GradientBoostingClassifier(random_state=seed)),
        ]),
        "svm_rbf": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("classifier", SVC(C=1.5, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]),
        "mlp": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(48, 24),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=600,
                random_state=seed,
            )),
        ]),
    }

    model_metrics = {}
    artifacts = {}
    prediction_rows = _base_patient_prediction_rows(test_rows, target)
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        row_metrics = _binary_metrics(y_test, probabilities)
        patient_predictions = _aggregate_patient_predictions(test_rows, target, probabilities, name)
        patient_metrics = _binary_metrics(
            patient_predictions["actual_label"].astype(int),
            patient_predictions[f"{name}_probability"],
            prefix="patient_level_",
        )
        model_metrics[name] = {**row_metrics, **patient_metrics, "model_type": "cycle_tabular_classifier"}
        artifact_path = output_path / f"{name}_{target}.joblib"
        joblib.dump(model, artifact_path)
        artifacts[name] = str(artifact_path)
        prediction_rows = prediction_rows.merge(patient_predictions, on=["patient_id", "actual_label"], how="left")

    return {
        "models": model_metrics,
        "artifacts": artifacts,
        "predictions": prediction_rows,
    }


def _train_response_regression(train_rows, test_rows, output_path, seed):
    target = RESPONSE_REGRESSION_TARGET
    if target not in train_rows.columns or target not in test_rows.columns:
        return {
            "metrics": {"status": "unavailable", "reason": f"{target} column is missing"},
            "best_model": None,
            "artifacts": {},
            "predictions_csv": None,
        }

    train = train_rows.dropna(subset=[target]).copy()
    test = test_rows.dropna(subset=[target]).copy()
    if train.empty or test.empty:
        return {
            "metrics": {"status": "unavailable", "reason": "No non-null response regression labels"},
            "best_model": None,
            "artifacts": {},
            "predictions_csv": None,
        }

    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train[target].astype(float)
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[target].astype(float)
    models = {
        "ridge_regression": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("regressor", Ridge(alpha=1.0)),
        ]),
        "random_forest_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                random_state=seed,
            )),
        ]),
        "extra_trees_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", ExtraTreesRegressor(
                n_estimators=350,
                max_depth=8,
                min_samples_leaf=3,
                random_state=seed,
            )),
        ]),
        "gradient_boosting_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", GradientBoostingRegressor(random_state=seed)),
        ]),
        "svr_rbf_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("regressor", SVR(C=2.0, kernel="rbf")),
        ]),
    }

    model_metrics = {}
    artifacts = {}
    predictions = _base_patient_regression_rows(test, target)
    for name, model in models.items():
        model.fit(X_train, y_train)
        row_predictions = model.predict(X_test)
        patient_predictions = _aggregate_patient_regression_predictions(test, target, row_predictions, name)
        model_metrics[name] = {
            **_regression_metrics(y_test, row_predictions),
            **_regression_metrics(
                patient_predictions["actual_response_score_percent"],
                patient_predictions[f"{name}_response_score_percent"],
                prefix="patient_level_",
            ),
            "model_type": "cycle_tabular_regressor",
            "target": target,
            "interpretation": "Positive values estimate tumor-size reduction percent; negative values indicate growth/progression signal in the synthetic simulator.",
        }
        artifact_path = output_path / f"{name}_{target}.joblib"
        joblib.dump(model, artifact_path)
        artifacts[f"{name}_response_regression"] = str(artifact_path)
        predictions = predictions.merge(patient_predictions, on=["patient_id", "actual_response_score_percent"], how="left")

    best_model = min(
        model_metrics,
        key=lambda name: model_metrics[name].get("patient_level_mae", float("inf")),
    )
    predictions_csv = output_path / "complete_synthetic_response_regression_predictions.csv"
    predictions.to_csv(predictions_csv, index=False)
    return {
        "metrics": {
            "status": "trained",
            "task": "response_score_regression",
            "target": target,
            "models": model_metrics,
            "best_model_by_patient_level_mae": best_model,
            "target_definition": "Continuous MRI response signal: baseline-to-current tumor-size reduction percent; higher is stronger shrinkage, negative means growth.",
        },
        "best_model": best_model,
        "artifacts": artifacts,
        "predictions_csv": str(predictions_csv),
    }


def _train_sequence_cnn_baseline(train_rows, test_rows, target, output_path, seed, epochs, batch_size):
    return _train_sequence_torch_model(
        model_name="temporal_baseline_cnn",
        model_factory=lambda input_features: BaselineTemporalCnn(input_features=input_features),
        train_rows=train_rows,
        test_rows=test_rows,
        target=target,
        output_path=output_path,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
    )


def _train_sequence_cnn(train_rows, test_rows, target, output_path, seed, epochs, batch_size):
    return _train_sequence_torch_model(
        model_name="temporal_1d_cnn",
        model_factory=lambda input_features: TemporalCnn(input_features=input_features),
        train_rows=train_rows,
        test_rows=test_rows,
        target=target,
        output_path=output_path,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
    )


def _train_sequence_gru(train_rows, test_rows, target, output_path, seed, epochs, batch_size):
    return _train_sequence_torch_model(
        model_name="temporal_gru",
        model_factory=lambda input_features: TemporalGru(input_features=input_features),
        train_rows=train_rows,
        test_rows=test_rows,
        target=target,
        output_path=output_path,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
    )


def _train_sequence_torch_model(model_name, model_factory, train_rows, test_rows, target, output_path, seed, epochs, batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    preprocessor = _preprocessor(scale_numeric=True)
    preprocessor.fit(train_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X_train, y_train, train_patient_ids = _sequence_tensor(train_rows, target, preprocessor)
    X_test, y_test, test_patient_ids = _sequence_tensor(test_rows, target, preprocessor)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    model = model_factory(X_train.shape[2])
    positive_weight = _positive_class_weight(y_train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        test_probs = _predict_cnn(model, X_test)
        history.append({
            "epoch": int(epoch),
            "train_loss": round(float(np.mean(losses)), 4),
            **_binary_metrics(y_test.astype(int), test_probs),
        })

    probabilities = _predict_cnn(model, X_test)
    prediction_frame = pd.DataFrame({
        "patient_id": test_patient_ids,
        "actual_label": y_test.astype(int),
        f"{model_name}_probability": np.round(probabilities, 6),
        f"{model_name}_predicted_label": (probabilities >= 0.5).astype(int),
    })
    metrics = {
        **_binary_metrics(y_test.astype(int), probabilities, prefix="patient_level_"),
        "model_type": f"patient_sequence_{model_name}",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "history": history,
        "learning_curve": history,
        "false_negative_examples": _false_negative_examples(prediction_frame, model_name),
        "temporal_saliency_examples": _temporal_saliency_examples(model, X_test, test_patient_ids),
    }
    artifact_path = output_path / f"{model_name}_{target}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_features": X_train.shape[2],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": target,
        "model_name": model_name,
    }, artifact_path)
    joblib.dump(preprocessor, output_path / f"{model_name}_preprocessor_{target}.joblib")

    return {
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "predictions": prediction_frame,
    }


def _preprocessor(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)


def _base_patient_prediction_rows(test_rows, target):
    return (
        test_rows.groupby("patient_id", as_index=False)[target]
        .max()
        .rename(columns={target: "actual_label"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def _aggregate_patient_predictions(test_rows, target, probabilities, model_name):
    rows = test_rows[["patient_id", target]].copy()
    rows["probability"] = probabilities
    grouped = (
        rows.groupby("patient_id")
        .agg(actual_label=(target, "max"), probability=("probability", "mean"))
        .reset_index()
        .rename(columns={"probability": f"{model_name}_probability"})
    )
    grouped[f"{model_name}_probability"] = grouped[f"{model_name}_probability"].round(6)
    grouped[f"{model_name}_predicted_label"] = (grouped[f"{model_name}_probability"] >= 0.5).astype(int)
    return grouped


def _base_patient_regression_rows(test_rows, target):
    last_rows = (
        test_rows.sort_values(["patient_id", "cycle"])
        .groupby("patient_id", as_index=False)
        .tail(1)
    )
    return (
        last_rows[["patient_id", target]]
        .rename(columns={target: "actual_response_score_percent"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def _aggregate_patient_regression_predictions(test_rows, target, predictions, model_name):
    rows = test_rows[["patient_id", "cycle", target]].copy()
    rows["prediction"] = predictions
    last_rows = (
        rows.sort_values(["patient_id", "cycle"])
        .groupby("patient_id", as_index=False)
        .tail(1)
        .rename(columns={target: "actual_response_score_percent", "prediction": f"{model_name}_response_score_percent"})
    )
    last_rows[f"{model_name}_response_score_percent"] = last_rows[f"{model_name}_response_score_percent"].round(3)
    return (
        last_rows[["patient_id", "actual_response_score_percent", f"{model_name}_response_score_percent"]]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


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


def _binary_metrics(labels, probabilities, prefix=""):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    labels = np.asarray(labels).astype(int)
    tn, fp, fn, tp = _confusion_counts(labels, predictions)
    metrics = {
        f"{prefix}accuracy": round(float(accuracy_score(labels, predictions)), 3),
        f"{prefix}balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 3),
        f"{prefix}f1": round(float(f1_score(labels, predictions, zero_division=0)), 3),
        f"{prefix}precision": round(float(precision_score(labels, predictions, zero_division=0)), 3),
        f"{prefix}sensitivity": round(float(recall_score(labels, predictions, zero_division=0)), 3),
        f"{prefix}specificity": round(float(tn / (tn + fp)), 3) if (tn + fp) else None,
        f"{prefix}brier_score": round(float(brier_score_loss(labels, probabilities)), 3),
        f"{prefix}calibration": _probability_calibration_diagnostics(labels, probabilities),
        f"{prefix}confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    if len(set(labels.tolist())) > 1:
        metrics[f"{prefix}roc_auc"] = round(float(roc_auc_score(labels, probabilities)), 3)
        metrics[f"{prefix}average_precision"] = round(float(average_precision_score(labels, probabilities)), 3)
    else:
        metrics[f"{prefix}roc_auc"] = None
        metrics[f"{prefix}average_precision"] = None
    return metrics


def _probability_calibration_diagnostics(labels, probabilities):
    labels = np.asarray(labels).astype(int)
    probabilities = np.clip(np.asarray(probabilities).astype(float), 1e-5, 1 - 1e-5)
    before = {
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 4),
        "ece": _expected_calibration_error(labels, probabilities),
    }
    best_temperature = 1.0
    best_brier = before["brier_score"]
    best_probs = probabilities
    logits = np.log(probabilities / (1 - probabilities))
    for temperature in np.linspace(0.5, 3.0, 26):
        scaled = 1 / (1 + np.exp(-(logits / temperature)))
        brier = float(brier_score_loss(labels, scaled))
        if brier < best_brier:
            best_brier = brier
            best_temperature = float(temperature)
            best_probs = scaled
    return {
        "before_temperature_scaling": before,
        "after_temperature_scaling": {
            "temperature": round(best_temperature, 3),
            "brier_score": round(best_brier, 4),
            "ece": _expected_calibration_error(labels, best_probs),
        },
        "method": "posthoc_temperature_grid_on_evaluation_split",
    }


def _expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        if high == 1:
            mask = (probabilities >= low) & (probabilities <= high)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += (float(np.mean(mask)) * abs(accuracy - confidence))
    return round(ece, 4)


def _regression_metrics(labels, predictions, prefix=""):
    labels = np.asarray(labels).astype(float)
    predictions = np.asarray(predictions).astype(float)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return {
        f"{prefix}mae": round(float(mean_absolute_error(labels, predictions)), 3),
        f"{prefix}rmse": round(float(rmse), 3),
        f"{prefix}r2": round(float(r2_score(labels, predictions)), 3) if len(labels) >= 2 else None,
    }


def _confusion_counts(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _sequence_tensor(rows, target, preprocessor):
    rows = rows.sort_values(["patient_id", "cycle"]).copy()
    max_cycles = int(rows["cycle"].max())
    transformed = preprocessor.transform(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = transformed.astype(np.float32)

    feature_dim = transformed.shape[1]
    rows = rows.reset_index(drop=True)
    patient_ids = rows["patient_id"].drop_duplicates().tolist()
    sequences = np.zeros((len(patient_ids), max_cycles, feature_dim), dtype=np.float32)
    labels = np.zeros(len(patient_ids), dtype=np.float32)
    patient_index = {patient_id: idx for idx, patient_id in enumerate(patient_ids)}
    for row_idx, row in rows.iterrows():
        seq_idx = patient_index[row["patient_id"]]
        cycle_idx = int(row["cycle"]) - 1
        sequences[seq_idx, cycle_idx, :] = transformed[row_idx]
        labels[seq_idx] = max(labels[seq_idx], int(row[target]))
    return sequences, labels, patient_ids


class BaselineTemporalCnn(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_features, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.classifier(self.encoder(x))


class TemporalCnn(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.classifier(self.encoder(x))


class TemporalGru(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


def _positive_class_weight(labels):
    positives = max(float(np.sum(labels == 1)), 1.0)
    negatives = max(float(np.sum(labels == 0)), 1.0)
    return torch.tensor([negatives / positives], dtype=torch.float32)


def _predict_cnn(model, sequences):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(sequences)).squeeze(1)
        return torch.sigmoid(logits).numpy()


def _false_negative_examples(predictions, model_name, limit=10):
    probability_col = f"{model_name}_probability"
    label_col = f"{model_name}_predicted_label"
    if probability_col not in predictions.columns or label_col not in predictions.columns:
        return []
    rows = predictions[
        (predictions["actual_label"].astype(int) == 1)
        & (predictions[label_col].astype(int) == 0)
    ].copy()
    rows = rows.sort_values(probability_col, ascending=False).head(limit)
    return [
        {
            "patient_id": row["patient_id"],
            "actual_label": int(row["actual_label"]),
            "predicted_probability": round(float(row[probability_col]), 6),
            "review_note": "Synthetic false-negative example for error analysis; not a clinical miss.",
        }
        for _, row in rows.iterrows()
    ]


def _temporal_saliency_examples(model, sequences, patient_ids, limit=5):
    if len(sequences) == 0:
        return []
    model.eval()
    tensor = torch.from_numpy(sequences[:limit]).clone().detach().requires_grad_(True)
    logits = model(tensor).squeeze(1)
    logits.sum().backward()
    saliency = (tensor.grad.detach().abs() * tensor.detach().abs()).sum(dim=2).numpy()
    examples = []
    for index, patient_id in enumerate(patient_ids[:limit]):
        cycle_scores = saliency[index]
        total = float(np.sum(cycle_scores)) or 1.0
        examples.append({
            "patient_id": patient_id,
            "cycle_saliency": [
                {
                    "cycle": int(cycle + 1),
                    "relative_saliency": round(float(score / total), 4),
                }
                for cycle, score in enumerate(cycle_scores)
            ],
            "method": "absolute gradient times input aggregated by treatment cycle",
            "safety": "Simple temporal model-behavior explanation on synthetic data, not clinical causality.",
        })
    return examples


def _dl_experiment_report(models):
    sequence_names = [name for name in ["temporal_baseline_cnn", "temporal_1d_cnn", "temporal_gru"] if name in models]
    return {
        "implemented": {
            "baseline_cnn": "temporal_baseline_cnn" in models,
            "regularized_cnn": "temporal_1d_cnn" in models,
            "recurrent_sequence_baseline": "temporal_gru" in models,
            "augmentation_experiment": "synthetic generator applies noise and missingness; image augmentation remains in BreastDCEDL CNN path.",
            "learning_curves": {name: bool(models[name].get("learning_curve")) for name in sequence_names},
            "confusion_and_error_examples": {name: bool(models[name].get("false_negative_examples") is not None) for name in sequence_names},
            "calibration_before_after_temperature_scaling": True,
            "simple_visual_explanation": {name: "temporal_saliency_examples" in models[name] for name in sequence_names},
        },
        "not_overclaimed": {
            "transfer_learning_baseline": "Not run for the temporal tabular CNN. Use the BreastDCEDL imaging CNN endpoint for image-transfer experiments when pretrained weights/data are available.",
            "grad_cam": "Not applicable to the temporal tabular CNN; temporal saliency is provided instead. Use Grad-CAM only for image CNN experiments.",
        },
        "claim_boundary": "These are synthetic-data ML discipline checks and do not validate clinical response prediction.",
    }
