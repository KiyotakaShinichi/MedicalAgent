import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_DIR = "Data/complete_synthetic_training"

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
    rows = pd.read_csv(ml_csv_path)
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
        all_models["temporal_1d_cnn"] = cnn_results["metrics"]
        all_models["temporal_gru"] = gru_results["metrics"]
        artifacts["temporal_1d_cnn"] = cnn_results["artifact_path"]
        artifacts["temporal_gru"] = gru_results["artifact_path"]
        predictions = predictions.merge(
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

    best_model = max(
        all_models,
        key=lambda name: (
            all_models[name].get("patient_level_roc_auc")
            if all_models[name].get("patient_level_roc_auc") is not None
            else all_models[name].get("roc_auc", -1)
        ),
    )
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
        },
        "models": all_models,
        "best_model_by_patient_level_roc_auc": best_model,
        "artifacts": {
            **artifacts,
            "predictions_csv": str(output_path / "complete_synthetic_model_predictions.csv"),
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
    metrics = {
        **_binary_metrics(y_test.astype(int), probabilities, prefix="patient_level_"),
        "model_type": f"patient_sequence_{model_name}",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "history": history,
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

    predictions = pd.DataFrame({
        "patient_id": test_patient_ids,
        "actual_label": y_test.astype(int),
        f"{model_name}_probability": np.round(probabilities, 6),
        f"{model_name}_predicted_label": (probabilities >= 0.5).astype(int),
    })
    return {
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "predictions": predictions,
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


def _binary_metrics(labels, probabilities, prefix=""):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    labels = np.asarray(labels).astype(int)
    metrics = {
        f"{prefix}accuracy": round(float(accuracy_score(labels, predictions)), 3),
        f"{prefix}balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 3),
        f"{prefix}f1": round(float(f1_score(labels, predictions, zero_division=0)), 3),
    }
    if len(set(labels.tolist())) > 1:
        metrics[f"{prefix}roc_auc"] = round(float(roc_auc_score(labels, probabilities)), 3)
    else:
        metrics[f"{prefix}roc_auc"] = None
    return metrics


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
