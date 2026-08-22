"""Temporal (PyTorch) sequence models and their shared training loop.

Holds the three architectures, the tensor construction that turns per-patient
row sequences into padded batches, the training loop, the saliency and
false-negative example extractors, and the deep-learning experiment report.

``_train_sequence_torch_model`` seeds ``torch`` then ``numpy`` at entry. That
call order is load-bearing for reproducibility and is preserved verbatim.
"""

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.services.complete_synthetic_training.data_preparation import _preprocessor
from backend.services.complete_synthetic_training.feature_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from backend.services.complete_synthetic_training.metrics import _binary_metrics


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
