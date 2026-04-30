import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


def run_breastdcedl_small_cnn(
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv",
    metrics_json_path: str = "Data/breastdcedl_spy1_cnn_metrics.json",
    max_patients: int = 120,
    epochs: int = 4,
    batch_size: int = 8,
):
    torch.manual_seed(42)
    np.random.seed(42)

    dataset_rows = _eligible_rows(manifest_csv_path, max_patients=max_patients)
    train_rows, val_rows = train_test_split(
        dataset_rows,
        test_size=0.25,
        random_state=42,
        stratify=dataset_rows["pcr_label"].astype(int),
    )

    train_loader = DataLoader(BreastDCEDLSliceDataset(train_rows), batch_size=batch_size, shuffle=True)
    val_dataset = BreastDCEDLSliceDataset(val_rows)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SmallDceCnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_positive_class_weight(train_rows))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels.float())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = _evaluate(model, val_loader)
        history.append({
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4) if losses else None,
            **val_metrics,
        })

    metrics = {
        "rows": int(len(dataset_rows)),
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(val_rows)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "final_validation": history[-1],
        "history": history,
        "model_type": "small_2d_cnn_dce_slice",
        "warning": "Exploratory PoC only. CPU-trained small CNN, not clinically validated.",
    }

    output_path = Path(metrics_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


class BreastDCEDLSliceDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = _build_three_channel_slice(row)
        label = int(row["pcr_label"])
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.float32)


class SmallDceCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(48, 1)

    def forward(self, x):
        encoded = self.encoder(x).flatten(1)
        return self.classifier(encoded)


def _eligible_rows(manifest_csv_path, max_patients):
    manifest = pd.read_csv(manifest_csv_path)
    rows = manifest[
        (manifest["has_core_dce_inputs"] == True)
        & (manifest["has_mask"] == True)
        & (manifest["pcr_label"].notna())
    ].copy()
    rows["pcr_label"] = rows["pcr_label"].astype(int)
    if max_patients:
        rows = rows.head(max_patients)
    if rows["pcr_label"].nunique() < 2:
        raise ValueError("CNN baseline needs both pCR classes")
    return rows


def _build_three_channel_slice(row):
    acq0 = _load_volume(row["acq0_path"])
    acq1 = _load_volume(row["acq1_path"])
    acq2 = _load_volume(row["acq2_path"])
    mask = _load_volume(row["mask_path"]) > 0
    z_index = int(np.argmax(np.sum(mask, axis=(0, 1))))

    slices = [
        _resize_slice(_normalize_slice(acq0[:, :, z_index])),
        _resize_slice(_normalize_slice(acq1[:, :, z_index])),
        _resize_slice(_normalize_slice(acq2[:, :, z_index])),
    ]
    image = np.stack(slices, axis=0).astype(np.float32)
    return image


def _load_volume(path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def _normalize_slice(slice_array):
    low, high = np.percentile(slice_array, [1, 99])
    data = np.clip(slice_array, low, high)
    data = data - np.min(data)
    max_value = np.max(data)
    if max_value > 0:
        data = data / max_value
    return data


def _resize_slice(slice_array, size=128):
    image = Image.fromarray((slice_array * 255).astype(np.uint8))
    image = image.resize((size, size))
    return np.asarray(image).astype(np.float32) / 255.0


def _positive_class_weight(rows):
    y = rows["pcr_label"].astype(int)
    positives = max(int(y.sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return torch.tensor([negatives / positives], dtype=torch.float32)


def _evaluate(model, val_loader):
    model.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for images, batch_labels in val_loader:
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            labels.extend(batch_labels.numpy().astype(int).tolist())
            probabilities.extend(probs.numpy().tolist())

    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    metrics = {
        "val_accuracy": round(float(accuracy_score(labels, predictions)), 3),
        "val_balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 3),
    }
    if len(set(labels)) > 1:
        metrics["val_roc_auc"] = round(float(roc_auc_score(labels, probabilities)), 3)
    else:
        metrics["val_roc_auc"] = None
    return metrics
