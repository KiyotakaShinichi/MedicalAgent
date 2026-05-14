from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.services.imaging_baseline_experiments import _collect_ultrasound_rows


DEFAULT_OUTPUT_PATH = "Data/public_imaging/ultrasound_transfer_baseline/metrics.json"


def run_ultrasound_transfer_baseline(
    dataset_root: str = "Datasets/BUSI",
    output_path: str = DEFAULT_OUTPUT_PATH,
    max_images: int = 60,
    epochs: int = 1,
    batch_size: int = 16,
    pretrained: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root)
    rows = _balanced_ultrasound_rows(root, max_images=max_images)
    labels = sorted({row["label"] for row in rows})
    if len(rows) < 30 or len(labels) < 2:
        payload = _unavailable(
            "Dataset is missing, too small, or has fewer than two labels.",
            "Put BUSI images under Datasets/BUSI with benign/malignant/normal folders.",
        )
        _write_json(output_path, payload)
        return payload

    try:
        import torch
        from PIL import Image
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except Exception as exc:
        payload = _unavailable(f"Required transfer-learning dependency is unavailable: {exc}", "Install torch, torchvision, pillow, and scikit-learn.")
        _write_json(output_path, payload)
        return payload

    encoder = LabelEncoder()
    y = encoder.fit_transform([row["label"] for row in rows])
    indices = np.arange(len(rows))
    stratify = y if min(np.bincount(y)) >= 2 else None
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42, stratify=stratify)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = _ImageDataset([rows[int(i)] for i in train_idx], encoder, transform, Image)
    test_ds = _ImageDataset([rows[int(i)] for i in test_idx], encoder, transform, Image)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    weights = None
    model_family = "mobilenet_v3_small_random_init"
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model_family = "mobilenet_v3_small_imagenet_transfer"
        except Exception:
            weights = None
    model = models.mobilenet_v3_small(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(encoder.classes_))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    model.train()
    for _ in range(max(epochs, 1)):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            actuals.extend(batch_y.numpy().tolist())

    payload = {
        "schema_version": "public_ultrasound_transfer_baseline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dataset_root": str(root),
        "task": "breast_ultrasound_transfer_learning_classification_baseline",
        "model_family": model_family,
        "pretrained_requested": bool(pretrained),
        "device": str(device),
        "image_count": len(rows),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "epochs": int(epochs),
        "classes": list(encoder.classes_),
        "balanced_accuracy": round(float(balanced_accuracy_score(actuals, preds)), 4),
        "macro_f1": round(float(f1_score(actuals, preds, average="macro")), 4),
        "confusion_matrix": confusion_matrix(actuals, preds).astype(int).tolist(),
        "final_train_loss": round(float(losses[-1]), 6) if losses else None,
        "claim_boundary": (
            "Transfer-learning baseline for public BUSI engineering experiments only. "
            "It is not a diagnostic model and does not validate clinical deployment."
        ),
    }
    _write_json(output_path, payload)
    return payload


class _ImageDataset:
    def __init__(self, rows, encoder, transform, image_module):
        self.rows = rows
        self.encoder = encoder
        self.transform = transform
        self.image_module = image_module

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self.image_module.open(row["path"]).convert("L")
        return self.transform(image), int(self.encoder.transform([row["label"]])[0])


def _unavailable(reason: str, expected_layout: str) -> dict[str, Any]:
    return {
        "schema_version": "public_ultrasound_transfer_baseline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "reason": reason,
        "expected_layout": expected_layout,
        "claim_boundary": "Unavailable artifacts are explicit so the dashboard does not imply hidden validation.",
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _balanced_ultrasound_rows(root: Path, max_images: int) -> list[dict[str, Any]]:
    candidates = _collect_ultrasound_rows(root, max_images=max(max_images * 6, 1200))
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_label.setdefault(row["label"], []).append(row)
    if len(by_label) < 2:
        return candidates[:max_images]
    per_label = max(max_images // len(by_label), 1)
    rows: list[dict[str, Any]] = []
    for label in sorted(by_label):
        rows.extend(by_label[label][:per_label])
    if len(rows) < max_images:
        used = {id(row) for row in rows}
        for row in candidates:
            if id(row) not in used:
                rows.append(row)
            if len(rows) >= max_images:
                break
    return rows[:max_images]
