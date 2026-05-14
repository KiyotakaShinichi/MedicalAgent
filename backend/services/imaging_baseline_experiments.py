from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ULTRASOUND_OUTPUT_PATH = "Data/public_imaging/ultrasound_baseline/metrics.json"
DEFAULT_ULTRASOUND_PREDICTIONS_PATH = "Data/public_imaging/ultrasound_baseline/predictions.csv"
DEFAULT_CT_WORKFLOW_PATH = "Data/public_imaging/ct_lesion_workflow/report.json"

ULTRASOUND_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def run_ultrasound_baseline(
    dataset_root: str = "Datasets/BUSI",
    output_path: str = DEFAULT_ULTRASOUND_OUTPUT_PATH,
    predictions_path: str = DEFAULT_ULTRASOUND_PREDICTIONS_PATH,
    max_images: int = 600,
) -> dict[str, Any]:
    root = Path(dataset_root)
    rows = _collect_ultrasound_rows(root, max_images=max_images)
    if len(rows) < 20 or len({row["label"] for row in rows}) < 2:
        payload = _unavailable_payload(
            task="breast_ultrasound_baseline",
            reason="Dataset is missing, too small, or has fewer than two labels.",
            expected_layout="Put BUSI images under Datasets/BUSI with benign/malignant/normal folders.",
        )
        _write_json(output_path, payload)
        return payload

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        payload = _unavailable_payload(
            task="breast_ultrasound_baseline",
            reason=f"Required ML dependency is unavailable: {exc}",
            expected_layout="Install scikit-learn and rerun.",
        )
        _write_json(output_path, payload)
        return payload

    features = np.array([row["features"] for row in rows], dtype=np.float32)
    labels = np.array([row["label"] for row in rows])
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    indices = np.arange(len(rows))

    stratify = y if min(np.bincount(y)) >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = {}
    prediction_rows = []
    for name, model in models.items():
        model.fit(features[train_idx], y[train_idx])
        probabilities = _predict_proba_aligned(model, features[test_idx], len(encoder.classes_))
        predictions = np.argmax(probabilities, axis=1)
        metrics = _classification_metrics(y[test_idx], predictions, probabilities, list(encoder.classes_))
        results[name] = metrics

        for row_index, actual, pred, probs in zip(test_idx, y[test_idx], predictions, probabilities):
            prediction_rows.append({
                "model": name,
                "path": rows[int(row_index)]["path"],
                "actual_label": str(encoder.inverse_transform([int(actual)])[0]),
                "predicted_label": str(encoder.inverse_transform([int(pred)])[0]),
                "correct": bool(int(actual) == int(pred)),
                "max_probability": round(float(np.max(probs)), 6),
            })

    best_model = max(
        results,
        key=lambda model_name: (
            results[model_name].get("balanced_accuracy") or 0,
            results[model_name].get("macro_f1") or 0,
        ),
    )
    payload = {
        "schema_version": "public_ultrasound_baseline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dataset_root": str(root),
        "task": "breast_ultrasound_image_classification_baseline",
        "model_family": "handcrafted_image_features_plus_sklearn_baselines",
        "image_count": len(rows),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "label_counts": _label_counts(labels),
        "feature_names": _feature_names(),
        "models": results,
        "best_model": best_model,
        "predictions_path": predictions_path,
        "claim_boundary": (
            "This is an engineering baseline on public ultrasound images. "
            "It is not a diagnostic or clinical validation model."
        ),
    }
    _write_json(output_path, payload)
    _write_predictions(predictions_path, prediction_rows)
    return payload


def build_ct_lesion_workflow_report(
    dataset_root: str = "Datasets/DeepLesion",
    output_path: str = DEFAULT_CT_WORKFLOW_PATH,
) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.exists():
        payload = _unavailable_payload(
            task="ct_lesion_workflow",
            reason="DeepLesion or PET/CT lesion dataset not found locally.",
            expected_layout="Place DeepLesion under Datasets/DeepLesion or PET/CT DICOM/NIfTI data under Datasets/FDG-PET-CT-Lesions.",
        )
        _write_json(output_path, payload)
        return payload

    files = [path for path in root.rglob("*") if path.is_file()]
    image_files = [path for path in files if _is_ct_like(path)]
    metadata_files = [path for path in files if path.suffix.lower() in {".csv", ".tsv", ".xlsx", ".json"}]
    annotation_rows = _count_annotation_rows(metadata_files)
    payload = {
        "schema_version": "ct_lesion_workflow_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "workflow_ready" if image_files or metadata_files else "no_ct_files_found",
        "dataset_root": str(root),
        "image_file_count": len(image_files),
        "metadata_file_count": len(metadata_files),
        "estimated_annotation_rows": annotation_rows,
        "sample_image_files": [str(path.relative_to(root)) for path in image_files[:10]],
        "sample_metadata_files": [str(path.relative_to(root)) for path in metadata_files[:10]],
        "recommended_model_track": [
            "Start with lesion localization/segmentation only.",
            "Do not infer metastatic breast cancer origin from generic CT lesion labels.",
            "Use report-text metastatic indicators as the patient-facing safety route until labels are curated.",
        ],
        "claim_boundary": (
            "CT/PET-CT support is currently workflow and metadata readiness. "
            "No diagnostic CT reader is claimed."
        ),
    }
    _write_json(output_path, payload)
    return payload


def _collect_ultrasound_rows(root: Path, max_images: int) -> list[dict[str, Any]]:
    if not root.exists():
        return []

    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ULTRASOUND_IMAGE_SUFFIXES:
            continue
        if _looks_like_mask(path):
            continue
        label = _label_from_path(path)
        if not label:
            continue
        try:
            features = _image_features(path)
        except Exception:
            continue
        rows.append({"path": str(path), "label": label, "features": features})
        if len(rows) >= max_images:
            break
    return rows


def _image_features(path: Path) -> list[float]:
    from PIL import Image

    image = Image.open(path).convert("L").resize((128, 128))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx**2 + gy**2)
    hist, _ = np.histogram(arr, bins=16, range=(0, 1), density=True)
    return [
        float(arr.mean()),
        float(arr.std()),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 90)),
        float(grad.mean()),
        float(grad.std()),
        float((arr > 0.7).mean()),
        float((arr < 0.2).mean()),
        *[float(value) for value in hist],
    ]


def _feature_names() -> list[str]:
    return [
        "mean_intensity",
        "std_intensity",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "mean_gradient",
        "std_gradient",
        "bright_fraction",
        "dark_fraction",
        *[f"hist_bin_{index:02d}" for index in range(16)],
    ]


def _classification_metrics(y_true, y_pred, probabilities, classes: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
        "classes": classes,
    }
    try:
        if len(classes) == 2:
            metrics["auroc"] = round(float(roc_auc_score(y_true, probabilities[:, 1])), 4)
        else:
            metrics["auroc_ovr"] = round(float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")), 4)
    except Exception:
        metrics["auroc"] = None
    return metrics


def _predict_proba_aligned(model, x, n_classes: int) -> np.ndarray:
    probs = model.predict_proba(x)
    if probs.shape[1] == n_classes:
        return probs
    aligned = np.zeros((len(x), n_classes), dtype=np.float32)
    for source_index, class_index in enumerate(getattr(model, "classes_", range(probs.shape[1]))):
        aligned[:, int(class_index)] = probs[:, source_index]
    return aligned


def _label_from_path(path: Path) -> str | None:
    lower = str(path).lower()
    for label in ("benign", "malignant", "normal"):
        if label in lower:
            return label
    return None


def _looks_like_mask(path: Path) -> bool:
    lower = path.stem.lower()
    return any(term in lower for term in ("mask", "segmentation", "seg", "label"))


def _is_ct_like(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii.gz") or path.suffix.lower() in {".dcm", ".nii", ".npy", ".npz", ".nrrd", ".png", ".jpg", ".jpeg"}


def _count_annotation_rows(metadata_files: list[Path]) -> int | None:
    total = 0
    found = False
    for path in metadata_files[:20]:
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                total += max(sum(1 for _ in csv.reader(handle, delimiter=delimiter)) - 1, 0)
                found = True
        except Exception:
            continue
    return total if found else None


def _label_counts(labels) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def _unavailable_payload(task: str, reason: str, expected_layout: str) -> dict[str, Any]:
    return {
        "schema_version": f"{task}_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "reason": reason,
        "expected_layout": expected_layout,
        "claim_boundary": "Unavailable artifacts are explicit so the dashboard does not imply hidden validation.",
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_predictions(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
