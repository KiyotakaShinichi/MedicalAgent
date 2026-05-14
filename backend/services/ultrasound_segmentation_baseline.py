from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_OUTPUT_PATH = "Data/public_imaging/ultrasound_segmentation_baseline/metrics.json"
ULTRASOUND_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def run_ultrasound_segmentation_baseline(
    dataset_root: str = "Datasets/BUSI",
    output_path: str = DEFAULT_OUTPUT_PATH,
    max_pairs: int = 500,
    image_size: int = 256,
) -> dict[str, Any]:
    root = Path(dataset_root)
    pairs = _collect_pairs(root, max_pairs=max_pairs)
    if len(pairs) < 10:
        payload = _unavailable(
            "Dataset is missing or fewer than 10 image/mask pairs were found.",
            "Expected BUSI image files plus corresponding mask files under Datasets/BUSI.",
        )
        _write_json(output_path, payload)
        return payload

    rows = []
    for image_path, mask_path in pairs:
        image = _load_gray(image_path, image_size)
        truth = _load_mask(mask_path, image_size)
        threshold = _otsu_threshold(image)
        pred_dark = image <= threshold
        pred_bright = image >= threshold
        dice_dark = _dice(pred_dark, truth)
        dice_bright = _dice(pred_bright, truth)
        pred = pred_dark if dice_dark >= dice_bright else pred_bright
        rows.append({
            "image": str(image_path),
            "mask": str(mask_path),
            "label": _label_from_path(image_path),
            "dice": _dice(pred, truth),
            "iou": _iou(pred, truth),
            "foreground_rate": float(pred.mean()),
        })

    payload = {
        "schema_version": "public_ultrasound_segmentation_baseline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dataset_root": str(root),
        "task": "breast_ultrasound_mask_segmentation_baseline",
        "model_family": "classical_otsu_threshold_baseline",
        "pair_count": len(rows),
        "mean_dice": round(float(np.mean([row["dice"] for row in rows])), 4),
        "median_dice": round(float(np.median([row["dice"] for row in rows])), 4),
        "mean_iou": round(float(np.mean([row["iou"] for row in rows])), 4),
        "label_breakdown": _label_breakdown(rows),
        "sample_rows": rows[:20],
        "claim_boundary": (
            "Classical mask baseline for engineering comparison only. It is not a lesion detector, "
            "not a radiology model, and not validated for clinical use."
        ),
    }
    _write_json(output_path, payload)
    return payload


def _collect_pairs(root: Path, max_pairs: int) -> list[tuple[Path, Path]]:
    if not root.exists():
        return []
    masks = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in ULTRASOUND_SUFFIXES and _looks_like_mask(path)]
    pairs = []
    for mask in masks:
        image = _matching_image(mask)
        if image and image.exists():
            pairs.append((image, mask))
        if len(pairs) >= max_pairs:
            break
    return pairs


def _matching_image(mask: Path) -> Path | None:
    stem = mask.stem.lower()
    candidates = [
        stem.replace("_mask", ""),
        stem.replace(" mask", ""),
        stem.replace("-mask", ""),
        stem.replace("_segmentation", ""),
    ]
    for candidate_stem in candidates:
        for suffix in ULTRASOUND_SUFFIXES:
            candidate = mask.with_name(candidate_stem + suffix)
            if candidate.exists() and not _looks_like_mask(candidate):
                return candidate
    return None


def _load_gray(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_mask(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size))
    return np.asarray(image, dtype=np.float32) > 0.5


def _otsu_threshold(arr: np.ndarray) -> float:
    hist, edges = np.histogram(arr.ravel(), bins=128, range=(0.0, 1.0))
    total = arr.size
    sum_total = float(np.dot(hist, edges[:-1]))
    weight_bg = 0.0
    sum_bg = 0.0
    best_score = -1.0
    best_threshold = 0.5
    for idx, count in enumerate(hist):
        weight_bg += count
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += edges[idx] * count
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        score = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if score > best_score:
            best_score = score
            best_threshold = float(edges[idx])
    return best_threshold


def _dice(pred: np.ndarray, truth: np.ndarray) -> float:
    intersection = float(np.logical_and(pred, truth).sum())
    denom = float(pred.sum() + truth.sum())
    return 1.0 if denom == 0 else (2.0 * intersection / denom)


def _iou(pred: np.ndarray, truth: np.ndarray) -> float:
    intersection = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    return 1.0 if union == 0 else (intersection / union)


def _looks_like_mask(path: Path) -> bool:
    stem = path.stem.lower()
    return any(term in stem for term in ("mask", "segmentation", "seg", "label"))


def _label_from_path(path: Path) -> str:
    lower = str(path).lower()
    for label in ("benign", "malignant", "normal"):
        if label in lower:
            return label
    return "unknown"


def _label_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for label in sorted({row["label"] for row in rows}):
        subset = [row for row in rows if row["label"] == label]
        output[label] = {
            "count": len(subset),
            "mean_dice": round(float(np.mean([row["dice"] for row in subset])), 4),
            "mean_iou": round(float(np.mean([row["iou"] for row in subset])), 4),
        }
    return output


def _unavailable(reason: str, expected_layout: str) -> dict[str, Any]:
    return {
        "schema_version": "public_ultrasound_segmentation_baseline_v1",
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
