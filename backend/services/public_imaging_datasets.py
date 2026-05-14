from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = "Data/public_imaging/public_imaging_manifest.json"

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".dcm",
    ".nii",
    ".gz",
    ".npy",
    ".npz",
    ".nrrd",
}
MASK_HINTS = ("mask", "segmentation", "seg", "label", "annotation")


DATASET_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "busi_breast_ultrasound",
        "name": "BUSI Breast Ultrasound",
        "source_url": "https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset/data",
        "expected_roots": ["BUSI", "busi", "Dataset_BUSI_with_GT", "Breast Ultrasound Images Dataset"],
        "task": "breast_ultrasound_segmentation_or_classification",
        "modality": "ultrasound",
        "claim_boundary": "Benign/malignant/normal image experiment only; not diagnosis.",
    },
    {
        "id": "bus_uclm_breast_ultrasound",
        "name": "BUS-UCLM Breast Ultrasound",
        "source_url": "https://www.nature.com/articles/s41597-025-04562-3",
        "expected_roots": ["BUS-UCLM", "bus-uclm", "BUS_UCLM"],
        "task": "breast_ultrasound_segmentation",
        "modality": "ultrasound",
        "claim_boundary": "Segmentation benchmark only; not diagnosis.",
    },
    {
        "id": "nih_deeplesion",
        "name": "NIH DeepLesion",
        "source_url": "https://nihcc.app.box.com/v/DeepLesion",
        "expected_roots": ["DeepLesion", "deeplesion"],
        "task": "ct_lesion_detection_pretraining_or_benchmark",
        "modality": "ct",
        "claim_boundary": "Lesion localization experiment only; not metastatic breast cancer diagnosis.",
    },
    {
        "id": "tcia_fdg_pet_ct_lesions",
        "name": "TCIA FDG-PET-CT-Lesions",
        "source_url": "https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/",
        "expected_roots": ["FDG-PET-CT-Lesions", "fdg_pet_ct_lesions", "FDG_PET_CT_Lesions"],
        "task": "pet_ct_lesion_segmentation_workflow",
        "modality": "pet_ct",
        "claim_boundary": "Whole-body lesion workflow experiment only; not disease-origin diagnosis.",
    },
    {
        "id": "tcia_qin_breast",
        "name": "TCIA QIN-BREAST",
        "source_url": "https://www.cancerimagingarchive.net/collection/qin-breast/",
        "expected_roots": ["QIN-BREAST", "qin_breast", "QIN_BREAST"],
        "task": "breast_pet_ct_mri_response_workflow",
        "modality": "pet_ct_mri",
        "claim_boundary": "Imaging workflow exploration only; not clinical validation.",
    },
]


def build_public_imaging_manifest(
    dataset_root: str = "Datasets",
    output_path: str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    root = Path(dataset_root)
    datasets = [_inspect_candidate(root, candidate) for candidate in DATASET_CANDIDATES]
    available = [item for item in datasets if item["available"]]

    payload = {
        "schema_version": "public_imaging_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "status": "ready_for_experiments" if available else "datasets_not_downloaded",
        "available_dataset_count": len(available),
        "datasets": datasets,
        "recommended_next_task": _recommended_next_task(available),
        "claim_boundary": (
            "Public imaging datasets are used for narrow engineering experiments. "
            "They do not make MedicalAgent a diagnostic CT or ultrasound reader."
        ),
    }
    payload["manifest_hash"] = _stable_hash(payload)
    if output_path:
        _write_json(output_path, payload)
    return payload


def _inspect_candidate(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    found_root = _find_existing_root(root, candidate["expected_roots"])
    base = {
        **candidate,
        "available": found_root is not None,
        "local_path": str(found_root) if found_root else None,
        "file_count": 0,
        "image_count": 0,
        "mask_count": 0,
        "metadata_count": 0,
        "class_counts": {},
        "sample_files": [],
        "readiness": "missing_download",
    }
    if not found_root:
        return base

    files = [path for path in found_root.rglob("*") if path.is_file()]
    image_files = [path for path in files if _is_image_file(path)]
    metadata_files = [path for path in files if path.suffix.lower() in {".csv", ".tsv", ".xlsx", ".json"}]
    masks = [path for path in image_files if _looks_like_mask(path)]
    class_counts = _class_counts(found_root, image_files)
    readiness = _readiness(candidate["id"], image_files, metadata_files, masks, class_counts)

    base.update({
        "file_count": len(files),
        "image_count": len(image_files),
        "mask_count": len(masks),
        "metadata_count": len(metadata_files),
        "class_counts": class_counts,
        "sample_files": [str(path.relative_to(found_root)) for path in image_files[:8]],
        "readiness": readiness,
    })
    return base


def _find_existing_root(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def _is_image_file(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii.gz") or path.suffix.lower() in IMAGE_SUFFIXES


def _looks_like_mask(path: Path) -> bool:
    lower = str(path).lower()
    return any(hint in lower for hint in MASK_HINTS)


def _class_counts(root: Path, image_files: list[Path]) -> dict[str, int]:
    counts = {"benign": 0, "malignant": 0, "normal": 0, "unknown": 0}
    for path in image_files:
        lower = str(path.relative_to(root)).lower()
        label = "unknown"
        for candidate in ("benign", "malignant", "normal"):
            if candidate in lower:
                label = candidate
                break
        counts[label] += 1
    return {key: value for key, value in counts.items() if value}


def _readiness(dataset_id: str, image_files: list[Path], metadata_files: list[Path], masks: list[Path], class_counts: dict[str, int]) -> str:
    if not image_files:
        return "no_images_found"
    if dataset_id in {"busi_breast_ultrasound", "bus_uclm_breast_ultrasound"}:
        label_count = len([key for key in ("benign", "malignant", "normal") if class_counts.get(key, 0) > 0])
        if label_count >= 2:
            return "classification_ready" if not masks else "classification_and_segmentation_ready"
        return "images_found_labels_unclear"
    if metadata_files:
        return "image_metadata_workflow_ready"
    return "images_found_metadata_missing"


def _recommended_next_task(available: list[dict[str, Any]]) -> str:
    ids = {item["id"] for item in available}
    if "busi_breast_ultrasound" in ids:
        return "Run ultrasound baseline: python scripts/run_ultrasound_baseline.py --dataset-root Datasets/BUSI"
    if "bus_uclm_breast_ultrasound" in ids:
        return "Run ultrasound baseline against BUS-UCLM for cross-dataset robustness."
    if "nih_deeplesion" in ids:
        return "Build CT lesion workflow report: python scripts/run_ct_lesion_workflow.py --dataset-root Datasets/DeepLesion"
    return "Download one public dataset into Datasets/ and rebuild this manifest."


def _stable_hash(payload: dict[str, Any]) -> str:
    material = json.dumps(
        {key: value for key, value in payload.items() if key not in {"generated_at", "manifest_hash"}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
