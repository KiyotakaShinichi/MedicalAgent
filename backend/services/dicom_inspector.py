from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def inspect_dicom_tree(root_path, patient_id=None, preview_dir="Data/dicom_previews", max_files=None):
    root = Path(root_path)
    if patient_id and (root / patient_id).is_dir():
        root = root / patient_id
    files = list(root.rglob("*"))
    dicom_files = [path for path in files if path.is_file()]
    if max_files:
        dicom_files = dicom_files[:max_files]

    series = {}
    grouped = defaultdict(list)

    for path in dicom_files:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue

        subject = str(getattr(dataset, "PatientID", "")) or _subject_from_path(path)
        if patient_id and subject != patient_id:
            continue

        series_uid = str(getattr(dataset, "SeriesInstanceUID", path.parent.name))
        grouped[(subject, series_uid)].append(path)

        if (subject, series_uid) not in series:
            series[(subject, series_uid)] = {
                "patient_id": subject,
                "study_date": str(getattr(dataset, "StudyDate", "")),
                "modality": str(getattr(dataset, "Modality", "")),
                "series_description": str(getattr(dataset, "SeriesDescription", "")),
                "series_uid": series_uid,
                "folder": str(path.parent),
                "instances": 0,
            }

    for key, paths in grouped.items():
        series[key]["instances"] = len(paths)

    summaries = sorted(series.values(), key=lambda item: (item["patient_id"], item["study_date"], item["series_description"]))
    preview = _write_preview(grouped, series, preview_dir)

    return {
        "root_path": str(root),
        "dicom_files_scanned": len(dicom_files),
        "series_count": len(summaries),
        "series": summaries,
        "preview": preview,
    }


def _subject_from_path(path):
    for part in path.parts:
        if part.startswith("QIN-BREAST-02-"):
            return part
    return ""


def _write_preview(grouped, series, preview_dir):
    if not grouped:
        return None

    best_key = max(grouped, key=lambda key: len(grouped[key]))
    paths = sorted(grouped[best_key])
    middle_path = paths[len(paths) // 2]

    try:
        dataset = pydicom.dcmread(middle_path, force=True)
        pixels = dataset.pixel_array.astype(float)
    except Exception:
        return None

    if pixels.ndim > 2:
        pixels = pixels[0]

    pixels = pixels - np.min(pixels)
    max_value = np.max(pixels)
    if max_value > 0:
        pixels = pixels / max_value
    pixels = (pixels * 255).astype(np.uint8)

    output_dir = Path(preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject, series_uid = best_key
    output_path = output_dir / f"{subject}_{series_uid[:12]}_preview.png"
    Image.fromarray(pixels).save(output_path)

    return {
        "patient_id": subject,
        "series_description": series[best_key]["series_description"],
        "source_file": str(middle_path),
        "preview_path": str(output_path),
    }
