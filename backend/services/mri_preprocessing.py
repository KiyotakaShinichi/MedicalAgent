import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


PREVIEW_ROLES = ("dce", "dwi", "t1w")


def preprocess_mri_manifest_previews(
    manifest_csv_path: str = "Data/qin_breast_02_mri_manifest.csv",
    output_dir: str = "Data/qin_mri_previews",
):
    manifest_path = Path(manifest_csv_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"MRI manifest not found: {manifest_csv_path}")

    rows = pd.read_csv(manifest_path).fillna("")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preview_rows = []
    errors = []

    for _, row in rows.iterrows():
        patient_id = row["patient_id"]
        for role in PREVIEW_ROLES:
            folder = row.get(f"{role}_folder", "")
            if not folder:
                errors.append({"patient_id": patient_id, "role": role, "error": "missing selected folder"})
                continue

            try:
                preview = write_series_middle_slice_preview(
                    series_folder=folder,
                    output_path=output_path / f"{patient_id}_{role}.png",
                )
                preview_rows.append({
                    "patient_id": patient_id,
                    "role": role,
                    "series_description": row.get(f"{role}_series_description", ""),
                    "source_folder": folder,
                    "preview_path": str(preview),
                })
            except Exception as exc:
                errors.append({"patient_id": patient_id, "role": role, "error": str(exc)})

    index_csv_path = output_path / "preview_index.csv"
    _write_preview_index(preview_rows, index_csv_path)

    return {
        "manifest_csv_path": str(manifest_path),
        "output_dir": str(output_path),
        "previews_created": len(preview_rows),
        "errors": errors,
        "preview_index_csv": str(index_csv_path),
    }


def write_series_middle_slice_preview(series_folder: str, output_path: Path):
    dicom_files = sorted(path for path in Path(series_folder).rglob("*") if path.is_file())
    if not dicom_files:
        raise ValueError(f"No files found in selected series folder: {series_folder}")

    middle_index = len(dicom_files) // 2
    fallback_errors = []

    for path in [dicom_files[middle_index], *dicom_files]:
        try:
            dataset = pydicom.dcmread(path, force=True)
            pixels = normalize_pixels(dataset.pixel_array)
            image = Image.fromarray(pixels).resize((256, 256))
            image.save(output_path)
            return output_path
        except Exception as exc:
            fallback_errors.append(str(exc))

    raise ValueError(f"Could not read a pixel slice from {series_folder}: {fallback_errors[-1]}")


def normalize_pixels(pixel_array):
    pixels = pixel_array.astype(float)
    if pixels.ndim > 2:
        pixels = pixels[0]

    pixels = pixels - np.min(pixels)
    max_value = np.max(pixels)
    if max_value > 0:
        pixels = pixels / max_value
    return (pixels * 255).astype(np.uint8)


def _write_preview_index(preview_rows, index_csv_path):
    if not preview_rows:
        index_csv_path.write_text("", encoding="utf-8")
        return

    with index_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(preview_rows[0].keys()))
        writer.writeheader()
        writer.writerows(preview_rows)
