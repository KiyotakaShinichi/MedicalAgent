import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image


def generate_breastdcedl_previews(
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv",
    output_dir: str = "Data/breastdcedl_previews",
    max_patients: int = 40,
):
    manifest = pd.read_csv(manifest_csv_path)
    eligible = manifest[
        (manifest["has_core_dce_inputs"] == True)
        & (manifest["has_mask"] == True)
    ].head(max_patients)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    for _, row in eligible.iterrows():
        try:
            preview_path = output_path / f"{row['patient_id']}_dce_mask_overlay.png"
            write_dce_mask_overlay(row, preview_path)
            rows.append({
                "patient_id": row["patient_id"],
                "pcr_label": row.get("pcr_label"),
                "molecular_subtype": row.get("molecular_subtype"),
                "preview_path": str(preview_path),
            })
        except Exception as exc:
            errors.append({"patient_id": row["patient_id"], "error": str(exc)})

    index_path = output_path / "preview_index.csv"
    _write_index(rows, index_path)
    return {
        "manifest_csv_path": manifest_csv_path,
        "output_dir": str(output_path),
        "previews_created": len(rows),
        "errors": errors,
        "preview_index_csv": str(index_path),
    }


def write_dce_mask_overlay(row, output_path):
    acq1 = _load_volume(row["acq1_path"])
    mask = _load_volume(row["mask_path"]) > 0
    if not np.any(mask):
        raise ValueError("Mask has no positive voxels")

    z_index = _largest_mask_slice(mask)
    image_slice = _normalize_to_uint8(acq1[:, :, z_index])
    mask_slice = mask[:, :, z_index]

    rgb = np.stack([image_slice, image_slice, image_slice], axis=-1)
    rgb[mask_slice, 0] = 255
    rgb[mask_slice, 1] = (rgb[mask_slice, 1] * 0.25).astype(np.uint8)
    rgb[mask_slice, 2] = (rgb[mask_slice, 2] * 0.25).astype(np.uint8)

    Image.fromarray(rgb).resize((320, 320)).save(output_path)


def _load_volume(path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def _largest_mask_slice(mask):
    slice_counts = np.sum(mask, axis=(0, 1))
    return int(np.argmax(slice_counts))


def _normalize_to_uint8(slice_array):
    data = slice_array.astype(float)
    low, high = np.percentile(data, [1, 99])
    data = np.clip(data, low, high)
    data = data - np.min(data)
    max_value = np.max(data)
    if max_value > 0:
        data = data / max_value
    return (data * 255).astype(np.uint8)


def _write_index(rows, index_path):
    if not rows:
        index_path.write_text("", encoding="utf-8")
        return

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
