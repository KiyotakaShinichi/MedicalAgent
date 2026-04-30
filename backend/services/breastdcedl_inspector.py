import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = (".nii", ".nii.gz", ".npy", ".npz")
METADATA_SUFFIXES = (".csv", ".tsv", ".xlsx", ".json")


def inspect_breastdcedl_dataset(path):
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"BreastDCEDL path not found: {path}")

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        return _inspect_zip(dataset_path)
    if dataset_path.is_file() and _is_tar_archive(dataset_path):
        return _inspect_tar(dataset_path)
    if dataset_path.is_dir():
        return _inspect_directory(dataset_path)

    raise ValueError(f"Unsupported BreastDCEDL path type: {path}")


def build_breastdcedl_manifest(
    root_path: str = "Datasets/BreastDCEDL_spy1",
    output_csv_path: str = "Data/breastdcedl_spy1_manifest.csv",
):
    root = Path(root_path)
    metadata_path = root / "BreastDCEDL_spy1_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"BreastDCEDL metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    dce_files = _group_dce_files(root / "spt1_dce")
    mask_files = _group_mask_files(root / "spy1_mask")
    rows = []

    for _, item in metadata.iterrows():
        patient_id = str(item["pid"])
        acquisitions = dce_files.get(patient_id, {})
        row = {
            "patient_id": patient_id,
            "age": _clean_value(item.get("age")),
            "er_status": _binary_status(item.get("ER")),
            "pr_status": _binary_status(item.get("PR")),
            "hr_status": _binary_status(item.get("HR")),
            "her2_status": _binary_status(item.get("HER2")),
            "molecular_subtype": _clean_value(item.get("HR_HER2_STATUS")),
            "baseline_longest_diameter_mm": _clean_value(item.get("MRI_LD_Baseline")),
            "pcr_label": _clean_value(item.get("pCR")),
            "rcb_class": _clean_value(item.get("rcb_class")),
            "voi_start_x": _clean_value(item.get("voi_start_x")),
            "voi_start_y": _clean_value(item.get("voi_start_y")),
            "voi_start_z": _clean_value(item.get("voi_start_z")),
            "voi_end_x": _clean_value(item.get("voi_end_x")),
            "voi_end_y": _clean_value(item.get("voi_end_y")),
            "voi_end_z": _clean_value(item.get("voi_end_z")),
            "mask_path": str(mask_files.get(patient_id, "")),
            "acq0_path": str(acquisitions.get("acq0", "")),
            "acq1_path": str(acquisitions.get("acq1", "")),
            "acq2_path": str(acquisitions.get("acq2", "")),
        }
        row["has_core_dce_inputs"] = bool(row["acq0_path"] and row["acq1_path"] and row["acq2_path"])
        row["has_mask"] = bool(row["mask_path"])
        rows.append(row)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    return {
        "root_path": str(root),
        "metadata_rows": len(metadata),
        "manifest_rows": len(rows),
        "patients_with_acq0_acq1_acq2": sum(1 for row in rows if row["has_core_dce_inputs"]),
        "patients_with_masks": sum(1 for row in rows if row["has_mask"]),
        "pcr_counts": pd.Series([row["pcr_label"] for row in rows]).value_counts(dropna=False).to_dict(),
        "output_csv_path": str(output_path),
        "training_readiness": "baseline_model_ready",
    }


def summarize_paths(paths, source_path, source_type):
    extension_counts = Counter(_compound_suffix(path) for path in paths)
    top_level_counts = Counter(_top_level(path) for path in paths)
    image_files = [path for path in paths if path.lower().endswith(IMAGE_SUFFIXES)]
    metadata_files = [path for path in paths if path.lower().endswith(METADATA_SUFFIXES)]

    return {
        "source_path": str(source_path),
        "source_type": source_type,
        "file_count": len(paths),
        "image_file_count": len(image_files),
        "metadata_file_count": len(metadata_files),
        "extension_counts": dict(extension_counts.most_common(12)),
        "top_level_entries": dict(top_level_counts.most_common(12)),
        "sample_image_files": image_files[:10],
        "sample_metadata_files": metadata_files[:10],
        "training_readiness": _training_readiness(image_files, metadata_files),
    }


def _group_dce_files(dce_dir):
    grouped = {}
    if not dce_dir.exists():
        return grouped

    for path in dce_dir.glob("*.nii.gz"):
        patient_id = path.name.split("_spy1_", 1)[0]
        acquisition = path.name.rsplit("_", 1)[-1].replace(".nii.gz", "")
        grouped.setdefault(patient_id, {})[acquisition] = path

    return grouped


def _group_mask_files(mask_dir):
    grouped = {}
    if not mask_dir.exists():
        return grouped

    for path in mask_dir.glob("*.nii.gz"):
        patient_id = path.name.split("_spy1_", 1)[0]
        grouped[patient_id] = path

    return grouped


def _binary_status(value):
    if pd.isna(value):
        return None
    return "Positive" if float(value) == 1.0 else "Negative"


def _clean_value(value):
    if pd.isna(value):
        return None
    return value


def _inspect_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    return summarize_paths(names, path, "zip")


def _inspect_tar(path):
    with tarfile.open(path) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    return summarize_paths(names, path, "tar")


def _inspect_directory(path):
    names = [str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()]
    return summarize_paths(names, path, "directory")


def _is_tar_archive(path):
    name = path.name.lower()
    return name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz")


def _compound_suffix(path):
    lower = path.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(path).suffix.lower() or "[no suffix]"


def _top_level(path):
    normalized = path.replace("\\", "/")
    return normalized.split("/", 1)[0] if normalized else ""


def _training_readiness(image_files, metadata_files):
    if image_files and metadata_files:
        return "ready_for_manifest_mapping"
    if image_files:
        return "images_found_metadata_missing"
    if metadata_files:
        return "metadata_found_images_missing"
    return "not_ready"
