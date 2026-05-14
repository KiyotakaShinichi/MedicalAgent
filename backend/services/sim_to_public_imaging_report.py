from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.public_imaging_datasets import DEFAULT_OUTPUT_PATH as PUBLIC_IMAGING_MANIFEST_PATH, build_public_imaging_manifest


DEFAULT_OUTPUT_PATH = "Data/public_imaging/sim_to_public_imaging_report.json"


def build_sim_to_public_imaging_report(
    synthetic_mri_path: str = "Data/complete_synthetic_breast_journeys/mri_reports.csv",
    public_manifest_path: str = PUBLIC_IMAGING_MANIFEST_PATH,
    output_path: str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    synthetic_summary = _synthetic_imaging_summary(synthetic_mri_path)
    public_manifest = _load_or_build_public_manifest(public_manifest_path)
    available_modalities = _available_modalities(public_manifest)
    gap_table = _gap_table(synthetic_summary, public_manifest)

    payload = {
        "schema_version": "sim_to_public_imaging_gap_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "computed",
        "synthetic_summary": synthetic_summary,
        "public_imaging_availability": {
            "status": public_manifest.get("status"),
            "available_dataset_count": public_manifest.get("available_dataset_count", 0),
            "available_modalities": available_modalities,
        },
        "gap_table": gap_table,
        "recommended_actions": _recommended_actions(gap_table),
        "claim_boundary": (
            "This report identifies engineering coverage gaps. It does not validate clinical performance."
        ),
    }
    if output_path:
        _write_json(output_path, payload)
    return payload


def _synthetic_imaging_summary(path: str) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {
            "status": "missing",
            "path": path,
            "row_count": 0,
            "modalities": {},
            "report_type_counts": {},
            "size_cm": None,
            "metastatic_keyword_rows": 0,
        }

    df = pd.read_csv(csv_path)
    findings = df["findings"].astype(str) if "findings" in df else pd.Series([""] * len(df))
    impression = df["impression"].astype(str) if "impression" in df else pd.Series([""] * len(df))
    text = (findings + " " + impression).str.lower()
    size_col = _first_existing(df, ["tumor_size_cm", "largest_tumor_size_cm", "size_cm"])
    size_summary = None
    if size_col:
        values = pd.to_numeric(df[size_col], errors="coerce").dropna()
        if len(values):
            size_summary = {
                "mean": round(float(values.mean()), 4),
                "median": round(float(values.median()), 4),
                "p10": round(float(values.quantile(0.10)), 4),
                "p90": round(float(values.quantile(0.90)), 4),
            }

    return {
        "status": "loaded",
        "path": path,
        "row_count": int(len(df)),
        "patient_count": int(df["patient_id"].nunique()) if "patient_id" in df else None,
        "modalities": _value_counts(df, "modality"),
        "report_type_counts": _value_counts(df, "report_type"),
        "size_cm": size_summary,
        "metastatic_keyword_rows": int(text.str.contains("metasta|ascites|peritoneal|pleural|hepatic|bone", regex=True).sum()),
    }


def _load_or_build_public_manifest(path: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return build_public_imaging_manifest(output_path=path)


def _available_modalities(manifest: dict[str, Any]) -> list[str]:
    modalities = []
    for dataset in manifest.get("datasets", []):
        if dataset.get("available"):
            modalities.append(dataset.get("modality", "unknown"))
    return sorted(set(modalities))


def _gap_table(synthetic: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = manifest.get("datasets", [])
    available_ids = {dataset["id"] for dataset in datasets if dataset.get("available")}
    synthetic_modalities = {key.lower() for key in (synthetic.get("modalities") or {}).keys()}
    return [
        {
            "area": "breast_mri_response",
            "synthetic_coverage": "covered" if synthetic.get("row_count", 0) else "missing",
            "public_coverage": "covered_if_downloaded",
            "available_now": "tcia_qin_breast" in available_ids,
            "gap": "Synthetic MRI response exists; public MRI/PET-CT data must be downloaded for image-level validation.",
        },
        {
            "area": "breast_ultrasound",
            "synthetic_coverage": "not_modeled" if not any("ultrasound" in item for item in synthetic_modalities) else "present",
            "public_coverage": "covered_if_downloaded",
            "available_now": bool({"busi_breast_ultrasound", "bus_uclm_breast_ultrasound"} & available_ids),
            "gap": "Ultrasound is not part of the synthetic treatment timeline yet; use public US as a separate narrow imaging benchmark.",
        },
        {
            "area": "ct_pet_ct_lesions",
            "synthetic_coverage": "report_text_only",
            "public_coverage": "partially_covered_if_downloaded",
            "available_now": bool({"nih_deeplesion", "tcia_fdg_pet_ct_lesions", "tcia_qin_breast"} & available_ids),
            "gap": "CT/PET-CT public data can support lesion workflows, but disease-origin and metastatic breast cancer labels remain limited.",
        },
        {
            "area": "ascites_peritoneal_disease",
            "synthetic_coverage": "report_text_indicator_rules",
            "public_coverage": "labels_not_selected",
            "available_now": False,
            "gap": "Current safest support is report-text extraction. Image-level ascites requires curated labels or annotation.",
        },
    ]


def _recommended_actions(gaps: list[dict[str, Any]]) -> list[str]:
    actions = [
        "Keep report-text extraction as the patient-facing CT/US route until image labels are task-specific.",
        "Download BUSI or BUS-UCLM first for the fastest image-model proof of work.",
        "Use DeepLesion or FDG-PET-CT-Lesions for CT/PET-CT ingestion and lesion-localization workflow only.",
        "Add synthetic ultrasound/CT report events only as monitoring signals, not diagnosis labels.",
    ]
    if any(not gap.get("available_now") for gap in gaps):
        actions.append("Show missing public datasets explicitly in the admin dashboard to avoid overclaiming.")
    return actions


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).to_dict().items()}


def _first_existing(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df:
            return column
    return None


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
