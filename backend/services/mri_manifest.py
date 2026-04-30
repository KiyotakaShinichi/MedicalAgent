import csv
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from backend.models import BreastCancerProfile, MRISeriesIndex, Patient


MODEL_INPUT_ROLES = ("dce", "dwi", "t1w")


def build_qin_mri_manifest(
    db: Session,
    clinical_xlsx_path: str | None = "Datasets/QIN-BREAST-02_clinicalData-Transformed-20191022-Revised20200210.xlsx",
    output_csv_path: str | None = "Data/qin_breast_02_mri_manifest.csv",
):
    clinical_labels = _load_qin_clinical_labels(clinical_xlsx_path)
    rows = []

    patients = (
        db.query(Patient)
        .filter(Patient.id.like("QIN-BREAST-02-%"))
        .order_by(Patient.id)
        .all()
    )

    for patient in patients:
        profile = (
            db.query(BreastCancerProfile)
            .filter(BreastCancerProfile.patient_id == patient.id)
            .first()
        )
        series = (
            db.query(MRISeriesIndex)
            .filter(MRISeriesIndex.patient_id == patient.id)
            .all()
        )
        selected = select_model_input_series(series)
        labels = clinical_labels.get(patient.id, {})

        row = {
            "patient_id": patient.id,
            "study_date": _first_value(selected, "study_date"),
            "cancer_stage": profile.cancer_stage if profile else None,
            "er_status": profile.er_status if profile else None,
            "pr_status": profile.pr_status if profile else None,
            "her2_status": profile.her2_status if profile else None,
            "molecular_subtype": profile.molecular_subtype if profile else None,
            "baseline_tumor_size_cm": labels.get("baseline_tumor_size_cm"),
            "response": labels.get("response"),
            "scan_1_completed": labels.get("scan_1_completed"),
            "scan_2_completed": labels.get("scan_2_completed"),
            "scan_3_completed": labels.get("scan_3_completed"),
            "scan_4_completed": labels.get("scan_4_completed"),
            "has_all_core_mri_roles": all(role in selected for role in MODEL_INPUT_ROLES),
        }

        for role in MODEL_INPUT_ROLES:
            item = selected.get(role)
            row[f"{role}_series_uid"] = item.series_uid if item else None
            row[f"{role}_series_description"] = item.series_description if item else None
            row[f"{role}_folder"] = item.folder if item else None
            row[f"{role}_instance_count"] = item.instance_count if item else None

        rows.append(row)

    if output_csv_path:
        _write_manifest_csv(rows, output_csv_path)

    return {
        "rows": rows,
        "summary": _summarize_manifest(rows),
        "output_csv_path": output_csv_path,
        "note": "This manifest selects DICOM series folders and labels for preprocessing/training experiments. It does not train a model.",
    }


def select_model_input_series(series):
    selected = {}
    for role in MODEL_INPUT_ROLES:
        candidates = [item for item in series if item.candidate_role == role]
        if not candidates:
            continue
        selected[role] = sorted(
            candidates,
            key=lambda item: (
                item.instance_count or 0,
                _description_priority(role, item.series_description),
                item.series_description or "",
            ),
            reverse=True,
        )[0]
    return selected


def _description_priority(role, description):
    text = (description or "").lower()
    if role == "dce":
        if "dynamic" in text or "dyn" in text:
            return 3
        if "dce" in text:
            return 2
    if role == "dwi":
        if "b800" in text or "b0200800" in text:
            return 3
        if "dwi" in text:
            return 2
    if role == "t1w":
        if "thrive" in text:
            return 3
        if "t1" in text:
            return 2
    return 1


def _load_qin_clinical_labels(clinical_xlsx_path):
    if not clinical_xlsx_path or not Path(clinical_xlsx_path).exists():
        return {}

    df = pd.read_excel(clinical_xlsx_path)
    df = df[df["NBIA ID"].astype(str).str.startswith("QIN-BREAST-02")]
    labels = {}

    for _, row in df.iterrows():
        patient_id = str(row["NBIA ID"])
        labels[patient_id] = {
            "baseline_tumor_size_cm": _clean_value(row.get("Size (cm)  ")),
            "response": _clean_value(row.get("Response")),
            "scan_1_completed": _clean_value(row.get("Pre-treatment (Scan 1) Completed")),
            "scan_2_completed": _clean_value(row.get("Scan 2 Completed")),
            "scan_3_completed": _clean_value(row.get("Scan 3 Completed")),
            "scan_4_completed": _clean_value(row.get("Scan 4 Completed")),
        }

    return labels


def _write_manifest_csv(rows, output_csv_path):
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_manifest(rows):
    response_counts = {}
    complete_core_roles = 0

    for row in rows:
        response = row.get("response") or "unknown"
        response_counts[response] = response_counts.get(response, 0) + 1
        if row.get("has_all_core_mri_roles"):
            complete_core_roles += 1

    return {
        "patient_count": len(rows),
        "patients_with_dce_dwi_t1w": complete_core_roles,
        "response_counts": response_counts,
        "model_readiness": "exploratory_only" if len(rows) < 50 else "small_dataset",
    }


def _first_value(selected, attribute):
    for role in MODEL_INPUT_ROLES:
        item = selected.get(role)
        if item is not None:
            return str(getattr(item, attribute)) if getattr(item, attribute) is not None else None
    return None


def _clean_value(value):
    if pd.isna(value):
        return None
    return value
