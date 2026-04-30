from datetime import date

import pandas as pd

from backend.models import BreastCancerProfile, ImagingReport, Patient


def import_breastdcedl_patients_to_dashboard(
    db,
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv",
    limit: int = 25,
):
    manifest = pd.read_csv(manifest_csv_path)
    imported = 0
    skipped = 0

    for _, row in manifest.head(limit).iterrows():
        patient_id = str(row["patient_id"])
        existing = db.query(Patient).filter(Patient.id == patient_id).first()
        if existing:
            skipped += 1
            continue

        db.add(Patient(
            id=patient_id,
            name=f"BreastDCEDL {patient_id}",
            diagnosis="Breast cancer - I-SPY1 BreastDCEDL public dataset",
        ))
        db.add(BreastCancerProfile(
            patient_id=patient_id,
            cancer_stage=None,
            er_status=row.get("er_status"),
            pr_status=row.get("pr_status"),
            her2_status=row.get("her2_status"),
            molecular_subtype=row.get("molecular_subtype"),
            treatment_intent="neoadjuvant treatment response modeling",
            menopausal_status=None,
        ))
        db.add(ImagingReport(
            patient_id=patient_id,
            date=date(2025, 1, 1),
            modality="DCE-MRI",
            report_type="BreastDCEDL baseline metadata",
            body_site="Breast",
            findings=(
                f"Baseline longest diameter {row.get('baseline_longest_diameter_mm')} mm. "
                f"pCR label {row.get('pcr_label')}. Mask available: {row.get('has_mask')}."
            ),
            impression="Public BreastDCEDL I-SPY1 metadata imported for PoC dashboard display.",
        ))
        imported += 1

    db.commit()
    return {
        "manifest_csv_path": manifest_csv_path,
        "limit": limit,
        "patients_imported": imported,
        "patients_skipped": skipped,
    }
