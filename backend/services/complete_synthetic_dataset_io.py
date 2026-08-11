"""Serialization and schema descriptions for the synthetic journey generator."""

from __future__ import annotations

import pandas as pd


def empty_tables():
    return {
        "patients": [],
        "diagnoses": [],
        "treatment_sessions": [],
        "labs": [],
        "medications": [],
        "symptoms": [],
        "mri_reports": [],
        "interventions": [],
        "outcomes": [],
        "temporal_ml_rows": [],
    }


def write_tables(output_path, tables):
    manifest = {}
    for name, rows in tables.items():
        file_path = output_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(file_path, index=False)
        manifest[name] = str(file_path)
    return manifest


def data_dictionary():
    return {
        "patients": "One row per synthetic patient.",
        "diagnoses": "Synthetic diagnosis and receptor/subtype profile.",
        "treatment_sessions": "One row per scheduled treatment cycle with regimen, dates, and dose status.",
        "labs": "CBC rows at baseline/pre-cycle/nadir/recovery, with WBC, ANC, RBC, hemoglobin, and platelets.",
        "medications": "Anti-cancer regimen entries, supportive medications, and intervention medications/products.",
        "symptoms": "Patient-reported symptoms around treatment cycles.",
        "mri_reports": "Synthetic imaging report events. MRI is common; CT and ultrasound are optional monitoring/staging signals and are not required for every patient.",
        "interventions": "Clinical support events such as growth-factor support, transfusions, antibiotics, or urgent review.",
        "outcomes": "Synthetic end-of-journey response labels and maintenance status.",
        "temporal_ml_rows": "Training-ready cycle-level features with final outcome labels.",
        "extra_labels": "Synthetic labels include treatment_success_binary, response_score_percent, maintenance_needed, toxicity_risk_binary, support_intervention_needed, urgent_intervention_needed, final_response_multiclass, and cycle_response_trend_class.",
        "warning": "All tables are synthetic and should be used only for engineering demos and ML practice.",
    }


__all__ = ["data_dictionary", "empty_tables", "write_tables"]
