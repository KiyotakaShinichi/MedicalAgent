from __future__ import annotations

import csv
import json

import pandas as pd

from backend.services.cbioportal_clinical_export import build_cbioportal_clinical_export
from backend.services.external_distribution_alignment import build_external_distribution_alignment
from backend.services.student_constraint_elevation_plan import build_student_constraint_elevation_plan


def test_cbioportal_clinical_export_maps_fixture_rows_to_canonical_schema(tmp_path):
    report = build_cbioportal_clinical_export(
        output_path=str(tmp_path / "export.json"),
        output_dir=str(tmp_path / "cbio"),
        combined_csv=str(tmp_path / "combined.csv"),
        live_fetch=False,
        fixture_records=_fixture_records(),
        schema_output_path=str(tmp_path / "schema.json"),
    )

    assert report["status"] == "strong"
    assert report["combined"]["row_count"] == 3
    assert report["combined"]["validation"]["status"] == "passed"
    assert report["combined"]["coverage"]["roles_supported"]["full_oncotrack_temporal_validation"] is False
    assert "do not validate treatment response" in report["claim_boundary"]

    rows = list(csv.DictReader((tmp_path / "combined.csv").open(encoding="utf-8")))
    by_id = {row["source_record_id"]: row for row in rows}
    assert by_id["MB-0001"]["er_status"] == "positive"
    assert by_id["MB-0001"]["treatment_combination_pattern"] == "chemotherapy+endocrine+radiation+surgery"
    assert by_id["TCGA-01"]["stage"] == "Stage IIA"


def test_external_distribution_alignment_uses_real_export_shape_without_validation_claim(tmp_path):
    synthetic = tmp_path / "synthetic.csv"
    breastdcedl = tmp_path / "breastdcedl.csv"
    cbio = tmp_path / "combined.csv"
    _write_synthetic(synthetic)
    _write_breastdcedl_canonical(breastdcedl)
    build_cbioportal_clinical_export(
        output_path=str(tmp_path / "export.json"),
        output_dir=str(tmp_path / "cbio"),
        combined_csv=str(cbio),
        live_fetch=False,
        fixture_records=_fixture_records(),
        schema_output_path=str(tmp_path / "schema.json"),
    )

    report = build_external_distribution_alignment(
        synthetic_csv=str(synthetic),
        breastdcedl_csv=str(breastdcedl),
        cbioportal_csv=str(cbio),
        output_path=str(tmp_path / "alignment.json"),
    )

    assert report["status"] == "strong"
    assert report["cohort_sizes"]["cbioportal_tcga_metabric"] == 3
    assert "age" in report["numeric_alignment"]
    assert report["treatment_context_alignment"]["cbioportal_tcga_metabric"]["rates"]["chemotherapy"] > 0
    assert "not validation of clinical prediction" in report["claim_boundary"]


def test_student_constraint_elevation_plan_prioritizes_controllable_artifacts(tmp_path):
    report = build_student_constraint_elevation_plan(
        output_path=str(tmp_path / "plan.json"),
        doc_path=str(tmp_path / "plan.md"),
    )

    assert report["status"] == "strong"
    assert report["highest_leverage_next_steps"][0]["proof_artifact"] == "Data/evals/models/latest_external_distribution_alignment.json"
    assert any("Do not claim real-world response prediction" in item for item in report["do_not_do_yet"])
    assert "not clinical validation" in report["claim_boundary"]
    assert (tmp_path / "plan.md").exists()


def _fixture_records():
    return {
        "brca_metabric": [
            {
                "patient_id": "MB-0001",
                "AGE_AT_DIAGNOSIS": "55",
                "SEX": "Female",
                "ER_IHC": "Positive",
                "PR_STATUS": "Positive",
                "HER2_STATUS": "Negative",
                "CLAUDIN_SUBTYPE": "LumA",
                "TUMOR_STAGE": "2",
                "CHEMOTHERAPY": "YES",
                "HORMONE_THERAPY": "YES",
                "RADIO_THERAPY": "YES",
                "BREAST_SURGERY": "MASTECTOMY",
                "OS_STATUS": "0:LIVING",
                "OS_MONTHS": "120",
            },
            {
                "patient_id": "MB-0002",
                "AGE_AT_DIAGNOSIS": "63",
                "SEX": "Female",
                "ER_IHC": "Negative",
                "PR_STATUS": "Negative",
                "HER2_STATUS": "Positive",
                "CLAUDIN_SUBTYPE": "Her2",
                "TUMOR_STAGE": "3",
                "CHEMOTHERAPY": "NO",
                "HORMONE_THERAPY": "NO",
                "RADIO_THERAPY": "NO",
                "BREAST_SURGERY": "BREAST CONSERVING",
                "RFS_STATUS": "1:Recurred",
            },
        ],
        "brca_tcga_pan_can_atlas_2018": [
            {
                "patient_id": "TCGA-01",
                "AGE": "48",
                "SEX": "Female",
                "AJCC_PATHOLOGIC_TUMOR_STAGE": "Stage IIA",
                "SUBTYPE": "BRCA_LumA",
                "RADIATION_THERAPY": "YES",
                "HISTORY_NEOADJUVANT_TRTYN": "No",
                "OS_STATUS": "0:LIVING",
                "MUTATION_COUNT": "24",
            }
        ],
    }


def _write_synthetic(path):
    rows = []
    for patient_idx in range(6):
        for cycle in range(1, 3):
            rows.append({
                "patient_id": f"SYN-{patient_idx}",
                "cycle": cycle,
                "age": 45 + patient_idx,
                "molecular_subtype": "HR+/HER2-" if patient_idx % 2 else "HER2+",
                "mri_tumor_size_cm": 2.5 + patient_idx * 0.1,
                "regimen": "TCHP" if patient_idx % 2 == 0 else "dose-dense AC then paclitaxel",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_breastdcedl_canonical(path):
    rows = [
        {
            "source_dataset": "breastdcedl_spy1",
            "source_record_id": "B1",
            "patient_id": "BREASTDCEDL:B1",
            "timepoint_index": 0,
            "age": 50,
            "stage": "unknown",
            "molecular_subtype": "HER2pos",
            "er_status": "unknown",
            "pr_status": "unknown",
            "her2_status": "positive",
            "treatment_modalities": json.dumps(["chemotherapy_context", "MRI"]),
            "imaging_features": json.dumps({"baseline_longest_diameter_mm": 45}),
            "claim_boundary": "test boundary",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
