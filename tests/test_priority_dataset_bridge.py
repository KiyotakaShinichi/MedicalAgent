from __future__ import annotations

import csv
import json

from backend.services.priority_dataset_bridge import build_priority_dataset_bridge


def test_priority_dataset_bridge_writes_templates_without_local_exports(tmp_path):
    report = build_priority_dataset_bridge(
        output_path=str(tmp_path / "priority_bridge.json"),
        doc_path=str(tmp_path / "priority_bridge.md"),
        template_dir=str(tmp_path / "templates"),
        genie_canonical_csv=str(tmp_path / "genie.csv"),
        duke_canonical_csv=str(tmp_path / "duke.csv"),
        schema_output_path=str(tmp_path / "schema.json"),
    )

    assert report["status"] == "ready_for_mapping"
    assert report["summary"]["ready_for_mapping_count"] == 2
    assert report["summary"]["full_oncotrack_temporal_validation_ready"] == 0
    assert (tmp_path / "templates" / "genie_bpc_brca_field_contract.csv").exists()
    assert (tmp_path / "templates" / "duke_breast_mri_field_contract.csv").exists()
    assert "clinical validation" in report["claim_boundary"]


def test_priority_dataset_bridge_maps_fixture_rows_into_canonical_schema(tmp_path):
    genie = tmp_path / "genie_fixture.csv"
    duke = tmp_path / "duke_fixture.csv"
    _write_csv(genie, [
        {
            "PATIENT_ID": "G001",
            "AGE_AT_DIAGNOSIS": "44",
            "SEX": "Female",
            "STAGE_AT_DIAGNOSIS": "Stage II",
            "ER_STATUS": "Positive",
            "PR_STATUS": "Positive",
            "HER2_STATUS": "Positive",
            "HUGO_SYMBOL": "ERBB2",
            "VARIANT_CLASSIFICATION": "amplification",
            "REGIMEN": "paclitaxel + trastuzumab + pertuzumab",
            "TREATMENT_SETTING": "neoadjuvant",
            "BEST_RESPONSE": "partial response",
        }
    ])
    _write_csv(duke, [
        {
            "Patient ID": "D001",
            "Age": "51",
            "Sex": "F",
            "Stage": "II",
            "ER": "Negative",
            "PR": "Negative",
            "HER2": "Negative",
            "Mol Subtype": "Triple Negative",
            "NAC": "neoadjuvant chemotherapy",
            "Radiation Therapy": "yes",
            "pCR": "1",
            "Tumor Size": "32.5",
            "washout_mean": "-0.14",
        }
    ])

    report = build_priority_dataset_bridge(
        genie_csv=str(genie),
        duke_csv=str(duke),
        output_path=str(tmp_path / "priority_bridge.json"),
        doc_path=str(tmp_path / "priority_bridge.md"),
        template_dir=str(tmp_path / "templates"),
        genie_canonical_csv=str(tmp_path / "canonical_genie.csv"),
        duke_canonical_csv=str(tmp_path / "canonical_duke.csv"),
        schema_output_path=str(tmp_path / "schema.json"),
    )

    assert report["status"] == "strong"
    assert report["summary"]["mapped_dataset_count"] == 2
    assert report["datasets"]["genie_bpc_brca"]["validation"]["status"] == "passed"
    assert report["datasets"]["duke_breast_mri"]["validation"]["status"] == "passed"

    genie_rows = list(csv.DictReader((tmp_path / "canonical_genie.csv").open(encoding="utf-8")))
    duke_rows = list(csv.DictReader((tmp_path / "canonical_duke.csv").open(encoding="utf-8")))
    assert genie_rows[0]["patient_id"] == "GENIE_BPC_BRCA:G001"
    assert "chemotherapy" in genie_rows[0]["treatment_modalities"]
    assert "targeted_anti_her2" in genie_rows[0]["treatment_modalities"]
    assert duke_rows[0]["patient_id"] == "DUKE_BREAST_MRI:D001"
    assert duke_rows[0]["imaging_available"] == "True"
    assert duke_rows[0]["outcome_label_name"] == "pCR"

    saved = json.loads((tmp_path / "priority_bridge.json").read_text(encoding="utf-8"))
    assert saved["blocked_claims"]


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
