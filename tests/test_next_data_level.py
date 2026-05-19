from __future__ import annotations

import csv
import json

import pandas as pd

from backend.services.external_failure_case_analysis import build_external_failure_case_analysis
from backend.services.restricted_data_access_packet import build_restricted_data_access_packet
from backend.services.strict_common_feature_ab_eval import run_strict_common_feature_ab_eval
from backend.services.tcga_metabric_canonical_mapping import build_tcga_metabric_canonical_mapping
from backend.services.toxicity_review_target_v2 import run_toxicity_review_target_v2


def test_tcga_metabric_canonical_mapping_is_schema_only_and_claim_bounded(tmp_path):
    cbio = tmp_path / "cbio.json"
    cbio.write_text(json.dumps({
        "status": "ready",
        "datasets": {
            "tcga_brca_pan_can_atlas_2018": {
                "status": "mapped",
                "study_id": "brca_tcga_pan_can_atlas_2018",
                "label": "TCGA-BRCA",
                "mapped_groups": _mapped_groups(),
            },
            "metabric": {
                "status": "mapped",
                "study_id": "brca_metabric",
                "label": "METABRIC",
                "mapped_groups": _mapped_groups(),
            },
        },
    }), encoding="utf-8")

    report = build_tcga_metabric_canonical_mapping(
        output_path=str(tmp_path / "report.json"),
        mapping_path=str(tmp_path / "mapping.json"),
        source_mapping_path=str(cbio),
        live_fetch=False,
    )

    assert report["status"] == "strong"
    assert report["mapped_dataset_count"] == 2
    assert "molecular_subtype" in report["strict_common_feature_set"]
    assert "not validation" in report["claim_boundary"]
    assert report["datasets"]["metabric"]["target_mismatch"]


def test_strict_common_feature_ab_eval_uses_shared_fields_and_holds_promotion(tmp_path):
    synthetic = tmp_path / "synthetic.csv"
    external = tmp_path / "external.csv"
    _write_synthetic_rows(synthetic, n_patients=44)
    _write_external_rows(external, n_rows=60)

    report = run_strict_common_feature_ab_eval(
        synthetic_csv=str(synthetic),
        breastdcedl_csv=str(external),
        output_path=str(tmp_path / "ab.json"),
        seed=5,
    )

    assert report["status"] == "strong"
    assert report["feature_set"] == ["age", "baseline_tumor_size_mm", "hr_positive", "her2_positive", "triple_negative"]
    assert report["ab_decision"]["promotion_allowed"] is False
    assert report["datasets"]["synthetic_patient_level"]["metrics"]["status"] == "computed"
    assert report["datasets"]["breastdcedl_spy1"]["metrics"]["status"] == "computed"
    assert "not be described as clinical validation" in report["claim_boundary"]


def test_toxicity_review_target_v2_is_candidate_not_clinical_prediction(tmp_path):
    source = tmp_path / "toxicity_rows.csv"
    _write_toxicity_rows(source, n_patients=48)

    report = run_toxicity_review_target_v2(source_csv=str(source), output_path=str(tmp_path / "toxicity_v2.json"))

    assert report["status"] in {"candidate", "needs_attention"}
    assert report["target"] == "toxicity_review_priority_v2"
    assert report["shortcut_comparison"]["legacy_rule_does_not_define_v2"] is True
    assert "not a toxicity diagnosis" in report["claim_boundary"]


def test_external_failure_case_analysis_groups_by_subtype_and_confidence(tmp_path):
    predictions = tmp_path / "predictions.csv"
    _write_csv(predictions, [
        {"patient_id": "A", "pcr_label": "1", "best_model_predicted_label": "0", "best_model_pcr_probability": "0.08", "molecular_subtype": "HER2pos"},
        {"patient_id": "B", "pcr_label": "0", "best_model_predicted_label": "1", "best_model_pcr_probability": "0.86", "molecular_subtype": "HER2pos"},
        {"patient_id": "C", "pcr_label": "0", "best_model_predicted_label": "0", "best_model_pcr_probability": "0.20", "molecular_subtype": "TripleNeg"},
    ])

    report = build_external_failure_case_analysis(predictions_csv=str(predictions), output_path=str(tmp_path / "failures.json"))

    assert report["status"] == "strong"
    assert report["summary"]["failure_count"] == 2
    assert report["summary"]["high_confidence_failure_count"] == 2
    assert report["by_molecular_subtype"][0]["molecular_subtype"] == "HER2pos"
    assert "not patient-level clinical adjudication" in report["claim_boundary"]


def test_restricted_data_access_packet_is_future_request_only(tmp_path):
    report = build_restricted_data_access_packet(
        output_path=str(tmp_path / "packet.json"),
        md_path=str(tmp_path / "packet.md"),
    )

    assert report["status"] == "ready_for_future_access_request"
    assert {row["dataset"] for row in report["datasets"]} >= {
        "AACR GENIE BPC Breast Cancer",
        "SEER breast registry",
        "SEER-Medicare",
    }
    assert "does not mean access has been granted" in report["claim_boundary"]


def _mapped_groups():
    return {
        "er_status": [{"id": "ER_STATUS", "display_name": "ER status"}],
        "pr_status": [{"id": "PR_STATUS", "display_name": "PR status"}],
        "her2_status": [{"id": "HER2_STATUS", "display_name": "HER2 status"}],
        "subtype": [{"id": "PAM50", "display_name": "PAM50 subtype"}],
        "stage": [{"id": "STAGE", "display_name": "Stage"}],
        "survival": [{"id": "OS_STATUS", "display_name": "Overall survival status"}],
        "genomic": [{"id": "MUTATION_COUNT", "display_name": "Mutation count"}],
    }


def _write_synthetic_rows(path, *, n_patients: int):
    rows = []
    subtypes = ["HR+/HER2-", "HER2+", "triple-negative"]
    for patient_idx in range(n_patients):
        label = int(patient_idx % 2 == 0)
        for cycle in range(1, 4):
            rows.append({
                "patient_id": f"SYN-{patient_idx:03d}",
                "cycle": cycle,
                "age": 35 + patient_idx % 30,
                "molecular_subtype": subtypes[patient_idx % len(subtypes)],
                "mri_tumor_size_cm": 2.5 + (patient_idx % 8) * 0.2 - label * 0.4,
                "treatment_success_binary": label,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_external_rows(path, *, n_rows: int):
    rows = []
    subtypes = ["HRposHER2neg", "HER2pos", "TripleNeg"]
    for idx in range(n_rows):
        label = int(idx % 3 == 0)
        rows.append({
            "patient_id": f"EXT-{idx:03d}",
            "pcr_label": label,
            "age": 38 + idx % 25,
            "baseline_longest_diameter_mm": 25 + (idx % 10) * 3 - label * 2,
            "molecular_subtype": subtypes[idx % len(subtypes)],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_toxicity_rows(path, *, n_patients: int):
    rows = []
    for patient_idx in range(n_patients):
        for cycle in range(1, 5):
            cbc_low = int((patient_idx + cycle) % 5 == 0)
            symptom_high = int((patient_idx * 2 + cycle) % 4 == 0)
            intervention = int(symptom_high or (patient_idx + cycle) % 7 == 0)
            rows.append({
                "patient_id": f"TOX-{patient_idx:03d}",
                "cycle": cycle,
                "treatment_date": f"2026-01-{cycle:02d}",
                "age": 42 + patient_idx % 20,
                "stage": "IIA",
                "molecular_subtype": "HR+/HER2-" if patient_idx % 2 else "HER2+",
                "regimen": "dose-dense AC then paclitaxel",
                "pre_wbc": 5.5 - cbc_low * 1.2,
                "pre_anc": 3.0 - cbc_low * 1.4,
                "pre_hemoglobin": 12.2 - cbc_low * 0.8,
                "pre_platelets": 240 - cbc_low * 60,
                "nadir_wbc": 3.0 - cbc_low * 1.5,
                "nadir_anc": 1.6 - cbc_low * 0.9,
                "nadir_hemoglobin": 11.4 - cbc_low * 0.9,
                "nadir_platelets": 170 - cbc_low * 90,
                "recovery_wbc": 4.8 - cbc_low * 1.1,
                "recovery_hemoglobin": 11.8 - cbc_low * 0.6,
                "recovery_platelets": 215 - cbc_low * 50,
                "mri_tumor_size_cm": 3.0,
                "mri_percent_change_from_baseline": -20,
                "max_symptom_severity": 3 + symptom_high * 5,
                "symptom_count": 1 + symptom_high * 3,
                "intervention_count": intervention,
                "dose_delayed": intervention,
                "dose_reduced": int(intervention and cycle % 2 == 0),
                "toxicity_risk_binary": cbc_low,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
