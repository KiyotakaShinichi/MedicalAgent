from __future__ import annotations

import csv
import json

import pandas as pd

from backend.services.common_feature_transfer_stress import run_common_feature_transfer_stress
from backend.services.public_distribution_realism_candidate import build_public_distribution_realism_candidate


def test_common_feature_transfer_stress_is_claim_bounded_and_monitor_only(tmp_path):
    synthetic = tmp_path / "synthetic.csv"
    breastdcedl = tmp_path / "breastdcedl.csv"
    cbio = tmp_path / "cbio.csv"
    _write_synthetic_rows(synthetic, n_patients=52)
    _write_breastdcedl_rows(breastdcedl, n_rows=64)
    _write_cbioportal_rows(cbio, n_rows=28)

    report = run_common_feature_transfer_stress(
        synthetic_csv=str(synthetic),
        breastdcedl_csv=str(breastdcedl),
        cbioportal_csv=str(cbio),
        output_path=str(tmp_path / "transfer.json"),
        seed=7,
    )

    assert report["status"] == "strong"
    assert report["feature_set"] == ["age", "baseline_tumor_size_mm", "hr_positive", "her2_positive", "triple_negative"]
    assert report["within_dataset_models"]["synthetic_treatment_success"]["status"] == "computed"
    assert report["within_dataset_models"]["breastdcedl_pcr"]["status"] == "computed"
    assert report["transfer_stress"]["synthetic_model_on_breastdcedl"]["clinical_interpretation_allowed"] is False
    assert report["promotion_decision"]["promotion_allowed"] is False
    assert "must not be described as clinical validation" in report["claim_boundary"]


def test_public_distribution_realism_candidate_writes_candidate_and_documents_boundary(tmp_path):
    synthetic = tmp_path / "synthetic.csv"
    alignment = tmp_path / "alignment.json"
    candidate = tmp_path / "candidate.csv"
    _write_synthetic_rows(synthetic, n_patients=12)
    alignment.write_text(json.dumps({
        "numeric_alignment": {
            "age": {
                "synthetic": {"mean": 45, "p10": 30, "p90": 60},
                "cbioportal_tcga_metabric": {"mean": 61, "p10": 41, "p90": 78},
            },
            "baseline_tumor_size_mm": {
                "synthetic": {"mean": 28, "p10": 20, "p90": 35},
                "breastdcedl": {"mean": 68, "p10": 32, "p90": 106},
            },
        }
    }), encoding="utf-8")

    report = build_public_distribution_realism_candidate(
        synthetic_csv=str(synthetic),
        alignment_path=str(alignment),
        output_path=str(tmp_path / "realism.json"),
        candidate_csv=str(candidate),
    )

    assert report["status"] == "candidate"
    assert report["realism_candidate_decision"]["production_replacement_allowed"] is False
    assert report["before_after_gaps"]["age"]["gap_improved"] is True
    assert report["before_after_gaps"]["baseline_tumor_size_mm_proxy"]["gap_improved"] is True
    assert candidate.exists()
    assert "still synthetic" in report["claim_boundary"]


def _write_synthetic_rows(path, *, n_patients: int):
    rows = []
    subtypes = ["HR+/HER2-", "HER2+", "triple-negative"]
    for patient_idx in range(n_patients):
        label = int(patient_idx % 2 == 0)
        for cycle in range(1, 4):
            rows.append({
                "patient_id": f"SYN-{patient_idx:03d}",
                "cycle": cycle,
                "age": 32 + patient_idx % 38,
                "molecular_subtype": subtypes[patient_idx % len(subtypes)],
                "mri_tumor_size_cm": 2.4 + (patient_idx % 9) * 0.16 - label * 0.18 + cycle * 0.03,
                "treatment_success_binary": label,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_breastdcedl_rows(path, *, n_rows: int):
    rows = []
    subtypes = ["HRposHER2neg", "HER2pos", "TripleNeg"]
    for idx in range(n_rows):
        label = int(idx % 3 == 0)
        rows.append({
            "patient_id": f"EXT-{idx:03d}",
            "pcr_label": label,
            "age": 38 + idx % 32,
            "baseline_longest_diameter_mm": 40 + (idx % 14) * 4 - label * 3,
            "molecular_subtype": subtypes[idx % len(subtypes)],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_cbioportal_rows(path, *, n_rows: int):
    rows = []
    subtypes = ["LumA", "Her2", "Basal", "claudin-low"]
    for idx in range(n_rows):
        rows.append({
            "source_dataset": "fixture_cbioportal",
            "source_record_id": f"CBIO-{idx:03d}",
            "patient_id": f"CBIO:{idx:03d}",
            "timepoint_index": 0,
            "age": 42 + idx % 40,
            "molecular_subtype": subtypes[idx % len(subtypes)],
            "er_status": "positive" if idx % 4 in {0, 3} else "negative",
            "pr_status": "positive" if idx % 4 == 0 else "negative",
            "her2_status": "positive" if idx % 4 == 1 else "negative",
            "imaging_features": json.dumps({"baseline_longest_diameter_mm": 30 + idx % 20}),
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
