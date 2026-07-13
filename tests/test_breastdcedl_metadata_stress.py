from __future__ import annotations

import json

import pandas as pd

from backend.services.breastdcedl_metadata_stress import FEATURE_SET, run_breastdcedl_metadata_stress


def test_breastdcedl_metadata_stress_computes_external_probe(tmp_path):
    canonical = tmp_path / "canonical_breastdcedl.csv"
    output = tmp_path / "stress.json"
    predictions = tmp_path / "predictions.csv"
    doc = tmp_path / "stress.md"
    _write_canonical_rows(canonical, n_rows=72)

    report = run_breastdcedl_metadata_stress(
        canonical_csv=canonical,
        output_path=output,
        predictions_path=predictions,
        doc_path=doc,
        seed=11,
        n_bootstrap=25,
    )

    assert output.exists()
    assert predictions.exists()
    assert doc.exists()
    assert report["status"] == "strong"
    assert report["clinical_validation"] is False
    assert report["production_training_allowed"] is False
    assert report["metadata_only"] is True
    assert report["image_pixel_training"] is False
    assert report["feature_set"] == FEATURE_SET
    assert report["stress_result"]["status"] == "computed"
    assert report["stress_result"]["clinical_interpretation_allowed"] is False
    assert report["integration_decision"]["model_promotion_allowed"] is False
    assert "not clinical validation" in report["claim_boundary"]


def test_breastdcedl_metadata_stress_exports_row_predictions_as_nonclinical(tmp_path):
    canonical = tmp_path / "canonical.csv"
    predictions = tmp_path / "predictions.csv"
    _write_canonical_rows(canonical, n_rows=64)

    run_breastdcedl_metadata_stress(
        canonical_csv=canonical,
        output_path=tmp_path / "stress.json",
        predictions_path=predictions,
        doc_path=tmp_path / "stress.md",
        n_bootstrap=10,
    )

    rows = pd.read_csv(predictions)
    assert {"patient_id", "pcr_label", "metadata_probe_pcr_probability", "clinical_interpretation_allowed"} <= set(rows.columns)
    assert rows["clinical_interpretation_allowed"].eq(False).all()
    assert rows["metadata_probe_pcr_probability"].between(0, 1).all()
    assert rows["target_mismatch_note"].str.contains("not equivalent", case=False).all()


def test_breastdcedl_metadata_stress_stays_target_mismatched(tmp_path):
    canonical = tmp_path / "canonical.csv"
    _write_canonical_rows(canonical, n_rows=48)

    report = run_breastdcedl_metadata_stress(
        canonical_csv=canonical,
        output_path=tmp_path / "stress.json",
        predictions_path=tmp_path / "predictions.csv",
        doc_path=tmp_path / "stress.md",
        n_bootstrap=10,
    )

    assert report["target"]["external_label"] == "pathologic complete response (pCR)"
    assert report["target"]["nlcare_label_equivalent"] is False
    assert "pCR" in report["target"]["target_mismatch"]
    assert "model promotion to patient-facing route" in report["blocked_claims"]


def test_breastdcedl_metadata_stress_handles_missing_or_small_export(tmp_path):
    report = run_breastdcedl_metadata_stress(
        canonical_csv=tmp_path / "missing.csv",
        output_path=tmp_path / "stress.json",
        predictions_path=tmp_path / "predictions.csv",
        doc_path=tmp_path / "stress.md",
    )

    assert report["status"] == "needs_attention"
    assert report["stress_result"]["status"] == "not_computed"
    assert report["clinical_validation"] is False
    assert report["integration_decision"]["live_model_update_allowed"] is False


def _write_canonical_rows(path, *, n_rows: int) -> None:
    rows = []
    subtypes = ["HRposHER2neg", "HER2pos", "TripleNeg", "LuminalA"]
    for idx in range(n_rows):
        pcr = int(idx % 4 == 0 or (idx % 9 == 0 and idx % 2 == 0))
        subtype = subtypes[idx % len(subtypes)]
        er = "negative" if subtype == "TripleNeg" else "positive"
        pr = "negative" if subtype in {"TripleNeg", "HER2pos"} else "positive"
        her2 = "positive" if subtype == "HER2pos" else "negative"
        rows.append(
            {
                "source_dataset": "breastdcedl_spy1",
                "source_record_id": f"ISPY1_{idx:04d}",
                "patient_id": f"BREASTDCEDL:ISPY1_{idx:04d}",
                "age": 34 + idx % 38,
                "molecular_subtype": subtype,
                "er_status": er,
                "pr_status": pr,
                "her2_status": her2,
                "imaging_features": json.dumps(
                    {
                        "baseline_longest_diameter_mm": 30 + (idx % 13) * 4 - pcr * 3,
                        "acq0_mask_mean": 80 + idx,
                    }
                ),
                "outcome_label_name": "pCR",
                "outcome_label_value": pcr,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
