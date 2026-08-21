from __future__ import annotations


from backend.services.public_biomarker_dataset_readiness import build_public_biomarker_dataset_readiness


def test_public_biomarker_dataset_readiness_keeps_tumor_markers_context_only(tmp_path):
    output = tmp_path / "readiness.json"

    report = build_public_biomarker_dataset_readiness(output_path=str(output), live_enrich=False)

    assert output.exists()
    assert report["status"] in {"strong", "acceptable"}
    assert report["summary"]["biomarker_external_candidate_count"] >= 4
    assert report["summary"]["tumor_marker_response_train_ready"] == 0
    assert report["summary"]["tumor_marker_policy"] == "context_only_until_longitudinal_treatment_response_data_exists"
    assert report["retraining_decision"]["production_retrain_now"] is False
    assert "clinical validation" in report["claim_boundary"]


def test_public_biomarker_dataset_readiness_names_expected_sources(tmp_path):
    report = build_public_biomarker_dataset_readiness(output_path=str(tmp_path / "readiness.json"), live_enrich=False)
    by_id = {dataset["id"]: dataset for dataset in report["datasets"]}

    for dataset_id in (
        "breastdcedl",
        "metabric_cbioportal",
        "tcga_brca_pan_can_atlas",
        "nci_edrn_breast_reference_set",
        "nci_tumor_marker_common_use",
    ):
        assert dataset_id in by_id
        assert by_id[dataset_id]["source_url"].startswith("https://")

    assert by_id["breastdcedl"]["outcomes"] == ["pathologic complete response"]
    assert "CA15-3" in by_id["nci_edrn_breast_reference_set"]["tumor_marker_fields"]
    assert "Must not be used to train response predictors." in by_id["nci_tumor_marker_common_use"]["limitations"]
