from __future__ import annotations

from backend.services.public_treatment_dataset_readiness import build_public_treatment_dataset_readiness


def test_public_treatment_dataset_readiness_tracks_combinations_without_recommendations(tmp_path):
    output = tmp_path / "treatment_readiness.json"

    report = build_public_treatment_dataset_readiness(output_path=str(output))

    assert output.exists()
    assert report["status"] in {"strong", "acceptable"}
    assert report["summary"]["treatment_combination_candidate_count"] >= 5
    assert report["summary"]["immediate_full_treatment_combo_training_ready"] == 0
    assert report["summary"]["best_future_real_world_treatment_dataset"] == "aacr_genie_bpc_brca"
    assert "does not recommend any treatment" in report["claim_boundary"]

    modality_ids = {entry["id"] for entry in report["treatment_modality_schema"]}
    assert {"chemotherapy", "radiation", "targeted_anti_her2", "endocrine", "surgery"}.issubset(modality_ids)
    combo_ids = {entry["id"] for entry in report["treatment_combination_patterns"]}
    assert "chemo_plus_targeted" in combo_ids
    assert "chemo_targeted_surgery_radiation_endocrine" in combo_ids


def test_public_treatment_dataset_readiness_names_expected_sources(tmp_path):
    report = build_public_treatment_dataset_readiness(output_path=str(tmp_path / "treatment_readiness.json"))
    by_id = {dataset["id"]: dataset for dataset in report["datasets"]}

    for dataset_id in (
        "aacr_genie_bpc_brca",
        "seer_breast",
        "seer_medicare",
        "breastdcedl_ispy2",
        "duke_breast_mri_tcia",
        "tcga_brca_gdc",
        "clinicaltrials_gov_breast_protocols",
    ):
        assert dataset_id in by_id
        assert by_id[dataset_id]["source_url"].startswith("https://")

    assert "HER2-directed treatment histories" in by_id["aacr_genie_bpc_brca"]["treatment_fields"]
    assert by_id["clinicaltrials_gov_breast_protocols"]["readiness"] == "vocabulary_only_not_patient_training"
