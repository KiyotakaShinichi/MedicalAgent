from __future__ import annotations

from backend.services.external_dataset_integration_matrix import (
    BLOCKED_GLOBAL_CLAIMS,
    build_external_dataset_integration_matrix,
)


def test_external_dataset_integration_matrix_writes_artifact_and_doc(tmp_path):
    output = tmp_path / "matrix.json"
    doc = tmp_path / "matrix.md"

    report = build_external_dataset_integration_matrix(output_path=output, doc_path=doc)

    assert output.exists()
    assert doc.exists()
    assert report["status"] == "strong"
    assert report["clinical_validation"] is False
    assert report["production_training_allowed"] is False
    assert report["dataset_count"] >= 15
    assert "clinical validation" in report["claim_boundary"]


def test_external_dataset_integration_matrix_names_expected_datasets(tmp_path):
    report = build_external_dataset_integration_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=tmp_path / "matrix.md",
    )
    by_id = {row["dataset_id"]: row for row in report["datasets"]}

    expected = {
        "breastdcedl",
        "ispy1_tcia",
        "ispy2_tcia",
        "duke_breast_mri_tcia",
        "mama_mia",
        "aacr_genie_bpc_brca",
        "tcga_brca_gdc",
        "metabric_cbioportal",
        "cptac_breast",
        "seer_breast",
        "seer_medicare",
        "mimic_iv",
        "clinvar",
        "brca_exchange",
        "nci_edrn_breast_reference_set",
    }
    assert expected <= set(by_id)
    for dataset_id in expected:
        assert by_id[dataset_id]["source_url"].startswith("https://")
        assert by_id[dataset_id]["clinical_validation"] is False


def test_highest_roi_sequence_prioritizes_external_stress_not_clinical_claims(tmp_path):
    report = build_external_dataset_integration_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=tmp_path / "matrix.md",
    )

    top = report["highest_roi_next_integrations"]
    assert top[0]["dataset_id"] == "breastdcedl"
    assert "stress" in top[0]["action"]
    assert "mimic_iv" in {item["dataset_id"] for item in top}
    assert "clinvar" in {item["dataset_id"] for item in top}
    assert report["recommended_sequence"][0].startswith("Integrate metadata-only")


def test_genetics_and_tumor_marker_sources_remain_boundary_only(tmp_path):
    report = build_external_dataset_integration_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=tmp_path / "matrix.md",
    )
    by_id = {row["dataset_id"]: row for row in report["datasets"]}

    assert by_id["clinvar"]["integration_category"] == "genetics_vus_safety_boundary"
    assert "Do not interpret" in by_id["clinvar"]["not_allowed_use"]
    assert by_id["brca_exchange"]["integration_category"] == "genetics_vus_safety_boundary"
    assert "patient-facing risk estimates" in by_id["brca_exchange"]["not_allowed_use"]
    assert by_id["nci_edrn_breast_reference_set"]["integration_category"] == "tumor_marker_limitation_context"
    assert "Do not train response predictors" in by_id["nci_edrn_breast_reference_set"]["not_allowed_use"]


def test_restricted_or_credentialed_datasets_are_not_marked_ready_for_training(tmp_path):
    report = build_external_dataset_integration_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=tmp_path / "matrix.md",
    )
    by_id = {row["dataset_id"]: row for row in report["datasets"]}

    assert by_id["seer_medicare"]["access_type"] == "restricted_application_required"
    assert "Leave in access-packet stage" in by_id["seer_medicare"]["next_integration_step"]
    assert by_id["mimic_iv"]["access_type"] == "credentialed_physionet_access"
    assert "synthetic-noise" in by_id["mimic_iv"]["recommended_use"]
    assert "Do not use MIMIC-IV to train breast cancer response" in by_id["mimic_iv"]["not_allowed_use"]


def test_blocked_claims_cover_medical_overclaim_surface(tmp_path):
    report = build_external_dataset_integration_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=tmp_path / "matrix.md",
    )

    assert set(BLOCKED_GLOBAL_CLAIMS) == set(report["blocked_global_claims"])
    for claim in (
        "clinical validation",
        "treatment recommendation",
        "prognosis or survival prediction",
        "genetic-risk interpretation",
        "tumor-marker interpretation",
    ):
        assert claim in report["blocked_global_claims"]


def test_document_keeps_non_clinical_boundary_visible(tmp_path):
    doc = tmp_path / "matrix.md"
    build_external_dataset_integration_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=doc,
    )

    text = doc.read_text(encoding="utf-8").lower()
    assert "does not authorize model promotion" in text
    assert "clinical validation" in text
    assert "not allowed use" in text
