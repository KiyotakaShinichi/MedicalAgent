from backend.services.clinical_ontology import (
    ONTOLOGY_VERSION,
    ontology_manifest,
    validate_record_against_ontology,
)
from backend.services.medical_claim_boundary import classify_medical_claim
from backend.services.medical_evidence_standards import standard_for, standards_manifest
from backend.services.medical_safety_contract import build_medical_safety_contract
from backend.services.post_generation_validator import validate_reply


def test_clinical_ontology_validates_allowed_values():
    result = validate_record_against_ontology(
        "imaging",
        {"modality": "MRI", "report_type": "follow_up"},
    )
    assert result["status"] == "passed"
    assert result["normalized"]["modality"] == "mri"
    assert result["ontology_version"] == ONTOLOGY_VERSION


def test_clinical_ontology_flags_unknown_values():
    result = validate_record_against_ontology(
        "genetic_test",
        {"test_type": "crystal_ball", "gene": "BRCA1"},
    )
    assert result["status"] == "needs_review"
    assert "invalid_value:test_type" in result["issues"]


def test_minimum_evidence_standard_names_response_regression():
    standard = standard_for("response_regression")
    assert standard["required_all"] == ("demographics",)
    assert "score" in standard["answer_policy"].lower()


def test_medical_claim_boundary_blocks_treatment_advice():
    result = classify_medical_claim("You should stop chemotherapy and take turmeric.")
    assert result["decision"] == "blocked"
    assert "treatment_recommendation" in result["blocked_claim_types"]


def test_post_generation_validator_surfaces_claim_boundary():
    result = validate_reply("Your elevated CA 15-3 means recurrence.")
    assert result.decision == "blocked"
    assert result.claim_boundary is not None
    assert "tumor_marker_overclaim" in result.claim_boundary["blocked_claim_types"]


def test_medical_safety_contract_builds_manifest(tmp_path):
    path = tmp_path / "contract.json"
    payload = build_medical_safety_contract(output_path=str(path))
    assert payload["status"] == "strong"
    assert path.exists()
    assert ontology_manifest()["version"] == payload["clinical_ontology"]["version"]
    assert standards_manifest()["version"] == payload["minimum_evidence_standards"]["version"]
