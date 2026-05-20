from backend.services.fhir_like_mapper import (
    build_fhir_alignment_readiness,
    map_cbc_observation,
    map_family_history,
    map_imaging_report,
    map_medication_statement,
)


def test_cbc_observation_mapping_allows_codes_and_units():
    row = {"pre_wbc": 5.1, "treatment_date": "2026-01-01"}
    obs = map_cbc_observation(row, "wbc")
    assert obs.resource_type == "ObservationLike"
    assert obs.coding.code == "6690-2"
    assert obs.unit == "10^9/L"
    assert obs.value == 5.1


def test_other_canonical_mappers_return_unreviewed_readiness_objects():
    row = {
        "regimen": "synthetic regimen",
        "modality": "MRI",
        "findings": "Synthetic finding",
        "impression": "Synthetic impression",
        "relationship": "mother",
        "cancer_type": "breast cancer",
        "age_at_diagnosis": 52,
    }
    assert map_medication_statement(row).resource_type == "MedicationStatementLike"
    assert map_imaging_report(row).resource_type == "DiagnosticReportLike"
    assert map_family_history(row).resource_type == "FamilyMemberHistoryLike"


def test_fhir_alignment_readiness_artifact_shape(tmp_path):
    payload = build_fhir_alignment_readiness(tmp_path / "fhir.json")
    assert payload["mapping_coverage"] >= 0.8
    assert "not certified FHIR" in payload["claim_boundary"]
