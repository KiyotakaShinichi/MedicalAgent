from backend.services.rag_claim_validator import validate_claims


def test_atomic_claim_splitting_handles_semicolons():
    result = validate_claims(
        "WBC may be monitored; CA 15-3 proves recurrence.",
        [{"id": "a", "text": "WBC may be monitored. CA 15-3 cannot diagnose recurrence by itself."}],
    )
    assert result.claim_count == 2
    assert result.unsupported_count >= 1


def test_numeric_mismatch_is_not_hidden_by_token_overlap():
    result = validate_claims(
        "The recommended dose is 8 mg.",
        [{"id": "a", "text": "Medication dosing requires clinician review; a 4 mg example is shown."}],
    )
    verdict = result.verdicts[0]
    assert verdict.status == "unsupported"
    assert verdict.alignment_checks["numeric_alignment"] == "failed"
    assert verdict.alignment_checks["missing_numeric_facts"] == ["8 mg"]


def test_generic_kb_cannot_support_patient_specific_diagnosis():
    result = validate_claims(
        "Your scan confirms metastatic cancer.",
        [{"id": "a", "text": "Scans may contribute evidence and require clinical interpretation."}],
    )
    verdict = result.verdicts[0]
    assert verdict.status == "unsupported"
    assert verdict.population_scope == "patient_specific"
    assert verdict.reason == "patient_specific_claim_supported_only_by_generic_evidence"


def test_patient_record_metadata_allows_scope_check_to_continue():
    result = validate_claims(
        "Your WBC is 3.1 g/dl.",
        [{"id": "record", "text": "WBC is 3.1 g/dl", "is_patient_record": True}],
    )
    verdict = result.verdicts[0]
    assert verdict.alignment_checks["patient_scope_alignment"] == "passed"
    assert verdict.alignment_checks["numeric_alignment"] == "passed"
