from backend.services.structured_claim_shadow_eval import (
    align_claim_to_source,
    build_structured_claim_shadow_eval,
    split_atomic_claims,
)


def test_atomic_claim_splitter_separates_compound_answer():
    claims = split_atomic_claims("The record contains a lab result. It does not establish a diagnosis; ask the care team.")
    assert len(claims) == 3


def test_numeric_unit_and_source_policy_mismatches_are_not_supported():
    numeric = align_claim_to_source(
        "The dose is 10 mg.", "The dose is 10 mL.",
        source_tier="T2", allowed_use="record_explanation", patient_facing=True,
    )
    blocked = align_claim_to_source(
        "Internal review categories are listed.", "Internal review categories are listed.",
        source_tier="T4", allowed_use="clinician_only", patient_facing=True,
    )
    assert numeric["status"] == "contradicted"
    assert numeric["numeric_facts_match"] is False
    assert blocked["status"] == "source_policy_blocked"


def test_shadow_eval_is_complete_but_not_live_or_clinical(tmp_path):
    report = build_structured_claim_shadow_eval(tmp_path / "claims.json")
    assert report["n_cases"] == 12
    assert report["passed_n"] == 12
    assert report["status"] == "strong"
    assert report["live_patient_agent_enabled"] is False
    assert report["clinical_validation"] is False
