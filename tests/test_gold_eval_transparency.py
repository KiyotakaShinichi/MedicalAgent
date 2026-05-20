from __future__ import annotations

from backend.services.governance_readiness_artifacts import write_rag_gold_claim_grounding_cases


REQUIRED_CASE_FIELDS = {
    "case_id",
    "user_query",
    "expected_intent",
    "allowed_answer_scope",
    "required_source_tiers",
    "gold_supported_claims",
    "unsupported_claims",
    "contradiction_traps",
    "expected_refusal_or_escalation",
    "expected_citation_requirements",
    "pass_criteria",
    "fail_criteria",
    "authored_by",
    "authored_date",
    "was_used_for_tuning",
    "internal_vs_external_authored",
    "contamination_disclosure",
    "baseline_version",
    "release_id",
}


def test_rag_gold_claim_grounding_cases_are_transparent():
    artifact = write_rag_gold_claim_grounding_cases()

    assert artifact["status"] == "strong"
    assert artifact["summary"]["case_count"] >= 10
    assert artifact["summary"]["n_size"] == artifact["summary"]["case_count"]
    assert artifact["summary"]["pass_count"] == artifact["summary"]["case_count"]
    assert artifact["summary"]["fail_count"] == 0
    assert artifact["summary"]["authored_by"] == "engineering"
    assert artifact["summary"]["was_used_for_tuning"] is True
    assert "contamination" in artifact["summary"]["contamination_disclosure"].lower()

    categories = {case["category"] for case in artifact["cases"]}
    assert {
        "breast_cancer_education",
        "cbc_lab_explanation",
        "urgent_symptom",
        "genetics_vus",
        "tumor_marker_limitations",
        "taglish_code_switch",
        "near_boundary",
    } <= categories

    for case in artifact["cases"]:
        assert REQUIRED_CASE_FIELDS <= set(case)
        assert case["pass_criteria"]
        assert case["fail_criteria"]
        assert case["contradiction_traps"]
