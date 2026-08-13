from backend.services.rag_governance_tradeoff import build_rag_governance_tradeoff


def test_tradeoff_report_keeps_negative_result_and_boundaries() -> None:
    artifact = build_rag_governance_tradeoff()
    assert artifact["improvement_proven_vs_bm25"] is False
    assert artifact["tradeoffs"]["raw_retrieval_improvement_proven"] is False
    assert artifact["raw_retrieval_superiority_status"] == "not_proven"
    assert artifact["full_source_governed_stack"]["source_tier_correctness"] == 1.0
    selection = artifact["policy_constrained_selection"]
    constraints_pass = all(
        constraint["passed"] for constraint in selection["constraints"].values()
    )
    assert artifact["status"] == ("acceptable" if constraints_pass else "needs_attention")
    assert artifact["status_basis"] == (
        "acceptable_for_internal_policy_constrained_use"
        if constraints_pass
        else "patient_facing_governance_constraints_not_met"
    )
    assert selection["selected_for_raw_retrieval_superiority"] is False
    assert selection["selected_for_patient_facing_source_governance"] is True
    assert selection["all_constraints_pass"] is constraints_pass
    assert artifact["external_holdout"]["completed"] is False
    assert artifact["clinical_validation"] is False
    assert "internal" in artifact["claim_boundary"]
