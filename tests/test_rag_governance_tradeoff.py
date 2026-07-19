from backend.services.rag_governance_tradeoff import build_rag_governance_tradeoff


def test_tradeoff_report_keeps_negative_result_and_boundaries() -> None:
    artifact = build_rag_governance_tradeoff()
    assert artifact["status"] == "needs_attention"
    assert artifact["improvement_proven_vs_bm25"] is False
    assert artifact["tradeoffs"]["raw_retrieval_improvement_proven"] is False
    assert artifact["full_source_governed_stack"]["source_tier_correctness"] == 1.0
    assert artifact["external_holdout"]["completed"] is False
    assert artifact["clinical_validation"] is False
    assert "internal" in artifact["claim_boundary"]
