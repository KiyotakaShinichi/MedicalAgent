from backend.services.rag_execution_policy import govern_candidates, plan_rag_execution
from backend.services.rag_intent_modes import MODES


def test_portal_help_skips_unneeded_expansion_and_rerank():
    policy, mode = plan_rag_execution(
        intent="portal_help",
        rewritten={"expanded_query": "how do i upload a file"},
    )
    assert mode is not None and mode.mode == "portal_help_rag"
    assert policy.apply_parent_child is False
    assert policy.apply_reranker is False
    assert policy.clinical_validation is False


def test_simple_education_keeps_ranking_but_skips_broad_expansion():
    policy, _ = plan_rag_execution(intent="education", rewritten={"expanded_query": "what is a cbc"})
    assert policy.apply_parent_child is False
    assert policy.apply_reranker is True


def test_governance_runs_before_generation_and_drops_wrong_tier():
    chunks = [
        {"id": "portal-help", "parent_id": "portal-help", "title": "Portal", "source_name": "Portal", "text": "upload help"},
        {"id": "cbc-monitoring", "parent_id": "cbc-monitoring", "title": "CBC", "source_name": "Curated", "text": "CBC education"},
    ]
    kept, trace = govern_candidates(chunks, MODES["portal_help_rag"])
    assert [chunk["id"] for chunk in kept] == ["portal-help"]
    assert trace["applied_before_generation"] is True
    assert trace["dropped_count"] == 1
