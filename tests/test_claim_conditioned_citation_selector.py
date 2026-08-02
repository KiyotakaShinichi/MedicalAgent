from backend.services.claim_conditioned_citation_selector import (
    select_citations_for_claims,
)


def _chunks():
    return [
        {
            "source_id": "her2-basics",
            "text": "HER2-positive describes breast cancer cells with higher HER2 protein levels.",
            "source_tier": "T1",
            "allowed_use": "general_patient_education",
            "retrieval_score": 0.82,
        },
        {
            "source_id": "unrelated-cbc",
            "text": "A complete blood count includes white blood cells and platelets.",
            "source_tier": "T1",
            "allowed_use": "general_patient_education",
            "retrieval_score": 0.95,
        },
        {
            "source_id": "clinician-protocol",
            "text": "HER2 protocol treatment selection and dosing.",
            "source_tier": "T4",
            "allowed_use": "clinician_only",
            "retrieval_score": 1.0,
        },
    ]


def test_selects_claim_aligned_source_over_higher_ranked_distractor():
    result = select_citations_for_claims(
        ["HER2-positive refers to higher HER2 protein levels in the cancer cells."],
        _chunks(),
    )
    assert result["selected_citation_ids"] == ["her2-basics"]
    assert result["all_claims_supported_by_proxy"] is True
    assert result["support_proxy_is_entailment"] is False


def test_drops_disallowed_and_stale_sources():
    chunks = _chunks() + [{
        "source_id": "stale-her2",
        "text": "HER2-positive means higher HER2 protein levels.",
        "source_tier": "T1",
        "stale": True,
        "retrieval_score": 1.0,
    }]
    result = select_citations_for_claims(["HER2-positive means higher HER2 protein levels."], chunks)
    assert "clinician-protocol" not in result["selected_citation_ids"]
    assert "stale-her2" not in result["selected_citation_ids"]


def test_unsupported_claim_remains_visible_instead_of_forcing_a_citation():
    claim = "This record proves a specific survival time."
    result = select_citations_for_claims([claim], _chunks())
    assert result["selected_citation_ids"] == []
    assert result["unsupported_claims"] == [claim]
    assert result["all_claims_supported_by_proxy"] is False


def test_refusal_route_strips_all_citations():
    result = select_citations_for_claims(
        ["A treatment dose should be changed."],
        _chunks(),
        refusal_route=True,
    )
    assert result["selected_citation_ids"] == []
    assert result["reason"] == "refusal_route_strips_citations"
    assert result["live_patient_route_changed"] is False
