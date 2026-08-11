from backend.services.section_aware_retrieval import (
    canonical_section,
    infer_preferred_sections,
    rerank_by_section,
)


def test_section_aliases_and_multilingual_cues():
    assert canonical_section("Patients and Methods") == "methods"
    assert infer_preferred_sections("What were the study results?") == ("results",)
    assert infer_preferred_sections("Aling source o anong paper ito?") == ("abstract",)


def test_section_reranker_promotes_requested_section_without_dropping_rows():
    rows = [
        {"id": "intro", "section": "introduction", "retrieval_score": 0.70},
        {"id": "results", "section": "results", "retrieval_score": 0.65},
        {"id": "refs", "section": "references", "retrieval_score": 0.80},
    ]
    ranked = rerank_by_section("What did the study find in its results?", rows)
    assert ranked[0]["id"] == "results"
    assert {row["id"] for row in ranked} == {"intro", "results", "refs"}
    assert all("section_rerank_original_rank" in row for row in ranked)


def test_no_section_cue_preserves_base_order():
    rows = [
        {"id": "first", "section": "abstract", "retrieval_score": 0.9},
        {"id": "second", "section": "results", "retrieval_score": 0.8},
    ]
    assert [row["id"] for row in rerank_by_section("breast cancer paper", rows)] == ["first", "second"]
