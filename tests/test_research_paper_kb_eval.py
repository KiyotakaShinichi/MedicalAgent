import json
from pathlib import Path

from backend.services.research_paper_kb_eval import (
    CASES_PATH,
    CONFIG_IDS,
    _load_cases,
    _paired_recall_comparison,
    _score_case,
    build_research_paper_kb_audit,
)


def test_case_bank_is_internal_non_tuning_and_covers_each_manifest_paper():
    cases = _load_cases(CASES_PATH)
    expected = {case["expected_pmcid"] for case in cases if case["expected_pmcid"]}
    assert len(cases) == 44
    assert len(expected) == 21
    assert all(case["was_used_for_tuning"] is False for case in cases)
    assert {case["category"] for case in cases} == {
        "exact_title",
        "topic_paraphrase",
        "negative_result",
        "no_research_evidence",
    }
    assert any("taglish" in case["style"] for case in cases)


def test_audit_rejects_pmc_identity_borrowed_by_curated_source():
    chunks = [
        {
            "id": "paper",
            "pmcid": "PMC12345",
            "source_path": "KnowledgeBase/raw/research_papers/PMC12345.txt",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/",
            "section": "abstract",
        },
        {
            "id": "curated",
            "pmcid": "PMC12345",
            "source_path": "KnowledgeBase/raw/curated_summary.txt",
            "source_url": "https://example.invalid/summary",
            "section": "summary",
        },
    ]
    manifest = {
        "items": [
            {
                "pmcid": "PMC12345",
                "title": "Fixture",
                "topic": "fixture",
                "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/",
            }
        ]
    }
    cases = [{"expected_pmcid": "PMC12345"}]
    report = build_research_paper_kb_audit(chunks, manifest, cases)
    assert report["status"] == "needs_attention"
    assert report["summary"]["false_pmcid_identity_chunk_count"] == 1
    assert report["clinical_validation"] is False
    assert report["independent_literature_review"] is False


def test_case_scoring_requires_paper_identity_section_and_provenance():
    case = {
        "case_id": "fixture",
        "query": "fixture",
        "category": "section_anchor",
        "style": "formal",
        "expected_pmcid": "PMC12345",
        "expected_section": "methods",
    }
    rows = [
        {
            "pmcid": "PMC12345",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/",
            "source_path": "KnowledgeBase/raw/research_papers/PMC12345.txt",
            "title": "Fixture",
            "section": "methods",
            "source_tier": "T2",
            "doi": "10.0000/fixture",
            "publication_date": "2026 Jan",
            "license": "CC BY",
            "patient_facing_suitability": "education_with_boundary",
        }
    ]
    result = _score_case(case, rows, 4.2, source_tier_filtered=True)
    assert result["recall_at_5"] == 1.0
    assert result["top1_paper_correct"] is True
    assert result["section_hit"] is True
    assert result["provenance_complete_relevant_chunk_count"] == 1
    assert result["failure_reasons"] == []


def test_no_evidence_case_counts_research_return_as_false_attribution():
    case = {
        "case_id": "fixture-none",
        "query": "prove an unsupported premise",
        "category": "no_research_evidence",
        "style": "boundary",
        "expected_pmcid": None,
        "expected_section": None,
    }
    result = _score_case(
        case,
        [{"pmcid": "PMC12345", "source_tier": "T2"}],
        1.0,
        source_tier_filtered=True,
    )
    assert result["false_paper_attribution"] is True
    assert "research_paper_returned_for_no_evidence_boundary" in result["failure_reasons"]


def test_paired_comparison_reports_uncertainty_instead_of_just_delta():
    baseline = [
        {"case_id": "a", "expected_pmcid": "PMC1", "recall_at_10": 1.0},
        {"case_id": "b", "expected_pmcid": "PMC2", "recall_at_10": 0.0},
    ]
    candidate = [
        {"case_id": "a", "expected_pmcid": "PMC1", "recall_at_10": 1.0},
        {"case_id": "b", "expected_pmcid": "PMC2", "recall_at_10": 1.0},
    ]
    report = _paired_recall_comparison(baseline, candidate)
    assert report["recall_at_10_delta"] == 0.5
    assert len(report["bootstrap_ci95"]) == 2
    assert 0 <= report["exact_p_value"] <= 1


def test_evaluator_compares_simple_and_governed_retrieval_stacks():
    assert CONFIG_IDS == (
        "bm25_only",
        "faiss_dense_only",
        "hybrid_rrf",
        "hybrid_rrf_query_rewrite",
        "hybrid_rrf_query_rewrite_parent_child",
        "hybrid_rrf_query_rewrite_parent_child_source_tier",
    )


def test_evaluator_source_does_not_invoke_live_patient_pipeline():
    source = Path("backend/services/research_paper_kb_eval.py").read_text(encoding="utf-8")
    assert "run_patient_agent_pipeline" not in source
    assert '"live_patient_route_changed": False' in source


def test_case_bank_is_valid_jsonl_with_unique_ids():
    rows = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["case_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_expanded_case_bank_is_frozen_and_not_an_independent_holdout():
    rows = _load_cases(CASES_PATH)
    assert CASES_PATH.name == "research_paper_grounding_cases_v2.jsonl"
    assert all(row["was_used_for_tuning"] is False for row in rows)
    assert any(row["style"] == "taglish" for row in rows)
    assert sum(row["expected_pmcid"] is None for row in rows) == 8
