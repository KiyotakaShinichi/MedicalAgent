import json

from backend.services.rag_paired_statistical_comparison import (
    _holm_adjust,
    build_rag_paired_statistical_comparison,
)


def _case(case_id, recall, source_tier, latency):
    return {
        "case_id": case_id,
        "recall_at_10": recall,
        "mrr": recall,
        "ndcg_at_10": recall,
        "citation_precision": recall,
        "claim_supported": bool(recall),
        "unsupported_context": not bool(recall),
        "refusal_correct": True,
        "source_tier_correct": source_tier,
        "latency_ms": latency,
    }


def test_paired_comparison_preserves_negative_full_stack_result(tmp_path):
    output = tmp_path / "result.json"
    doc = tmp_path / "result.md"
    result = build_rag_paired_statistical_comparison(
        output_path=output,
        doc_path=doc,
        bootstrap_replicates=300,
        permutation_replicates=500,
    )
    assert result["clinical_validation"] is False
    assert result["external_validation"] is False
    assert (
        result["headline"]["full_stack_improvement_proven_vs_bm25"] is False
    )
    assert (
        result["headline"]["full_stack_recall_at_10_favorable_delta"] < 0
    )
    assert output.exists()
    assert doc.exists()


def test_paired_comparison_can_prove_large_consistent_primary_lift(tmp_path):
    cases = [f"c{index}" for index in range(30)]
    configs = {}
    for name in (
        "bm25_only",
        "hybrid_rrf_query_rewrite",
        "hybrid_rrf_query_rewrite_parent_child_source_tier",
        "hybrid_rrf_query_rewrite_parent_child_source_tier_pruned",
    ):
        is_bm25 = name == "bm25_only"
        configs[name] = {
            "cases": [
                _case(case_id, 0.0 if is_bm25 else 1.0, True, 50)
                for case_id in cases
            ]
        }
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "goldset_path": "frozen.jsonl",
                "total_n": 30,
                "was_used_for_tuning": False,
                "internal_vs_external_authored": "internal",
                "configurations": configs,
            }
        ),
        encoding="utf-8",
    )
    result = build_rag_paired_statistical_comparison(
        input_path=source,
        output_path=tmp_path / "out.json",
        doc_path=tmp_path / "out.md",
        bootstrap_replicates=300,
        permutation_replicates=1000,
    )
    full = next(
        row
        for row in result["comparisons"]
        if row["id"] == "full_governed_stack_vs_bm25"
    )
    assert full["improvement_proven"] is True
    assert full["metrics"]["recall_at_10"]["favorable_delta_ci95"][0] > 0


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = _holm_adjust({"a": 0.001, "b": 0.02, "c": 0.2})
    assert 0 <= adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= 1
