import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "Data/evals/rag/latest_rag_baseline_comparison.json"
FAILURES_PATH = ROOT / "Data/evals/rag/latest_rag_baseline_failures.json"


def test_rag_baseline_comparison_artifact_schema():
    payload = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "rag_baseline_comparison_v1"
    assert payload["clinical_validation"] is False
    assert payload["was_used_for_tuning"] is False
    assert payload["total_n"] >= 50

    rows = payload["rows"]
    canonical = [
        "bm25_only",
        "faiss_dense_only",
        "hybrid_rrf",
        "hybrid_rrf_query_rewrite",
        "hybrid_rrf_query_rewrite_parent_child",
        "hybrid_rrf_query_rewrite_parent_child_source_tier",
    ]
    assert [row["configuration"] for row in rows[:6]] == canonical
    assert all(row["experimental"] is False for row in rows[:6])
    assert rows[6]["configuration"] == (
        "hybrid_rrf_query_rewrite_parent_child_source_tier_pruned"
    )
    assert rows[6]["experimental"] is True
    assert rows[6]["positioning"] == "negative_result_not_promoted"
    for row in rows:
        for metric in (
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
            "citation_precision",
            "claim_support_rate",
            "unsupported_context_rate",
            "refusal_correctness",
            "source_tier_correctness",
            "latency_p50_ms",
            "latency_p95_ms",
            "failure_count",
        ):
            assert metric in row


def test_rag_baseline_failures_artifact_schema():
    payload = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "rag_baseline_failures_v1"
    assert payload["clinical_validation"] is False
    assert payload["total_n"] >= 50
    assert isinstance(payload["failures"], list)
    assert isinstance(payload["by_configuration"], dict)
    assert isinstance(payload["by_reason"], dict)
