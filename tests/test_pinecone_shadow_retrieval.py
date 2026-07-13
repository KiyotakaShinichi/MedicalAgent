from __future__ import annotations

import json

from backend.services.pinecone_shadow_retrieval import build_pinecone_shadow_retrieval_comparison


def test_pinecone_shadow_retrieval_not_configured_by_default(tmp_path):
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "pinecone.json"
    doc = tmp_path / "pinecone.md"
    _write_baseline(baseline)

    report = build_pinecone_shadow_retrieval_comparison(
        baseline_path=baseline,
        output_path=output,
        doc_path=doc,
        env={},
    )

    assert output.exists()
    assert doc.exists()
    assert report["status"] == "ready_for_shadow_mode_not_configured"
    assert report["pinecone_config"]["configured"] is False
    assert report["comparison_completed"] is False
    assert report["clinical_validation"] is False
    assert report["phi_allowed"] is False
    assert report["live_patient_route_enabled"] is False
    assert report["promotion_gate"]["pinecone_can_replace_local_retrieval"] is False


def test_pinecone_shadow_retrieval_configured_still_dry_run_without_network(tmp_path):
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline)

    report = build_pinecone_shadow_retrieval_comparison(
        baseline_path=baseline,
        output_path=tmp_path / "pinecone.json",
        doc_path=tmp_path / "pinecone.md",
        env={
            "PINECONE_ENABLED": "true",
            "PINECONE_API_KEY": "test-key",
            "PINECONE_INDEX_HOST": "https://example-index.svc.pinecone.io",
            "PINECONE_NAMESPACE_KB": "nlcare_kb_demo_t1_t3",
        },
    )

    assert report["status"] == "configured_dry_run_only"
    assert report["pinecone_config"]["configured"] is True
    assert report["network_execution_allowed"] is False
    assert report["comparison_completed"] is False


def test_pinecone_shadow_retrieval_extracts_local_reference_metrics(tmp_path):
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline)

    report = build_pinecone_shadow_retrieval_comparison(
        baseline_path=baseline,
        output_path=tmp_path / "pinecone.json",
        doc_path=tmp_path / "pinecone.md",
        env={},
    )

    local = report["local_reference_metrics"]
    assert local["bm25_only"]["recall_at_10"] == 0.8
    assert local["source_governed_full_stack"]["source_tier_correctness"] == 1.0
    assert "raw recall superiority over BM25 is not proven" in local["current_honest_reading"]


def test_pinecone_shadow_retrieval_preserves_governance_contract(tmp_path):
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline)

    report = build_pinecone_shadow_retrieval_comparison(
        baseline_path=baseline,
        output_path=tmp_path / "pinecone.json",
        doc_path=tmp_path / "pinecone.md",
        env={},
    )

    assert "source_tier" in report["metadata_filter_contract"]
    assert "allowed_use" in report["metadata_filter_contract"]
    assert "patient_facing" in report["metadata_filter_contract"]
    assert report["namespace_contract"]["patient_data"] == "disallowed until compliance/security review"
    assert "HIPAA compliance" in report["blocked_claims"]


def _write_baseline(path) -> None:
    payload = {
        "status": "acceptable",
        "total_n": 2,
        "clinical_validation": False,
        "configurations": {
            "bm25_only": {
                "summary": {
                    "recall_at_5": 0.7,
                    "recall_at_10": 0.8,
                    "mrr": 0.6,
                    "ndcg_at_10": 0.5,
                    "citation_precision": 0.4,
                    "claim_support_rate": 0.9,
                    "unsupported_context_rate": 0.1,
                    "refusal_correctness": 1.0,
                    "source_tier_correctness": 0.5,
                    "latency_p50_ms": 30,
                    "latency_p95_ms": 70,
                }
            },
            "hybrid_rrf_query_rewrite_parent_child_source_tier": {
                "summary": {
                    "recall_at_5": 0.71,
                    "recall_at_10": 0.78,
                    "mrr": 0.62,
                    "ndcg_at_10": 0.55,
                    "citation_precision": 0.52,
                    "claim_support_rate": 0.84,
                    "unsupported_context_rate": 0.16,
                    "refusal_correctness": 1.0,
                    "source_tier_correctness": 1.0,
                    "latency_p50_ms": 270,
                    "latency_p95_ms": 500,
                }
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
