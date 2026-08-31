from __future__ import annotations

import json
from pathlib import Path

from backend.services.managed_vector_shadow_comparison import (
    FULL_STACK_ID,
    _joint_improvement,
    build_managed_vector_shadow_comparison,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_default_run_is_honest_readiness_without_network(tmp_path: Path):
    report = build_managed_vector_shadow_comparison(
        root_dir=ROOT_DIR,
        output_path=tmp_path / "comparison.json",
        failure_path=tmp_path / "failures.json",
    )
    assert report["status"] == "ready_for_managed_shadow_run"
    assert report["comparison_completed"] is False
    assert report["managed_network_request_performed"] is False
    assert report["retrieval_improvement_proven"] is False
    assert report["candidate_decision"] == "HOLD"
    assert report["clinical_validation"] is False


def test_injected_complete_results_produce_comparison_without_network(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    cases = [
        {
            "case_id": "case-1",
            "query": "education",
            "user_query": "education",
            "expected_source_ids": ["source-a"],
            "acceptable_source_tiers": ["T1", "T2", "T3"],
            "expected_intent": "education",
            "expected_refusal_or_insufficient_evidence": False,
            "clinical_validation": False,
            "was_used_for_tuning": False,
        },
        {
            "case_id": "case-2",
            "query": "boundary",
            "user_query": "boundary",
            "expected_source_ids": ["source-b"],
            "acceptable_source_tiers": ["T1", "T2", "T3"],
            "expected_intent": "diagnosis_refusal",
            "expected_refusal_or_insufficient_evidence": True,
            "clinical_validation": False,
            "was_used_for_tuning": False,
        },
    ]
    goldset.write_text(
        "\n".join(json.dumps(case) for case in cases) + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    summary = {
        "case_count": 2,
        "recall_at_5": 0.5,
        "recall_at_10": 0.5,
        "mrr": 0.5,
        "ndcg_at_10": 0.5,
        "citation_precision": 0.5,
        "claim_support_rate": 0.5,
        "unsupported_context_rate": 0.5,
        "refusal_correctness": 1.0,
        "source_tier_correctness": 1.0,
        "latency_p50_ms": 10.0,
        "latency_p95_ms": 12.0,
    }
    baseline.write_text(
        json.dumps(
            {
                "configurations": {
                    FULL_STACK_ID: {"summary": summary},
                    "bm25_only": {"summary": summary},
                }
            }
        ),
        encoding="utf-8",
    )
    metadata_a = {
        "source_id": "source-a",
        "chunk_id": "chunk-a",
        "parent_id": "source-a",
        "source_tier": "T2",
        "allowed_use": ["education"],
        "patient_facing": True,
        "data_scope": "curated_non_patient_kb",
        "clinical_validation": False,
    }
    metadata_b = {
        **metadata_a,
        "source_id": "source-b",
        "chunk_id": "chunk-b",
        "parent_id": "source-b",
        "source_tier": "T1",
    }
    report = build_managed_vector_shadow_comparison(
        root_dir=tmp_path,
        goldset_path=goldset,
        local_baseline_path=baseline,
        output_path=tmp_path / "comparison.json",
        failure_path=tmp_path / "failures.json",
        managed_case_results={
            "case-1": [{"record_id": "chunk-a", "score": 1.0, "metadata": metadata_a}],
            "case-2": [{"record_id": "chunk-b", "score": 1.0, "metadata": metadata_b}],
        },
        managed_case_latencies_ms={"case-1": 20.0, "case-2": 22.0},
    )
    assert report["comparison_completed"] is True
    assert report["managed_network_request_performed"] is False
    assert report["managed_summary"]["recall_at_10"] == 1.0
    assert report["quality_governance_joint_improvement_proven"] is False
    assert report["candidate_decision"] == "HOLD"
    assert report["operational_evidence"]["measured_cost_usd"] is None


def test_joint_improvement_requires_quality_and_governance_together():
    local = {
        "recall_at_10": 0.7,
        "citation_precision": 0.6,
        "unsupported_context_rate": 0.2,
    }
    managed = {
        "recall_at_10": 0.8,
        "citation_precision": 0.7,
        "unsupported_context_rate": 0.1,
        "source_tier_correctness": 1.0,
        "refusal_correctness": 1.0,
    }
    assert _joint_improvement(managed, local) is True
    assert _joint_improvement({**managed, "source_tier_correctness": 0.9}, local) is False
