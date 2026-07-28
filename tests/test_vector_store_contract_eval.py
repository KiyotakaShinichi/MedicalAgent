from __future__ import annotations

import json
from pathlib import Path

from backend.services.vector_store_contract_eval import build_vector_store_contract_eval


def test_contract_eval_uses_generated_gold_records_without_network(tmp_path: Path):
    gold = tmp_path / "Data" / "lakehouse" / "gold" / "vector_records.jsonl"
    gold.parent.mkdir(parents=True)
    gold.write_text(
        json.dumps(
            {
                "record_id": "chunk-a",
                "embedding_input": "education fixture",
                "namespace": "nlcare_kb_demo_t1_t3",
                "metadata": {
                    "source_id": "source-a",
                    "chunk_id": "chunk-a",
                    "parent_id": "source-a",
                    "source_tier": "T2",
                    "allowed_use": ["education"],
                    "patient_facing": True,
                    "staleness_status": "current",
                    "kb_fingerprint": "fixture",
                    "doc_type": "knowledge_chunk",
                    "data_scope": "curated_non_patient_kb",
                    "clinical_validation": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_vector_store_contract_eval(root_dir=tmp_path)
    assert report["status"] == "strong_contract_only"
    assert report["managed_network_request_performed"] is False
    assert report["managed_vector_comparison_completed"] is False
    assert report["retrieval_improvement_proven"] is False
    assert report["gold_record_count"] == 1
