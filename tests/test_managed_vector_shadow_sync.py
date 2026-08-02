from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.managed_vector_shadow_sync import (
    build_managed_vector_shadow_sync,
)
from backend.services.managed_vector_store import VectorStoreError


ROOT_DIR = Path(__file__).resolve().parents[1]


class FakeEncoder:
    def encode(self, sentences, **_kwargs):
        return [[1.0] + [0.0] * 383 for _ in sentences]


def _gold_row(record_id: str) -> dict:
    return {
        "data_contract": "vector_record_gold_v1",
        "embedding_input": f"curated non-patient fixture {record_id}",
        "namespace": "nlcare_kb_demo_t1_t3",
        "record_id": record_id,
        "metadata": {
            "allowed_use": ["education"],
            "chunk_id": record_id,
            "clinical_validation": False,
            "data_scope": "curated_non_patient_kb",
            "doc_type": "knowledge_chunk",
            "kb_fingerprint": "fixture-fingerprint",
            "parent_id": "fixture-parent",
            "patient_facing": True,
            "section": "body",
            "source_id": "fixture-source",
            "source_name": "Fixture",
            "source_tier": "T2",
            "source_url": "",
            "staleness_status": "current",
            "tags": ["education"],
            "title": "Fixture",
            "topic": "education",
        },
    }


def _write_gold(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_default_sync_is_validated_readiness_without_network():
    report = build_managed_vector_shadow_sync(root_dir=ROOT_DIR)
    expected_record_count = sum(
        bool(line.strip())
        for line in (
            ROOT_DIR / "Data/lakehouse/gold/vector_records.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    )
    assert report["status"] == "ready_for_opt_in_shadow_sync"
    assert report["record_count"] == expected_record_count
    assert expected_record_count > 0
    assert report["validation_passed"] is True
    assert report["network_request_performed"] is False
    assert report["sync_completed"] is False
    assert report["clinical_validation"] is False
    assert report["patient_data_allowed"] is False


def test_apply_requires_explicit_managed_network_gates(tmp_path):
    gold = _write_gold(tmp_path, [_gold_row("one")])
    with pytest.raises(VectorStoreError):
        build_managed_vector_shadow_sync(
            root_dir=tmp_path,
            gold_path=gold,
            apply=True,
            environment={},
            encoder=FakeEncoder(),
        )


def test_apply_checks_each_document_receipt(tmp_path):
    gold = _write_gold(tmp_path, [_gold_row("one"), _gold_row("two")])
    calls = []

    def transport(method, url, headers, payload, timeout):
        calls.append((method, url, headers, payload, timeout))
        return {
            "value": [
                {"key": row["id"], "status": True, "statusCode": 200}
                for row in payload["value"]
            ]
        }

    report = build_managed_vector_shadow_sync(
        root_dir=tmp_path,
        gold_path=gold,
        output_path=tmp_path / "report.json",
        apply=True,
        batch_size=1,
        encoder=FakeEncoder(),
        transport=transport,
        environment={
            "NLCARE_VECTOR_BACKEND": "azure_ai_search",
            "NLCARE_MANAGED_VECTOR_SHADOW_ENABLED": "true",
            "NLCARE_MANAGED_VECTOR_ALLOW_NETWORK": "true",
            "AZURE_SEARCH_ENDPOINT": "https://example.search.windows.net",
            "AZURE_SEARCH_INDEX_NAME": "nlcare-kb-shadow-v1",
            "AZURE_SEARCH_BEARER_TOKEN": "header.payload.signature",
        },
    )
    assert report["status"] == "shadow_sync_completed"
    assert report["sync_completed"] is True
    assert report["indexed_count"] == 2
    assert report["failed_count"] == 0
    assert len(calls) == 2
    assert "header.payload.signature" not in json.dumps(report)
    assert report["promotion_allowed"] is False


def test_invalid_metadata_blocks_readiness(tmp_path):
    row = _gold_row("bad")
    row["metadata"]["patient_id"] = "forbidden"
    gold = _write_gold(tmp_path, [row])
    report = build_managed_vector_shadow_sync(
        root_dir=tmp_path,
        gold_path=gold,
        output_path=tmp_path / "report.json",
    )
    assert report["status"] == "needs_attention"
    assert report["validation_passed"] is False
    assert report["validation_issue_count"] == 1
    assert report["network_request_performed"] is False


def test_approved_portal_help_source_namespace_is_preserved(tmp_path):
    row = _gold_row("portal")
    row["namespace"] = "nlcare_portal_help"
    row["metadata"]["source_tier"] = "T4"
    gold = _write_gold(tmp_path, [row])
    report = build_managed_vector_shadow_sync(
        root_dir=tmp_path,
        gold_path=gold,
        output_path=tmp_path / "report.json",
    )
    assert report["validation_passed"] is True
    assert report["validation_issue_count"] == 0
