from __future__ import annotations

from typing import Any, Mapping

import pytest

from backend.services.managed_vector_store import (
    AzureAISearchAdapter,
    InMemoryVectorStore,
    ManagedVectorConfig,
    PineconeAdapter,
    VectorRecord,
    VectorSearchRequest,
    VectorStoreError,
    load_managed_vector_config,
    validate_remote_record,
)


def _metadata(*, tier: str = "T2") -> dict[str, Any]:
    return {
        "source_id": "source-a",
        "chunk_id": "chunk-a",
        "parent_id": "source-a",
        "source_tier": tier,
        "allowed_use": ["education"],
        "patient_facing": True,
        "staleness_status": "current",
        "kb_fingerprint": "kb-fixture",
        "doc_type": "knowledge_chunk",
        "data_scope": "curated_non_patient_kb",
        "clinical_validation": False,
    }


def _config(provider: str, *, network: bool = True) -> ManagedVectorConfig:
    return ManagedVectorConfig(
        provider=provider,
        enabled=True,
        shadow_only=True,
        allow_network=network,
        namespace="nlcare_kb_demo_t1_t3",
        endpoint="https://vector.example",
        index_name="nlcare-shadow",
        api_version="2026-04-01" if provider == "azure_ai_search" else "2025-10",
        credential="fixture-secret",
        embedding_dimension=4,
    )


def test_config_defaults_to_local_and_network_off():
    config = load_managed_vector_config({})
    assert config.provider == "local_faiss"
    assert config.allow_network is False
    assert config.configured is True


def test_azure_config_requires_explicit_shadow_and_credentials():
    config = load_managed_vector_config(
        {
            "NLCARE_VECTOR_BACKEND": "azure_ai_search",
            "NLCARE_MANAGED_VECTOR_SHADOW_ENABLED": "true",
            "AZURE_SEARCH_ENDPOINT": "https://example.search.windows.net",
            "AZURE_SEARCH_INDEX_NAME": "nlcare",
            "AZURE_SEARCH_API_KEY": "secret",
        }
    )
    assert config.provider == "azure_ai_search"
    assert config.configured is True
    assert config.allow_network is False


def test_remote_record_rejects_patient_metadata():
    record = VectorRecord(
        record_id="unsafe",
        vector=(1.0, 0.0, 0.0, 0.0),
        text="fixture",
        metadata={**_metadata(), "patient_id": "P001"},
    )
    with pytest.raises(VectorStoreError, match="Forbidden metadata"):
        validate_remote_record(record, namespace="nlcare_kb_demo_t1_t3")


def test_remote_record_rejects_unapproved_namespace():
    record = VectorRecord(
        record_id="unsafe",
        vector=(1.0, 0.0, 0.0, 0.0),
        text="fixture",
        metadata=_metadata(),
    )
    with pytest.raises(VectorStoreError, match="namespace"):
        validate_remote_record(record, namespace="patient-P001")


def test_remote_record_rejects_unknown_metadata():
    record = VectorRecord(
        record_id="unsafe",
        vector=(1.0, 0.0, 0.0, 0.0),
        text="fixture",
        metadata={**_metadata(), "arbitrary_field": "not allowed"},
    )
    with pytest.raises(VectorStoreError, match="Unknown remote metadata"):
        validate_remote_record(record, namespace="nlcare_kb_demo_t1_t3")


def test_in_memory_adapter_applies_tier_filter():
    adapter = InMemoryVectorStore(dimension=4)
    adapter.upsert(
        [
            VectorRecord("allowed", (1.0, 0.0, 0.0, 0.0), "allowed", _metadata(tier="T2")),
            VectorRecord("blocked", (1.0, 0.0, 0.0, 0.0), "blocked", _metadata(tier="T4")),
        ]
    )
    results = adapter.search(
        VectorSearchRequest((1.0, 0.0, 0.0, 0.0), "education", allowed_tiers=("T2",))
    )
    assert [row.record_id for row in results] == ["allowed"]


def test_azure_payload_uses_hybrid_query_and_prefilter():
    adapter = AzureAISearchAdapter(_config("azure_ai_search"))
    payload = adapter.build_search_payload(
        VectorSearchRequest((1.0, 0.0, 0.0, 0.0), "cbc education")
    )
    assert payload["search"] == "cbc education"
    assert payload["vectorFilterMode"] == "preFilter"
    assert payload["vectorQueries"][0]["fields"] == "content_vector"
    assert "clinical_validation eq false" in payload["filter"]
    assert "data_scope eq 'curated_non_patient_kb'" in payload["filter"]
    assert "allowed_use/any" in payload["filter"]


def test_pinecone_payload_uses_canonical_metadata_filters():
    adapter = PineconeAdapter(_config("pinecone"))
    payload = adapter.build_search_payload(
        VectorSearchRequest((1.0, 0.0, 0.0, 0.0), "cbc education")
    )
    clauses = payload["filter"]["$and"]
    assert {"clinical_validation": {"$eq": False}} in clauses
    assert {"patient_facing": {"$eq": True}} in clauses
    assert {"data_scope": {"$eq": "curated_non_patient_kb"}} in clauses
    assert {"allowed_use": {"$in": ["education", "patient_safety", "monitoring_context"]}} in clauses
    assert payload["includeValues"] is False


def test_network_is_blocked_by_default():
    adapter = PineconeAdapter(_config("pinecone", network=False))
    with pytest.raises(VectorStoreError, match="network execution is disabled"):
        adapter.search(VectorSearchRequest((1.0, 0.0, 0.0, 0.0), "fixture"))


def test_azure_adapter_can_execute_against_injected_transport_without_real_network():
    calls: list[dict[str, Any]] = []

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        calls.append({"method": method, "url": url, "headers": headers, "payload": payload})
        return {"value": [{"id": "chunk-a", "content": "fixture", "@search.score": 1.2}]}

    adapter = AzureAISearchAdapter(_config("azure_ai_search"), transport=transport)
    results = adapter.search(VectorSearchRequest((1.0, 0.0, 0.0, 0.0), "fixture"))
    assert results[0].record_id == "chunk-a"
    assert calls[0]["url"].endswith("api-version=2026-04-01")
    assert calls[0]["method"] == "POST"
