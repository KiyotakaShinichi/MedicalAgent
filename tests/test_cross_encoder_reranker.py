from backend.services.cross_encoder_reranker import rerank_with_cross_encoder


def test_cross_encoder_fallback_preserves_metadata(monkeypatch):
    monkeypatch.setenv("RAG_ENABLE_CROSS_ENCODER", "false")
    rows = [
        {
            "id": "a",
            "text": "CBC monitoring text",
            "source_tier": "T2",
            "allowed_use": ["education"],
            "staleness": "current",
            "parent_id": "p1",
        }
    ]
    reranked, telemetry = rerank_with_cross_encoder("cbc", rows)
    assert telemetry["enabled"] is False
    assert reranked[0]["source_tier"] == "T2"
    assert reranked[0]["allowed_use"] == ["education"]
    assert reranked[0]["parent_id"] == "p1"
    assert reranked[0]["cross_encoder_score"] is None


def test_cross_encoder_unavailable_is_safe_fallback(monkeypatch):
    monkeypatch.setenv("RAG_ENABLE_CROSS_ENCODER", "true")
    rows = [{"id": "a", "title": "A", "text": "safe text", "source_tier": "T1"}]
    reranked, telemetry = rerank_with_cross_encoder("query", rows)
    assert len(reranked) == 1
    assert "reranker_latency_ms" in telemetry
    assert reranked[0]["source_tier"] == "T1"
