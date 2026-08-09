from __future__ import annotations

from backend.services import rag_vector_index


def test_prewarm_loads_index_and_executes_discarded_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rag_vector_index,
        "load_or_build_rag_vector_index",
        lambda **kwargs: {
            "document_count": 2,
            "metadata": {"retrieval_backend": "test_hybrid"},
        },
    )
    monkeypatch.setattr(
        rag_vector_index,
        "search_hybrid_index",
        lambda query, **kwargs: calls.append(query) or [],
    )
    state = rag_vector_index.prewarm_rag_vector_runtime(
        [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}],
        knowledge_fingerprint="test-fingerprint",
    )
    assert state["status"] == "ready"
    assert state["backend"] == "test_hybrid"
    assert state["document_count"] == 2
    assert state["startup_warmup_ms"] >= 0
    assert calls == ["breast cancer monitoring education"]
    assert state["clinical_validation"] is False


def test_prewarm_failure_is_visible_but_does_not_raise(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("synthetic test failure")

    monkeypatch.setattr(rag_vector_index, "load_or_build_rag_vector_index", fail)
    state = rag_vector_index.prewarm_rag_vector_runtime(
        [{"id": "a", "text": "alpha"}],
        knowledge_fingerprint="test-fingerprint",
    )
    assert state["status"] == "degraded"
    assert state["error_type"] == "RuntimeError"
    assert rag_vector_index.rag_runtime_readiness()["status"] == "degraded"

