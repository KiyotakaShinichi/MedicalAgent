from pathlib import Path

import joblib

from backend.services import rag_vector_index as vector_index


def _payload():
    return {
        "schema_version": vector_index._current_schema_version(),
        "knowledge_fingerprint": "cache-test",
        "documents": [],
        "document_count": 0,
        "metadata": {
            "retrieval_backend": vector_index._current_backend_name(),
        },
    }


def test_index_file_is_deserialized_once_until_file_changes(tmp_path, monkeypatch):
    path = tmp_path / "index.joblib"
    joblib.dump(_payload(), path)
    vector_index._write_index_manifest(path, _payload())
    vector_index.clear_rag_runtime_cache()

    real_load = joblib.load
    calls = []

    def counting_load(target):
        calls.append(Path(target))
        return real_load(target)

    monkeypatch.setattr(vector_index.joblib, "load", counting_load)
    first = vector_index.load_rag_vector_index(path)
    second = vector_index.load_rag_vector_index(path)

    assert first is second
    assert len(calls) == 1
    assert vector_index.rag_runtime_cache_stats()["index_file_hits"] == 1

    changed = {**_payload(), "knowledge_fingerprint": "changed"}
    joblib.dump(changed, path)
    vector_index._write_index_manifest(path, changed)
    third = vector_index.load_rag_vector_index(path)
    assert third["knowledge_fingerprint"] == "changed"
    assert len(calls) == 2


def test_index_is_rejected_before_deserialization_when_manifest_is_missing(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "legacy.joblib"
    joblib.dump(_payload(), path)
    calls = []
    monkeypatch.setattr(
        vector_index.joblib,
        "load",
        lambda target: calls.append(target) or _payload(),
    )

    assert vector_index.load_rag_vector_index(path) is None
    assert calls == []


def test_index_is_rejected_when_content_hash_does_not_match(tmp_path):
    path = tmp_path / "tampered.joblib"
    joblib.dump(_payload(), path)
    vector_index._write_index_manifest(path, _payload())
    path.write_bytes(path.read_bytes() + b"tamper")

    assert vector_index.load_rag_vector_index(path) is None


def test_bm25_runtime_object_is_reused(monkeypatch):
    builds = []

    class FakeBm25:
        def __init__(self, rows):
            builds.append(rows)

        def get_scores(self, _query):
            return [1.0, 0.0]

    monkeypatch.setattr(vector_index, "_BM25_AVAILABLE", True)
    monkeypatch.setattr(vector_index, "_BM25Okapi", FakeBm25)
    index = {"bm25_tokenized_corpus": [["cbc"], ["mri"]]}

    assert vector_index._compute_bm25_scores(index, ["cbc"]) == [1.0, 0.0]
    assert vector_index._compute_bm25_scores(index, ["mri"]) == [1.0, 0.0]
    assert len(builds) == 1


def test_tfidf_query_cache_is_bounded():
    cache = {}
    for index in range(vector_index._QUERY_CACHE_LIMIT + 5):
        vector_index._bounded_cache_put(cache, str(index), index)

    assert len(cache) == vector_index._QUERY_CACHE_LIMIT
    assert "0" not in cache
