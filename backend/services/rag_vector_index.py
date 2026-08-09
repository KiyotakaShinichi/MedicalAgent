"""
RAG vector index with true hybrid retrieval.

Backend selection (automatic at build time, based on installed dependencies):

  Dense hybrid  - sentence-transformers/all-MiniLM-L6-v2 + FAISS IndexFlatIP + BM25Okapi
                  schema_version  = "local_dense_faiss_hybrid_index_v2"
                  retrieval_backend = "local_dense_faiss_hybrid_index"
                  Fusion: Reciprocal Rank Fusion (RRF k=60) of dense + sparse lists

  Sparse fallback - BM25Okapi + TF-IDF bigram cosine (no sentence-transformers / faiss required)
                  schema_version  = "local_sparse_tfidf_bm25_index_v2"
                  retrieval_backend = "local_sparse_tfidf_bm25_index"

Retrieval trace fields returned per result:
  sparse_score, dense_score, rrf_score, fusion_score, metadata_score, backend

Swap note: replace this service with Qdrant / pgvector / Pinecone while keeping the
           search_hybrid_index() contract unchanged.
"""

import hashlib
import importlib.util
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

# sklearn.feature_extraction.text is ~2.5s to import (drags in
# scipy/sklearn/preprocessing). Deferred to inside the build path so
# warm chat calls that hit only the cached index don't pay the cost.

# -- Optional dense-retrieval dependencies -------------------------------------
_FORCE_SPARSE = os.getenv("RAG_FORCE_SPARSE", "").strip().lower() in {"1", "true", "yes"}

_DENSE_AVAILABLE = (
    not _FORCE_SPARSE
    and importlib.util.find_spec("sentence_transformers") is not None
    and importlib.util.find_spec("faiss") is not None
)

# -- Optional BM25 dependency --------------------------------------------------
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# -- Constants -----------------------------------------------------------------
_DENSE_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ENCODER_CACHE: dict = {}
_INDEX_FILE_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}
_INDEX_CACHE_LOCK = threading.RLock()
_CACHE_METRICS = {
    "index_file_hits": 0,
    "index_file_loads": 0,
    "bm25_builds": 0,
    "faiss_builds": 0,
    "dense_query_hits": 0,
    "tfidf_query_hits": 0,
}
_QUERY_CACHE_LIMIT = 256
_PREWARM_STATE: dict[str, object] = {
    "status": "not_started",
    "backend": None,
    "document_count": 0,
    "startup_warmup_ms": None,
    "error_type": None,
    "clinical_validation": False,
    "healthcare_production_ready": False,
}

DENSE_HYBRID_SCHEMA_VERSION = "local_dense_faiss_hybrid_index_v2"
SPARSE_BM25_SCHEMA_VERSION = "local_sparse_tfidf_bm25_index_v2"

DEFAULT_RAG_INDEX_PATH = "Data/rag_index/local_hybrid_rag_index.joblib"

# Backward-compat alias resolved at import time
RAG_INDEX_SCHEMA_VERSION = DENSE_HYBRID_SCHEMA_VERSION if _DENSE_AVAILABLE else SPARSE_BM25_SCHEMA_VERSION


# -- Backend helpers -----------------------------------------------------------

def _current_schema_version() -> str:
    return DENSE_HYBRID_SCHEMA_VERSION if _DENSE_AVAILABLE else SPARSE_BM25_SCHEMA_VERSION


def _current_backend_name() -> str:
    return "local_dense_faiss_hybrid_index" if _DENSE_AVAILABLE else "local_sparse_tfidf_bm25_index"


def _get_encoder():
    if _DENSE_AVAILABLE and "model" not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer

        _ENCODER_CACHE["model"] = SentenceTransformer(_DENSE_ENCODER_MODEL)
    return _ENCODER_CACHE.get("model")


def _get_faiss():
    if not _DENSE_AVAILABLE:
        return None
    if "faiss" not in _ENCODER_CACHE:
        import faiss

        _ENCODER_CACHE["faiss"] = faiss
    return _ENCODER_CACHE.get("faiss")


# -- Public API ----------------------------------------------------------------

def build_rag_vector_index(corpus, index_path=DEFAULT_RAG_INDEX_PATH, knowledge_fingerprint=None):
    documents = [_normalize_document(item) for item in corpus if item.get("id") and item.get("text")]
    texts = [_document_text(document) for document in documents]
    if not texts:
        raise ValueError("Cannot build RAG index without documents.")

    fingerprint = knowledge_fingerprint or corpus_fingerprint(documents)

    # TF-IDF - always built (used as dense-unavailable fallback and backward-compat).
    # Deferred import keeps the ~2.5s sklearn.feature_extraction.text load
    # off the hot chat path; the index build is the only call site.
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=12000,
        norm="l2",
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9/-]+\b",
    )
    matrix = vectorizer.fit_transform(texts)

    # BM25 tokenized corpus (stored as plain list; rebuilt cheaply at query time)
    tokenized_corpus = [doc["tokens"] for doc in documents]

    # Dense embeddings via sentence-transformers (L2-normalised for cosine-as-IP)
    doc_embeddings = None
    if _DENSE_AVAILABLE:
        encoder = _get_encoder()
        raw_emb = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        doc_embeddings = (raw_emb / norms).astype("float32")

    schema_version = _current_schema_version()
    backend_name = _current_backend_name()

    payload = {
        "schema_version": schema_version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_fingerprint": fingerprint,
        "document_count": len(documents),
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": documents,
        "bm25_tokenized_corpus": tokenized_corpus if _BM25_AVAILABLE else None,
        "doc_embeddings": doc_embeddings,
        "encoder_model": _DENSE_ENCODER_MODEL if _DENSE_AVAILABLE else None,
        "metadata": {
            "retrieval_backend": backend_name,
            "dense_component": (
                f"sentence-transformer cosine via FAISS IndexFlatIP ({_DENSE_ENCODER_MODEL})"
                if _DENSE_AVAILABLE else None
            ),
            "sparse_component": "BM25Okapi" if _BM25_AVAILABLE else "TF-IDF bigram cosine",
            "fusion": (
                "Reciprocal Rank Fusion (RRF k=60)"
                if _DENSE_AVAILABLE else "BM25 + metadata overlap"
            ),
            "swap_path": (
                "Replace with Qdrant/pgvector/Pinecone while keeping search_hybrid_index contract."
            ),
        },
    }
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    _seed_index_file_cache(path, payload)
    return index_summary(payload, path)


def search_hybrid_index(
    query,
    corpus,
    intent=None,
    index_path=DEFAULT_RAG_INDEX_PATH,
    knowledge_fingerprint=None,
    candidate_limit=12,
):
    if not query:
        return []
    index = load_or_build_rag_vector_index(
        corpus=corpus,
        index_path=index_path,
        knowledge_fingerprint=knowledge_fingerprint,
    )
    documents = index["documents"]
    query_tokens = _tokenize(query)
    query_token_set = set(query_tokens)

    # -- Sparse scores (BM25 preferred, TF-IDF cosine as fallback) ------------
    bm25_raw = _compute_bm25_scores(index, query_tokens)
    if bm25_raw is not None:
        bm25_max = max(bm25_raw) if bm25_raw else 1.0
        bm25_max = bm25_max if bm25_max > 0 else 1.0
        sparse_scores = [s / bm25_max for s in bm25_raw]
    else:
        sparse_scores = _compute_tfidf_scores(index, query)

    # -- Dense scores (sentence-transformer + FAISS, if available) ------------
    dense_scores = _compute_dense_scores(index, query)

    # -- Metadata scores -------------------------------------------------------
    meta_scores = [_metadata_score(query_token_set, doc) for doc in documents]

    # -- Fusion ----------------------------------------------------------------
    if dense_scores is not None:
        rrf_scores = _rrf_fuse(sparse_scores, dense_scores, k=60)
        backend = "local_dense_faiss_hybrid_index"
    else:
        rrf_scores = None
        backend = "local_sparse_tfidf_bm25_index"

    # TF-IDF cosine kept for backward-compat vector_score field
    tfidf_scores = _compute_tfidf_scores(index, query)

    rows = []
    for i, document in enumerate(documents):
        sparse_s = round(float(sparse_scores[i]), 4)
        dense_s = round(float(dense_scores[i]), 4) if dense_scores is not None else None
        rrf_s = round(float(rrf_scores[i]), 4) if rrf_scores is not None else None
        meta_s = round(float(meta_scores[i]), 4)
        intent_s = _intent_score(intent, document)

        if rrf_s is not None:
            # Dense hybrid: RRF is the primary signal
            score = 0.65 * rrf_s + 0.20 * meta_s + intent_s
        else:
            # Sparse fallback: BM25/TF-IDF + overlap + metadata
            score = (
                0.58 * sparse_s
                + 0.27 * _overlap_score(query_token_set, set(document["tokens"]))
                + 0.15 * meta_s
                + intent_s
            )

        if score <= 0:
            continue

        rows.append({
            **document["payload"],
            "retrieval_backend": backend,
            "retrieval_score": round(score, 4),
            # -- Honest retrieval trace ----------------------------------------
            "sparse_score": sparse_s,
            "dense_score": dense_s,
            "rrf_score": rrf_s,
            "fusion_score": rrf_s,  # alias kept for API clarity
            "metadata_score": meta_s,
            "backend": backend,
            # -- Backward-compat aliases ---------------------------------------
            "vector_score": (
                dense_s if dense_s is not None else round(float(tfidf_scores[i]), 4)
            ),
            "lexical_score": sparse_s,
            "matched_terms": sorted(query_token_set & set(document["tokens"]))[:10],
        })
    return sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)[:candidate_limit]


def load_or_build_rag_vector_index(corpus, index_path=DEFAULT_RAG_INDEX_PATH, knowledge_fingerprint=None):
    fingerprint = knowledge_fingerprint or corpus_fingerprint(corpus)
    existing = load_rag_vector_index(index_path=index_path)
    if existing and _is_current_index(existing, fingerprint):
        return existing
    build_rag_vector_index(corpus=corpus, index_path=index_path, knowledge_fingerprint=fingerprint)
    return load_rag_vector_index(index_path=index_path)


def load_rag_vector_index(index_path=DEFAULT_RAG_INDEX_PATH):
    path = Path(index_path)
    if not path.exists():
        return None
    cache_key = str(path.resolve())
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_FILE_CACHE.get(cache_key)
        if cached and cached[0] == signature:
            _CACHE_METRICS["index_file_hits"] += 1
            return cached[1]
    try:
        payload = joblib.load(path)
    except Exception:
        # Treat corrupted/partially-written local index files as cache misses.
        # The caller can rebuild from the source KB corpus; crashing here makes
        # a disposable retrieval cache look like an application failure.
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        with _INDEX_CACHE_LOCK:
            _INDEX_FILE_CACHE.pop(cache_key, None)
        return None
    if not isinstance(payload, dict):
        return None
    with _INDEX_CACHE_LOCK:
        _INDEX_FILE_CACHE[cache_key] = (signature, payload)
        _CACHE_METRICS["index_file_loads"] += 1
    return payload


def clear_rag_runtime_cache() -> None:
    """Clear disposable process-local retrieval objects.

    The persisted index remains untouched. Tests, ingestion, and maintenance
    jobs use this when they need to prove cold-start behavior.
    """
    with _INDEX_CACHE_LOCK:
        _INDEX_FILE_CACHE.clear()
        for key in _CACHE_METRICS:
            _CACHE_METRICS[key] = 0


def rag_runtime_cache_stats() -> dict[str, int]:
    with _INDEX_CACHE_LOCK:
        return {
            **_CACHE_METRICS,
            "cached_index_count": len(_INDEX_FILE_CACHE),
        }


def prewarm_rag_vector_runtime(
    corpus,
    *,
    index_path=DEFAULT_RAG_INDEX_PATH,
    knowledge_fingerprint=None,
) -> dict[str, object]:
    """Load the persisted index, encoder, and query-time runtime objects.

    The fixed warmup query is synthetic and its retrieval result is discarded.
    Failures are reported in readiness state instead of crashing application
    startup; the normal retrieval path can still attempt its own fallback.
    """
    started = perf_counter()
    with _INDEX_CACHE_LOCK:
        _PREWARM_STATE.update({
            "status": "warming",
            "backend": None,
            "document_count": 0,
            "startup_warmup_ms": None,
            "error_type": None,
        })
    try:
        index = load_or_build_rag_vector_index(
            corpus=corpus,
            index_path=index_path,
            knowledge_fingerprint=knowledge_fingerprint,
        )
        search_hybrid_index(
            "breast cancer monitoring education",
            corpus=corpus,
            intent="education",
            index_path=index_path,
            knowledge_fingerprint=knowledge_fingerprint,
            candidate_limit=1,
        )
        state = {
            "status": "ready",
            "backend": (index.get("metadata") or {}).get("retrieval_backend"),
            "document_count": int(index.get("document_count") or 0),
            "startup_warmup_ms": round((perf_counter() - started) * 1000.0, 2),
            "error_type": None,
            "clinical_validation": False,
            "healthcare_production_ready": False,
        }
    except Exception as exc:  # noqa: BLE001 - readiness must stay observable
        state = {
            "status": "degraded",
            "backend": None,
            "document_count": 0,
            "startup_warmup_ms": round((perf_counter() - started) * 1000.0, 2),
            "error_type": exc.__class__.__name__,
            "clinical_validation": False,
            "healthcare_production_ready": False,
        }
    with _INDEX_CACHE_LOCK:
        _PREWARM_STATE.update(state)
        return dict(_PREWARM_STATE)


def rag_runtime_readiness() -> dict[str, object]:
    with _INDEX_CACHE_LOCK:
        return dict(_PREWARM_STATE)


def _seed_index_file_cache(path: Path, payload: dict) -> None:
    try:
        stat = path.stat()
    except OSError:
        return
    with _INDEX_CACHE_LOCK:
        _INDEX_FILE_CACHE[str(path.resolve())] = (
            (stat.st_mtime_ns, stat.st_size),
            payload,
        )


def rag_index_status(corpus=None, index_path=DEFAULT_RAG_INDEX_PATH, knowledge_fingerprint=None):
    path = Path(index_path)
    payload = load_rag_vector_index(index_path)
    if payload is None:
        return {
            "status": "missing",
            "path": str(path),
            "schema_version": _current_schema_version(),
            "retrieval_backend": _current_backend_name(),
            "dense_available": _DENSE_AVAILABLE,
            "bm25_available": _BM25_AVAILABLE,
        }
    expected_fingerprint = knowledge_fingerprint or (
        corpus_fingerprint(corpus) if corpus is not None else None
    )
    current = expected_fingerprint is None or _is_current_index(payload, expected_fingerprint)
    summary = index_summary(payload, path)
    return {
        **summary,
        "status": "current" if current else "stale",
        "expected_knowledge_fingerprint": expected_fingerprint,
        "dense_available": _DENSE_AVAILABLE,
        "bm25_available": _BM25_AVAILABLE,
    }


def index_summary(payload, path=None):
    meta = payload.get("metadata") or {}
    return {
        "schema_version": payload.get("schema_version"),
        "path": str(path) if path else None,
        "built_at": payload.get("built_at"),
        "knowledge_fingerprint": payload.get("knowledge_fingerprint"),
        "document_count": payload.get("document_count") or len(payload.get("documents") or []),
        "retrieval_backend": meta.get("retrieval_backend"),
        "dense_component": meta.get("dense_component"),
        "sparse_component": meta.get("sparse_component"),
        "fusion": meta.get("fusion"),
        # Backward-compat
        "semantic_component": meta.get("dense_component") or meta.get("sparse_component"),
    }


def corpus_fingerprint(corpus):
    rows = []
    for item in corpus or []:
        rows.append({
            "id": item.get("id"),
            "parent_id": item.get("parent_id"),
            "title": item.get("title"),
            "source_name": item.get("source_name"),
            "source_url": item.get("source_url"),
            "tags": sorted(item.get("tags") or []),
            "topic": item.get("topic"),
            "section": item.get("section"),
            "text_hash": _hash(item.get("text") or ""),
        })
    encoded = json.dumps(sorted(rows, key=lambda row: row.get("id") or ""), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


# -- Score computation helpers -------------------------------------------------

def _compute_bm25_scores(index, query_tokens):
    """Returns list of raw BM25 scores (not normalised), or None if unavailable."""
    tokenized_corpus = index.get("bm25_tokenized_corpus")
    if not _BM25_AVAILABLE or not tokenized_corpus:
        return None
    runtime = index.setdefault("_runtime_cache", {})
    bm25 = runtime.get("bm25")
    if bm25 is None:
        bm25 = _BM25Okapi(tokenized_corpus)
        runtime["bm25"] = bm25
        _CACHE_METRICS["bm25_builds"] += 1
    return list(bm25.get_scores(query_tokens))


def _compute_tfidf_scores(index, query):
    """Returns list of TF-IDF cosine similarity scores in [0, 1]."""
    vectorizer = index.get("vectorizer")
    matrix = index.get("matrix")
    if vectorizer is None or matrix is None:
        n = len(index.get("documents") or [])
        return [0.0] * n
    runtime = index.setdefault("_runtime_cache", {})
    query_cache = runtime.setdefault("tfidf_queries", {})
    cache_key = str(query).strip().lower()
    if cache_key in query_cache:
        _CACHE_METRICS["tfidf_query_hits"] += 1
        return query_cache[cache_key]
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    values = [float(s) for s in scores]
    _bounded_cache_put(query_cache, cache_key, values)
    return values


def _compute_dense_scores(index, query):
    """Returns list of cosine similarities in [-1, 1], or None if dense unavailable."""
    doc_embeddings = index.get("doc_embeddings")
    if not _DENSE_AVAILABLE or doc_embeddings is None:
        return None

    encoder = _get_encoder()
    if encoder is None:
        return None

    runtime = index.setdefault("_runtime_cache", {})
    query_cache = runtime.setdefault("dense_queries", {})
    cache_key = str(query).strip().lower()
    q_emb = query_cache.get(cache_key)
    if q_emb is None:
        q_raw = encoder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        norm = np.linalg.norm(q_raw)
        q_emb = (q_raw / norm).astype("float32") if norm > 0 else q_raw.astype("float32")
        _bounded_cache_put(query_cache, cache_key, q_emb)
    else:
        _CACHE_METRICS["dense_query_hits"] += 1

    faiss_module = _get_faiss()
    if faiss_module is None:
        return None
    faiss_idx = runtime.get("faiss_index")
    if faiss_idx is None:
        d = doc_embeddings.shape[1]
        faiss_idx = faiss_module.IndexFlatIP(d)
        faiss_idx.add(doc_embeddings)
        runtime["faiss_index"] = faiss_idx
        _CACHE_METRICS["faiss_builds"] += 1
    n = doc_embeddings.shape[0]
    scores, indices = faiss_idx.search(q_emb, n)

    # Map back to original document order
    ordered = np.zeros(n, dtype="float32")
    for rank_idx, (doc_idx, score) in enumerate(zip(indices[0], scores[0])):
        ordered[int(doc_idx)] = float(score)
    return ordered.tolist()


def _bounded_cache_put(cache: dict, key: str, value) -> None:
    cache[key] = value
    if len(cache) > _QUERY_CACHE_LIMIT:
        cache.pop(next(iter(cache)))


# -- RRF -----------------------------------------------------------------------

def _rrf_fuse(sparse_scores, dense_scores, k=60):
    """Reciprocal Rank Fusion of two normalised score lists; output is normalised to [0,1]."""
    n = len(sparse_scores)
    sparse_ranks = _scores_to_ranks(sparse_scores)
    dense_ranks = _scores_to_ranks(dense_scores)
    rrf = [1.0 / (k + sparse_ranks[i]) + 1.0 / (k + dense_ranks[i]) for i in range(n)]
    rrf_max = max(rrf) if rrf else 1.0
    rrf_max = rrf_max if rrf_max > 0 else 1.0
    return [s / rrf_max for s in rrf]


def _scores_to_ranks(scores):
    """Convert score list to 1-based ranks (higher score -> rank 1)."""
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (idx, _) in enumerate(indexed):
        ranks[idx] = rank + 1
    return ranks


# -- Document normalisation ----------------------------------------------------

def _normalize_document(item):
    payload = dict(item)
    payload["tags"] = list(item.get("tags") or [])
    payload["modality"] = list(item.get("modality") or [])
    text = _document_text_from_payload(payload)
    tokens = _tokenize(text)
    return {
        "id": str(payload.get("id")),
        "payload": payload,
        "tokens": tokens,
        "tag_tokens": _tokenize(" ".join(payload.get("tags") or [])),
        "metadata_tokens": _tokenize(" ".join([
            payload.get("topic") or "",
            " ".join(payload.get("modality") or []),
            payload.get("care_stage") or "",
            payload.get("section") or "",
            payload.get("source_name") or "",
        ])),
    }


def _document_text(document):
    return _document_text_from_payload(document["payload"])


def _document_text_from_payload(payload):
    return " ".join([
        payload.get("title") or "",
        payload.get("source_name") or "",
        " ".join(payload.get("tags") or []),
        payload.get("topic") or "",
        " ".join(payload.get("modality") or []),
        payload.get("care_stage") or "",
        payload.get("section") or "",
        payload.get("text") or "",
    ])


# -- Index freshness -----------------------------------------------------------

def _is_current_index(payload, knowledge_fingerprint):
    """Stale if schema doesn't match current backend capability or fingerprint changed."""
    return (
        payload.get("schema_version") == _current_schema_version()
        and payload.get("knowledge_fingerprint") == knowledge_fingerprint
        and bool(payload.get("documents"))
    )


# -- Per-document scoring helpers ----------------------------------------------

def _overlap_score(query_tokens, document_tokens):
    if not query_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def _metadata_score(query_tokens, document):
    metadata_tokens = set(document["tag_tokens"]) | set(document["metadata_tokens"])
    if not metadata_tokens:
        return 0.0
    return len(query_tokens & metadata_tokens) / len(query_tokens or metadata_tokens)


def _intent_score(intent, document):
    if not intent:
        return 0.0
    desired = {
        "portal_help": {"upload", "portal", "labs", "mri", "symptoms"},
        "education": {"pcr", "cbc", "wbc", "chemotherapy", "effects", "mri", "response"},
        "urgent_escalation": {"urgent", "fever", "infection", "neutropenia", "anc", "safety"},
        "genetic_counselor_review": {"genetic", "counseling", "vus", "brca", "germline", "variant"},
        "tumor_marker_boundary": {"tumor", "marker", "ca", "15-3", "cea", "recurrence", "limitation"},
        "pharmacist_or_clinician_review": {"supplement", "interaction", "pharmacist", "st", "johns", "wort", "safety"},
        "treatment_refusal": {"treatment", "chemo", "chemotherapy", "doctor", "clinician", "safety"},
        "prognosis_refusal": {"prognosis", "survival", "boundary", "clinician", "safety"},
        "diagnosis_refusal": {"diagnosis", "progression", "imaging", "clinician", "safety"},
        "privacy_refusal": {"privacy", "patient", "record", "policy", "portal", "safety"},
        "patient_timeline_monitoring": {"score", "monitoring", "cbc", "response", "mri"},
        "safety_boundary": {"urgent", "fever", "infection", "neutropenia", "safety"},
        "treatment_decision_boundary": {"treatment", "doctor", "chemotherapy"},
        "emotional_support": {"symptoms", "doctor", "effects"},
    }.get(intent, set())
    if not desired:
        return 0.0
    tokens = set(document["tag_tokens"]) | set(document["metadata_tokens"]) | set(document["tokens"])
    if not (tokens & desired):
        return 0.0
    if intent in {
        "privacy_refusal",
        "prognosis_refusal",
        "diagnosis_refusal",
        "treatment_refusal",
        "tumor_marker_boundary",
        "genetic_counselor_review",
        "pharmacist_or_clinician_review",
        "urgent_escalation",
    }:
        return 0.22
    return 0.08


# -- Tokenisation --------------------------------------------------------------

def _tokenize(text):
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "how",
        "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "this", "to", "what",
        "when", "where", "why", "with", "you", "your",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9/-]+", (text or "").lower())
        if token not in stopwords and len(token) > 1
    ]


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
