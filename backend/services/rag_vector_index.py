import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_RAG_INDEX_PATH = "Data/rag_index/local_hybrid_rag_index.joblib"
RAG_INDEX_SCHEMA_VERSION = "local_hybrid_rag_index_v1"


def build_rag_vector_index(corpus, index_path=DEFAULT_RAG_INDEX_PATH, knowledge_fingerprint=None):
    documents = [_normalize_document(item) for item in corpus if item.get("id") and item.get("text")]
    texts = [_document_text(document) for document in documents]
    if not texts:
        raise ValueError("Cannot build RAG index without documents.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=12000,
        norm="l2",
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9/-]+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    fingerprint = knowledge_fingerprint or corpus_fingerprint(documents)
    payload = {
        "schema_version": RAG_INDEX_SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_fingerprint": fingerprint,
        "document_count": len(documents),
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": documents,
        "metadata": {
            "retrieval_backend": "local_tfidf_hybrid_index",
            "semantic_component": "tfidf_word_and_bigram_cosine",
            "lexical_component": "query_token_overlap",
            "swap_path": "Replace this service with Qdrant/pgvector/Pinecone while keeping search_hybrid_index contract.",
        },
    }
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
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
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    documents = index["documents"]
    query_vector = vectorizer.transform([query])
    vector_scores = (matrix @ query_vector.T).toarray().ravel()
    query_tokens = set(_tokenize(query))

    rows = []
    for position, document in enumerate(documents):
        lexical = _overlap_score(query_tokens, set(document["tokens"]))
        metadata = _metadata_score(query_tokens, document)
        intent_boost = _intent_score(intent, document)
        score = (0.58 * float(vector_scores[position])) + (0.27 * lexical) + (0.15 * metadata) + intent_boost
        if score <= 0:
            continue
        rows.append({
            **document["payload"],
            "retrieval_backend": "local_tfidf_hybrid_index",
            "retrieval_score": round(score, 4),
            "vector_score": round(float(vector_scores[position]), 4),
            "lexical_score": round(lexical, 4),
            "metadata_score": round(metadata, 4),
            "matched_terms": sorted(query_tokens & set(document["tokens"]))[:10],
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
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        return None
    return payload


def rag_index_status(corpus=None, index_path=DEFAULT_RAG_INDEX_PATH, knowledge_fingerprint=None):
    path = Path(index_path)
    payload = load_rag_vector_index(index_path)
    if payload is None:
        return {
            "status": "missing",
            "path": str(path),
            "schema_version": RAG_INDEX_SCHEMA_VERSION,
        }
    expected_fingerprint = knowledge_fingerprint or (corpus_fingerprint(corpus) if corpus is not None else None)
    current = expected_fingerprint is None or _is_current_index(payload, expected_fingerprint)
    summary = index_summary(payload, path)
    return {
        **summary,
        "status": "current" if current else "stale",
        "expected_knowledge_fingerprint": expected_fingerprint,
    }


def index_summary(payload, path=None):
    return {
        "schema_version": payload.get("schema_version"),
        "path": str(path) if path else None,
        "built_at": payload.get("built_at"),
        "knowledge_fingerprint": payload.get("knowledge_fingerprint"),
        "document_count": payload.get("document_count") or len(payload.get("documents") or []),
        "retrieval_backend": (payload.get("metadata") or {}).get("retrieval_backend"),
        "semantic_component": (payload.get("metadata") or {}).get("semantic_component"),
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


def _is_current_index(payload, knowledge_fingerprint):
    return (
        payload.get("schema_version") == RAG_INDEX_SCHEMA_VERSION
        and payload.get("knowledge_fingerprint") == knowledge_fingerprint
        and bool(payload.get("documents"))
    )


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
        "patient_timeline_monitoring": {"score", "monitoring", "cbc", "response", "mri"},
        "safety_boundary": {"urgent", "fever", "infection", "neutropenia", "safety"},
        "treatment_decision_boundary": {"treatment", "doctor", "chemotherapy"},
        "emotional_support": {"symptoms", "doctor", "effects"},
    }.get(intent, set())
    if not desired:
        return 0.0
    tokens = set(document["tag_tokens"]) | set(document["metadata_tokens"]) | set(document["tokens"])
    return 0.08 if tokens & desired else 0.0


def _tokenize(text):
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "how",
        "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "this", "to", "what",
        "when", "where", "why", "with", "you", "your",
    }
    import re

    return [
        token
        for token in re.findall(r"[a-z0-9/-]+", (text or "").lower())
        if token not in stopwords and len(token) > 1
    ]


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
