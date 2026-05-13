"""
RAG ablation study for the patient support agent.

The current production retrieval path is dense + sparse hybrid retrieval when
local dependencies are available:
  - dense: sentence-transformers/all-MiniLM-L6-v2 embeddings
  - vector search: FAISS IndexFlatIP over L2-normalized embeddings
  - sparse: BM25Okapi
  - fusion: Reciprocal Rank Fusion (RRF)

If dense dependencies are missing, the same search contract falls back to
BM25 + TF-IDF. This report exposes the active backend and compares retrieval
stages honestly on fixed educational regression cases.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

ABLATION_OUTPUT_PATH = "Data/agent_eval/rag_ablation.json"

_EDUCATION_CASE_IDS = [
    "education-pcr-definition",
    "education-her2",
    "education-neoadjuvant",
    "education-low-wbc",
    "education-nadir",
    "education-neutropenia",
    "education-mri-response",
    "education-dose-delay",
    "education-triple-negative",
    "education-gcsf-prophylaxis",
]


def run_rag_ablation(output_path: str = ABLATION_OUTPUT_PATH) -> dict:
    from backend.services.agent_regression_eval import DEFAULT_AGENT_EVAL_CASES
    from backend.services.agent_rag import (
        expand_parent_child_windows,
        get_rag_corpus,
        hybrid_retrieval,
        knowledge_base_fingerprint,
        rerank_context,
        rewrite_and_decompose,
    )
    from backend.services.rag_vector_index import rag_index_status, search_hybrid_index

    education_cases = [c for c in DEFAULT_AGENT_EVAL_CASES if c["id"] in _EDUCATION_CASE_IDS]
    corpus = get_rag_corpus()
    def bm25_only(rewritten: dict, intent: str) -> list[dict]:
        return _bm25_only_retrieval(rewritten["expanded_query"], corpus)

    def sparse_tfidf_bm25(rewritten: dict, intent: str) -> list[dict]:
        return _sparse_tfidf_bm25_retrieval(rewritten["expanded_query"], corpus)

    def dense_hybrid(rewritten: dict, intent: str) -> list[dict]:
        return search_hybrid_index(
            query=rewritten["expanded_query"],
            corpus=corpus,
            intent=intent,
            knowledge_fingerprint=knowledge_base_fingerprint(),
        )[:5]

    def dense_hybrid_agent(rewritten: dict, intent: str) -> list[dict]:
        return hybrid_retrieval(rewritten, intent)

    def full_pipeline(rewritten: dict, intent: str) -> list[dict]:
        retrieved = hybrid_retrieval(rewritten, intent)
        expanded = expand_parent_child_windows(retrieved)
        return rerank_context(expanded, rewritten, intent, {"level": "low_risk"})

    strategies = {
        "bm25_only": bm25_only,
        "sparse_tfidf_bm25": sparse_tfidf_bm25,
        "dense_faiss_bm25_rrf": dense_hybrid,
        "dense_faiss_bm25_rrf_agent_boosted": dense_hybrid_agent,
        "dense_faiss_bm25_rrf_reranked": full_pipeline,
        # Backward-compatible aliases for older frontend code.
        "hybrid": dense_hybrid,
        "hybrid_reranked": full_pipeline,
    }

    results = {
        strategy_name: _evaluate_strategy(strategy_name, retrieve_fn, education_cases)
        for strategy_name, retrieve_fn in strategies.items()
    }
    index_status = rag_index_status(corpus=corpus, knowledge_fingerprint=knowledge_base_fingerprint())

    output = {
        "schema_version": "rag_ablation_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Compares BM25-only, sparse BM25+TF-IDF, active dense/sparse hybrid retrieval, "
            "agent-boosted hybrid retrieval, and full reranked retrieval on education cases."
        ),
        "active_index": index_status,
        "strategies": results,
        "comparison": _comparison_notes(results, index_status),
        "limitations": [
            "All cases are synthetic educational regression cases, not real patient questions.",
            "Grounding score is a heuristic token-overlap proxy, not a formal clinical faithfulness metric.",
            "Dense retrieval depends on local sentence-transformers and FAISS availability.",
            "Expected source labels are sparse and should be expanded into a larger hand-labeled RAG benchmark.",
        ],
        "claim_boundary": (
            "These ablation results support engineering comparison only. They do not validate clinical accuracy."
        ),
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(__import__("json").dumps(output, indent=2), encoding="utf-8")

    return output


def _evaluate_strategy(strategy_name: str, retrieve_fn, cases: list[dict]) -> dict:
    hits = 0
    grounding_scores = []
    latencies = []
    source_hit_count = 0
    source_case_count = 0
    backend_names: set[str] = set()

    from backend.services.agent_rag import rewrite_and_decompose

    for case in cases:
        t0 = time.perf_counter()
        try:
            rewritten = rewrite_and_decompose(case["query"], "education")
            retrieved = retrieve_fn(rewritten, "education")
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            backend_names.update(
                str(item.get("backend") or item.get("retrieval_backend") or strategy_name)
                for item in retrieved
            )

            source_ids = {item.get("id") for item in retrieved if item.get("id")}
            expected = set(case.get("expected_sources") or [])

            if expected:
                source_case_count += 1
                if source_ids & expected:
                    source_hit_count += 1
                    hits += 1
            else:
                hits += 1

            grounding = _estimate_grounding(rewritten["query"], retrieved)
            if grounding is not None:
                grounding_scores.append(grounding)

        except Exception:
            latencies.append(None)

    n = len(cases)
    return {
        "case_count": n,
        "pass_rate": round(hits / n, 3) if n else None,
        "expected_source_hit_rate": (
            round(source_hit_count / source_case_count, 3)
            if source_case_count else "n/a (no expected sources)"
        ),
        "average_grounding_score": _mean(grounding_scores),
        "average_latency_ms": _mean([x for x in latencies if x is not None]),
        "backend": ", ".join(sorted(backend_names)) if backend_names else strategy_name,
    }


def _bm25_only_retrieval(query: str, corpus: list[dict], limit: int = 5) -> list[dict]:
    tokens = [_tokenize(_document_text(item)) for item in corpus]
    query_tokens = _tokenize(query)
    try:
        from rank_bm25 import BM25Okapi

        scores = list(BM25Okapi(tokens).get_scores(query_tokens))
    except Exception:
        query_terms = set(query_tokens)
        scores = [
            len(query_terms & set(row_tokens)) / max(1, len(query_terms))
            for row_tokens in tokens
        ]
    return _rank_rows(corpus, scores, "bm25_only", limit)


def _sparse_tfidf_bm25_retrieval(query: str, corpus: list[dict], limit: int = 5) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [_document_text(item) for item in corpus]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000)
    matrix = vectorizer.fit_transform(texts)
    tfidf_scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel().tolist()

    bm25_rows = _bm25_only_retrieval(query, corpus, limit=len(corpus))
    bm25_by_id = {row["id"]: row["retrieval_score"] for row in bm25_rows}
    bm25_max = max(bm25_by_id.values()) if bm25_by_id else 1.0
    bm25_max = bm25_max if bm25_max > 0 else 1.0

    scores = []
    for item, tfidf_score in zip(corpus, tfidf_scores):
        bm25_score = bm25_by_id.get(item.get("id"), 0.0) / bm25_max
        scores.append(0.65 * bm25_score + 0.35 * float(tfidf_score))

    return _rank_rows(corpus, scores, "local_sparse_tfidf_bm25_index", limit)


def _rank_rows(corpus: list[dict], scores: list[float], backend: str, limit: int) -> list[dict]:
    rows = []
    for item, score in zip(corpus, scores):
        if score <= 0:
            continue
        rows.append({
            **item,
            "retrieval_score": round(float(score), 4),
            "backend": backend,
            "retrieval_backend": backend,
        })
    return sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)[:limit]


def _estimate_grounding(query: str, retrieved: list[dict]) -> float | None:
    if not retrieved:
        return None
    query_tokens = set(_tokenize(query))
    scores = []
    for item in retrieved[:3]:
        text_tokens = set(_tokenize(str(item.get("text") or "")))
        if not text_tokens:
            continue
        overlap = len(query_tokens & text_tokens)
        scores.append(min(1.0, overlap / max(1, len(query_tokens))))
    return round(sum(scores) / len(scores), 3) if scores else None


def _mean(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def _comparison_notes(results: dict, index_status: dict) -> dict:
    preferred = "dense_faiss_bm25_rrf_reranked"
    notes = [
        f"Active backend: {index_status.get('retrieval_backend', 'unknown')}",
        f"Dense available: {index_status.get('dense_available')}",
        f"Fusion: {index_status.get('fusion') or 'n/a'}",
    ]

    bm25 = results.get("bm25_only", {})
    dense = results.get("dense_faiss_bm25_rrf", {})
    reranked = results.get(preferred, {})

    if isinstance(dense.get("pass_rate"), float) and isinstance(bm25.get("pass_rate"), float):
        notes.append(f"Dense hybrid vs BM25-only pass rate: {dense['pass_rate'] - bm25['pass_rate']:+.3f}")

    if isinstance(reranked.get("pass_rate"), float) and isinstance(dense.get("pass_rate"), float):
        notes.append(f"Reranked vs dense hybrid pass rate: {reranked['pass_rate'] - dense['pass_rate']:+.3f}")

    return {
        "notes": notes,
        "winner": preferred if results else "unavailable",
        "caveat": (
            "Winner is chosen by the full production pipeline design; expand labeled cases before treating "
            "small pass-rate differences as significant."
        ),
    }


def _document_text(item: dict) -> str:
    return " ".join([
        str(item.get("title") or ""),
        str(item.get("text") or ""),
        " ".join(item.get("tags") or []),
        str(item.get("topic") or ""),
        str(item.get("section") or ""),
    ])


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9/-]+", (text or "").lower())
