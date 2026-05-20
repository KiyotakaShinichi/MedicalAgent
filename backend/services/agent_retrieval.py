"""Retrieval + reranking + compression for the patient agent.

Pulled out of ``agent_rag.py`` as part of the god-module split.  The
public surface mirrors the original inline functions:

  - :func:`hybrid_retrieval` — sparse + dense + RRF fusion against the
    KB index, with agent-specific intent / domain / section / curated
    boosts.  Falls back to a metadata-overlap scorer when the dense
    index hasn't been built (fresh checkouts).
  - :func:`expand_parent_child_windows` — pulls sibling chunks from the
    same parent so retrieved evidence keeps surrounding context.
  - :func:`rerank_context` — final reranker combining lexical coverage,
    safety + source boosts, and an optional cross-encoder pass.
  - :func:`contextual_compression` — caps the post-rerank chunk list at
    ``MAX_CONTEXT_CHARS`` (1300 chars across at most 3 chunks).

Lazy imports
~~~~~~~~~~~~
``_knowledge_snippets`` and ``knowledge_base_fingerprint`` live in
``agent_rag``; importing them at module top would create a cycle
because ``agent_rag`` re-imports from this module.  Both are accessed
via local imports inside the functions that need them.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Mapping

from backend.services.agent_query_rewriting import tokenize
from backend.services.cross_encoder_reranker import rerank_with_cross_encoder
from backend.services.rag_vector_index import search_hybrid_index


# ─── Module-level constants + state ──────────────────────────────────────────


MAX_CONTEXT_CHARS: int = 1300


# Sources that the agent treats as "curated" — these get a +1.0 source
# boost during reranking so the model card / safety policy / portal
# guide always make it into the answer composition when retrieved.
CURATED_SOURCES: frozenset[str] = frozenset({
    "CDC",
    "National Cancer Institute",
    "American Cancer Society",
    "Project model card",
    "Project safety policy",
    "Project patient portal guide",
    "Project feature rationale",
    "Project agent design",
})


# Cross-encoder reranker state.  Populated lazily by
# :func:`_get_cross_encoder` on first successful load; errors are also
# cached so we don't retry the failing init on every request.
_CROSS_ENCODER_CACHE: dict[str, Any] = {}


# ─── Boost tables ────────────────────────────────────────────────────────────


_INTENT_BOOST_TARGETS: dict[str, frozenset[str]] = {
    "portal_help":               frozenset({"upload", "portal", "labs", "mri"}),
    "education":                 frozenset({
        "pcr", "cbc", "wbc", "chemotherapy", "side effects",
        "mri", "ct", "ascites", "imaging", "radiomics", "machine learning",
    }),
    "patient_timeline_monitoring": frozenset({"score", "monitoring", "cbc", "response", "mri_response_monitoring"}),
    "safety_boundary":           frozenset({"urgent", "fever", "infection", "clinical safety", "cbc_toxicity_monitoring"}),
    "treatment_decision_boundary": frozenset({"treatment", "doctor", "chemotherapy"}),
    "emotional_support":         frozenset({"symptoms", "doctor", "side effects"}),
}


# ─── Boost functions ─────────────────────────────────────────────────────────


def intent_boost(intent: str, snippet: Mapping[str, Any]) -> float:
    tags = set(snippet["tags"])
    desired = _INTENT_BOOST_TARGETS.get(intent, frozenset())
    topic = snippet.get("topic")
    return 0.25 if (tags & desired or topic in desired) else 0.0


def domain_boost(query_tokens: set[str], snippet: Mapping[str, Any]) -> float:
    tags = set(snippet.get("tags") or [])
    topic = snippet.get("topic") or ""
    modalities = {item.lower() for item in (snippet.get("modality") or [])}
    boost = 0.0
    if query_tokens & {"mri", "dce", "radiomics", "imaging", "pcr", "response"}:
        if "mri" in modalities or "mri" in tags or "mri" in topic:
            boost += 0.2
    if query_tokens & {"ct", "ascites", "peritoneal"}:
        if {"ct", "ascites", "peritoneal"} & (tags | modalities) or "ct_report" in topic:
            boost += 0.45
    if query_tokens & {"cbc", "wbc", "anc", "neutropenia", "platelets", "hemoglobin", "fever"}:
        if "cbc" in modalities or "cbc" in tags or "toxicity" in tags or "neutropenia" in topic:
            boost += 0.2
    if query_tokens & {"chemo", "chemotherapy", "neoadjuvant", "treatment"}:
        if "chemotherapy" in tags or "treatment" in modalities or "treatment" in topic:
            boost += 0.12
    return boost


def section_boost(snippet: Mapping[str, Any]) -> float:
    section = snippet.get("section") or ""
    if section in {"abstract", "conclusion", "conclusions", "clinical implications", "results"}:
        return 0.08
    if section in {"references"}:
        return -0.25
    return 0.0


# Underscore aliases for back-compat with agent_rag's inline references.
_intent_boost = intent_boost
_domain_boost = domain_boost
_section_boost = section_boost


# ─── Cross-encoder reranker ──────────────────────────────────────────────────


def _cross_encoder_enabled() -> bool:
    if os.getenv("RAG_ENABLE_CROSS_ENCODER", "").strip().lower() not in {"1", "true", "yes"}:
        return False
    return importlib.util.find_spec("sentence_transformers") is not None


def _get_cross_encoder():
    if not _cross_encoder_enabled():
        return None
    if "model" not in _CROSS_ENCODER_CACHE:
        try:
            from sentence_transformers import CrossEncoder

            model_name = os.getenv("RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            _CROSS_ENCODER_CACHE["model"] = CrossEncoder(model_name)
            _CROSS_ENCODER_CACHE["name"] = model_name
        except Exception as exc:  # noqa: BLE001 — failure is a fallback signal
            _CROSS_ENCODER_CACHE["error"] = str(exc)
            return None
    return _CROSS_ENCODER_CACHE.get("model")


def _cross_encoder_scores(query: str, expanded: list[dict]) -> list[float] | None:
    model = _get_cross_encoder()
    if model is None or not expanded:
        return None
    pairs = [(query, (item.get("title") or "") + "\n" + (item.get("text") or "")) for item in expanded]
    try:
        raw_scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        _CROSS_ENCODER_CACHE["error"] = str(exc)
        return None
    values = [float(score) for score in raw_scores]
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _reranker_backend(cross_score) -> str:
    if cross_score is not None:
        return f"cross_encoder:{_CROSS_ENCODER_CACHE.get('name') or 'enabled'}"
    if _CROSS_ENCODER_CACHE.get("error"):
        return f"heuristic_fallback:{_CROSS_ENCODER_CACHE['error'][:80]}"
    return "heuristic_metadata_safety_reranker"


# ─── Retrieval ───────────────────────────────────────────────────────────────


def hybrid_retrieval(rewritten: Mapping[str, Any], intent: str) -> list[dict[str, Any]]:
    """Sparse + dense + RRF fusion against the KB index, with
    agent-specific intent / domain / section / curated boosts.  Falls
    back to a metadata-overlap scorer when the dense index hasn't been
    built."""
    from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint

    query_tokens = set(tokenize(rewritten["expanded_query"]))
    snippets = _knowledge_snippets()
    indexed_rows = search_hybrid_index(
        query=rewritten["expanded_query"],
        corpus=snippets,
        intent=intent,
        knowledge_fingerprint=knowledge_base_fingerprint(),
        candidate_limit=int(os.getenv("RAG_RRF_CANDIDATE_LIMIT", "40")),
    )
    if indexed_rows:
        rows = []
        for item in indexed_rows:
            ib = intent_boost(intent, item)
            db = domain_boost(query_tokens, item)
            sb = section_boost(item)
            is_curated = item.get("builtin") or item.get("source_name") in CURATED_SOURCES
            curated_boost = 1.0 if is_curated else 0.0
            score = float(item.get("retrieval_score", 0)) + ib + db + sb + curated_boost
            rows.append({
                **item,
                "retrieval_score": round(score, 4),
                "retrieval_trace": {
                    "backend":       item.get("backend") or item.get("retrieval_backend"),
                    "sparse_score":  item.get("sparse_score"),
                    "dense_score":   item.get("dense_score"),
                    "rrf_score":     item.get("rrf_score"),
                    "fusion_score":  item.get("fusion_score"),
                    "metadata_score": item.get("metadata_score"),
                    # backward-compat aliases
                    "vector_score":  item.get("vector_score"),
                    "lexical_score": item.get("lexical_score"),
                    "agent_intent_boost":  round(ib, 4),
                    "agent_domain_boost":  round(db, 4),
                    "agent_section_boost": round(sb, 4),
                    "agent_curated_boost": round(curated_boost, 4),
                },
            })
        return sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)[:40]

    # Fallback path — no dense index available.
    rows = []
    for snippet in snippets:
        metadata_text = " ".join([
            snippet.get("title", ""),
            snippet.get("text", ""),
            " ".join(snippet.get("tags", [])),
            snippet.get("topic") or "",
            " ".join(snippet.get("modality", []) or []),
            snippet.get("care_stage") or "",
            snippet.get("section") or "",
        ])
        text_tokens = set(tokenize(metadata_text))
        lexical = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        metadata_terms = set(snippet.get("tags", []))
        metadata_terms.update(tokenize(snippet.get("topic") or ""))
        metadata_terms.update(tokenize(" ".join(snippet.get("modality", []) or [])))
        semantic = len(query_tokens & metadata_terms) / max(len(metadata_terms), 1)
        score = lexical + semantic + intent_boost(intent, snippet) + domain_boost(query_tokens, snippet) + section_boost(snippet)
        if score > 0:
            rows.append({
                **snippet,
                "retrieval_score": round(score, 4),
                "matched_terms": sorted(query_tokens & text_tokens)[:10],
            })
    return sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)[:40]


def expand_parent_child_windows(retrieved: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pull sibling chunks from the same parent so retrieved evidence
    keeps surrounding context."""
    from backend.services.agent_rag import _knowledge_snippets

    seen = {item["id"] for item in retrieved}
    expanded = list(retrieved)
    parent_ids = {item["parent_id"] for item in retrieved}
    for snippet in _knowledge_snippets():
        if snippet["parent_id"] in parent_ids and snippet["id"] not in seen:
            expanded.append({
                **snippet,
                "retrieval_score": 0.15,
                "matched_terms":   [],
                "expansion":       "parent_child_window",
            })
            seen.add(snippet["id"])
    return expanded


# ─── Reranking + compression ─────────────────────────────────────────────────


def rerank_context(
    expanded: list[Mapping[str, Any]],
    rewritten: Mapping[str, Any],
    intent: str,
    safety: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Final reranker — lexical coverage + safety + source boosts +
    optional cross-encoder."""
    query_tokens = set(tokenize(rewritten["expanded_query"]))
    reranked = []
    for item in expanded:
        tags = set(item["tags"])
        coverage = len(query_tokens & tags)
        safety_boost = 0.4 if safety.get("level") == "high_risk" and "urgent" in tags else 0
        is_curated = item.get("builtin") or item.get("source_name") in CURATED_SOURCES
        source_boost = 1.0 if is_curated else 0.05
        final_score = (
            float(item.get("retrieval_score", 0))
            + coverage * 0.18
            + safety_boost
            + source_boost
        )
        reranked.append({
            **item,
            "rerank_score":        round(final_score, 4),
            "reranker_backend":    "heuristic_metadata_safety_reranker",
        })
    heuristic_rows = sorted(reranked, key=lambda row: row["rerank_score"], reverse=True)
    top_rows, telemetry = rerank_with_cross_encoder(
        rewritten["expanded_query"],
        heuristic_rows,
        top_k=5,
        candidate_limit=int(os.getenv("RAG_CROSS_ENCODER_CANDIDATE_LIMIT", "40")),
    )
    return [
        {
            **row,
            "rerank_telemetry": telemetry,
        }
        for row in top_rows
    ]


def contextual_compression(reranked: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cap the post-rerank chunk list at MAX_CONTEXT_CHARS across at
    most 3 chunks.  Carries parent_id forward so the tier filter can
    resolve each chunk against the KB governance index."""
    compressed: list[dict[str, Any]] = []
    total = 0
    for item in reranked:
        text = item["text"]
        if total + len(text) > MAX_CONTEXT_CHARS and compressed:
            continue
        compressed.append({
            "id":                  item["id"],
            "parent_id":           item.get("parent_id"),
            "title":               item["title"],
            "source_name":         item["source_name"],
            "source_url":          item["source_url"],
            "section":             item.get("section"),
            "topic":               item.get("topic"),
            "confidence":          item.get("confidence"),
            "text":                text,
            "score":               item.get("rerank_score", item.get("retrieval_score")),
            "reranker_backend":    item.get("reranker_backend"),
            "cross_encoder_score": item.get("cross_encoder_score"),
            "cross_encoder_latency_ms": item.get("cross_encoder_latency_ms"),
            "retrieval_backend":   item.get("retrieval_backend"),
            "retrieval_trace":     item.get("retrieval_trace"),
            "allowed_use":         item.get("allowed_use"),
            "source_tier":         item.get("source_tier"),
            "staleness":           item.get("staleness"),
            "rerank_telemetry":    item.get("rerank_telemetry"),
        })
        total += len(text)
        if len(compressed) >= 3:
            break
    return compressed


# Back-compat: the agent_rag module previously held a ``_CURATED_SOURCES``
# set used by both retrieval and reranking.  Some old call sites in the
# codebase might still reference it through agent_rag, so we re-expose
# it as the underscore name too.
_CURATED_SOURCES = CURATED_SOURCES


__all__ = [
    "MAX_CONTEXT_CHARS",
    "CURATED_SOURCES",
    "_CURATED_SOURCES",
    "intent_boost", "_intent_boost",
    "domain_boost", "_domain_boost",
    "section_boost", "_section_boost",
    "hybrid_retrieval",
    "expand_parent_child_windows",
    "rerank_context",
    "contextual_compression",
]
