"""Knowledge-base corpus loader for the patient agent.

Owns the process-level cache that merges the hand-curated
``KNOWLEDGE_SNIPPETS`` seed corpus with the ingested chunks produced by
the KB-ingestion pipeline.  Three consumers reach into this module:

  - retrieval (``agent_retrieval``) needs the corpus on every query
  - the cache layer (``agent_cache``) needs the fingerprint to detect
    KB drift across rows
  - the orchestrator (``agent_rag``) calls
    :func:`_invalidate_kb_cache` after any in-process KB write

Public symbols
~~~~~~~~~~~~~~
- :func:`knowledge_snippets` (alias ``_knowledge_snippets``) — merged
  corpus (cached after first call).
- :func:`invalidate_kb_cache` (alias ``_invalidate_kb_cache``) — clear
  the cache so the next call re-reads from disk.
- :func:`get_rag_corpus` — public alias of ``knowledge_snippets`` for
  external tooling.
- :func:`knowledge_base_fingerprint` — stable hash of the merged corpus,
  used by the cache layer to detect KB drift.

Extracted from ``agent_rag.py`` as part of the agent_rag.py module
split.  Re-exported from ``backend.services.agent_rag`` so the existing
import contract is preserved.
"""
from __future__ import annotations

from backend.services.agent_knowledge_snippets import KNOWLEDGE_SNIPPETS
from backend.services.kb_ingestion import load_ingested_chunks
from backend.services.rag_vector_index import corpus_fingerprint


# Module-level cache for the merged KB corpus.  None means "not loaded
# yet"; calling :func:`invalidate_kb_cache` resets it so the next read
# pulls fresh ingested chunks.  Concurrency note: this is process-local
# state; the agent runs single-threaded per request, so we don't
# bother with a lock.
_KB_CORPUS_CACHE: list | None = None


def knowledge_snippets() -> list:
    """Return the merged corpus (KNOWLEDGE_SNIPPETS + ingested chunks).
    First call materializes; subsequent calls hit the cache."""
    global _KB_CORPUS_CACHE
    if _KB_CORPUS_CACHE is None:
        _KB_CORPUS_CACHE = list(KNOWLEDGE_SNIPPETS) + load_ingested_chunks()
    return _KB_CORPUS_CACHE


def invalidate_kb_cache() -> None:
    """Clear the merged-corpus cache.  Call after ingesting new KB
    chunks so the next pipeline call reloads from disk."""
    global _KB_CORPUS_CACHE
    _KB_CORPUS_CACHE = None


def get_rag_corpus() -> list:
    """Public alias for :func:`knowledge_snippets`.  External tooling
    (admin scripts, KB exporters) imports this name from agent_rag."""
    return knowledge_snippets()


def knowledge_base_fingerprint() -> str:
    """Stable hash of the merged corpus.  Used by the cache layer to
    detect KB drift across cache rows."""
    return corpus_fingerprint(knowledge_snippets())


# Underscore aliases preserve agent_rag's internal references.
_knowledge_snippets = knowledge_snippets
_invalidate_kb_cache = invalidate_kb_cache


__all__ = [
    "knowledge_snippets",
    "invalidate_kb_cache",
    "get_rag_corpus",
    "knowledge_base_fingerprint",
    "_knowledge_snippets",
    "_invalidate_kb_cache",
]
