"""Source-governance filtering facade for RAG retrieval results."""

from backend.services.rag_tier_filter import (  # noqa: F401
    ChunkFilterDecision,
    FilterResult,
    LEGACY_BUILTIN_SOURCE_GOVERNANCE,
    filter_chunks_by_mode,
    known_tier_for_source,
)

__all__ = [
    "ChunkFilterDecision",
    "FilterResult",
    "LEGACY_BUILTIN_SOURCE_GOVERNANCE",
    "filter_chunks_by_mode",
    "known_tier_for_source",
]
