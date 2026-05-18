"""Cache facade for the RAG agent."""

from backend.services.agent_cache import (  # noqa: F401
    AGENT_CACHE_SCHEMA_VERSION,
    AGENT_CACHE_TTL_DAYS,
    SEMANTIC_CACHE_MIN_SIMILARITY,
    exact_cache_check,
    is_cacheable,
    semantic_cache_check,
    store_cache,
)

__all__ = [
    "AGENT_CACHE_SCHEMA_VERSION",
    "AGENT_CACHE_TTL_DAYS",
    "SEMANTIC_CACHE_MIN_SIMILARITY",
    "exact_cache_check",
    "is_cacheable",
    "semantic_cache_check",
    "store_cache",
]
