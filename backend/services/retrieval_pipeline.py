"""Retrieval-pipeline facade for the patient RAG agent.

Dense/sparse search, parent-child expansion, reranking, and compression are
implemented in :mod:`backend.services.agent_retrieval`. This module is the
responsibility-named import surface for new code and tests.
"""

from backend.services.agent_retrieval import (  # noqa: F401
    contextual_compression,
    expand_parent_child_windows,
    hybrid_retrieval,
    rerank_context,
)

__all__ = [
    "contextual_compression",
    "expand_parent_child_windows",
    "hybrid_retrieval",
    "rerank_context",
]
