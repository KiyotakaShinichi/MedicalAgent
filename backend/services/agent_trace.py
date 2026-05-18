"""Pipeline-trace envelope for the patient agent.

A single pure function — :func:`build_pipeline_trace` (underscore-aliased
to ``_trace`` for back-compat with the four in-module call sites in
``agent_rag.py``) — produces the ``pipeline_trace`` dict that every
agent response carries.  Five terminal steps are defined as constants
so callers and reviewers can grep for the contract.

Why this lives in its own module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The trace structure is the single artifact that drives the admin
"trace replay" panel and several test assertions
(``pipeline_trace["terminal_step"] == "direct_support"`` and friends).
Pulling it out of the agent_rag god module makes the contract obvious
and unit-testable without spinning up the retrieval stack.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


# ─── Pipeline step constants ─────────────────────────────────────────────────


# The ordered list of named pipeline stages that ``build_pipeline_trace``
# embeds in every trace under the ``steps`` key.  Kept as a top-level
# tuple so reviewers can see the full agent contract at a glance.
PIPELINE_STEPS: tuple[str, ...] = (
    "safety_scope_check",
    "intent_router",
    "query_rewrite_decomposition",
    "exact_cache_check",
    "semantic_cache_check",
    "hybrid_retrieval",
    "parent_child_window_expansion",
    "reranker",
    "contextual_compression",
    "answer_generation",
    "validation_citation_check",
    "safe_cache_store",
)


# Recognized terminal_step values — one per branch that
# ``run_patient_agent_pipeline`` can exit through.  Tests and the trace
# replay panel match against these literals.
TERMINAL_INPUT_GUARDRAIL_BLOCK = "input_guardrail_block"
TERMINAL_CACHE_HIT             = "cache_hit"
TERMINAL_DIRECT_SUPPORT        = "direct_support"
TERMINAL_GENERATED             = "generated"

ALL_TERMINAL_STEPS: tuple[str, ...] = (
    TERMINAL_INPUT_GUARDRAIL_BLOCK,
    TERMINAL_CACHE_HIT,
    TERMINAL_DIRECT_SUPPORT,
    TERMINAL_GENERATED,
)


# ─── Trace builder ───────────────────────────────────────────────────────────


def build_pipeline_trace(
    safety: Mapping[str, Any],
    intent: str | None,
    rewritten: Mapping[str, Any],
    retrieved: Iterable[Any],
    reranked: Iterable[Any],
    compressed: Iterable[Any],
    terminal_step: str,
    cache_policy: Any = None,
) -> dict[str, Any]:
    """Build the pipeline_trace envelope embedded in every agent response.

    The signature mirrors the original inline ``_trace`` for drop-in
    compatibility — ``safety`` and ``rewritten`` are accessed via
    ``.get(...)``, so empty dicts work for the early-exit branches
    (input_guardrail_block, cache_hit, direct_support).
    """
    return {
        "steps":            list(PIPELINE_STEPS),
        "terminal_step":    terminal_step,
        "safety_level":     safety.get("level") if isinstance(safety, Mapping) else None,
        "intent":           intent,
        "subquery_count":   len((rewritten.get("subqueries") if isinstance(rewritten, Mapping) else None) or []),
        "retrieved_count":  _safe_len(retrieved),
        "reranked_count":   _safe_len(reranked),
        "compressed_count": _safe_len(compressed),
        "cache_policy":     cache_policy,
    }


def _safe_len(value: Iterable[Any]) -> int:
    """``len`` that tolerates generators and ``None``."""
    if value is None:
        return 0
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        # Generator / lazy iterable — count by materializing into a list.
        return sum(1 for _ in value)


# Back-compat alias: agent_rag.py's four call sites use ``_trace(...)``.
_trace = build_pipeline_trace


__all__ = [
    "PIPELINE_STEPS",
    "TERMINAL_INPUT_GUARDRAIL_BLOCK",
    "TERMINAL_CACHE_HIT",
    "TERMINAL_DIRECT_SUPPORT",
    "TERMINAL_GENERATED",
    "ALL_TERMINAL_STEPS",
    "build_pipeline_trace",
    "_trace",
]
