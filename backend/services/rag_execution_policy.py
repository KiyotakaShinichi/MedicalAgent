"""Intent-aware execution policy for the live RAG path.

The policy keeps source governance non-negotiable and decides only whether
optional retrieval stages are justified. It is deterministic, traceable, and
does not claim clinical correctness or retrieval superiority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from backend.services.rag_intent_modes import RagModeConfig, select_mode
from backend.services.rag_tier_filter import FilterResult, filter_chunks_by_mode


@dataclass(frozen=True)
class RagExecutionPolicy:
    mode: str | None
    apply_parent_child: bool
    apply_reranker: bool
    max_governed_candidates: int
    reason: str
    clinical_validation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_rag_execution(
    *,
    intent: str,
    rewritten: Mapping[str, Any],
    actor_role: str | None = None,
) -> tuple[RagExecutionPolicy, RagModeConfig | None]:
    mode = select_mode(intent, actor_role=actor_role)
    if mode is None:
        return RagExecutionPolicy(None, False, False, 0, "intent_has_no_rag_mode"), None

    query = str(rewritten.get("expanded_query") or rewritten.get("normalized_query") or "")
    subqueries = rewritten.get("subqueries") or rewritten.get("decomposed_queries") or []
    complex_query = len(query.split()) >= 11 or len(subqueries) > 1
    if mode.mode == "portal_help_rag":
        return RagExecutionPolicy(
            mode.mode,
            apply_parent_child=False,
            apply_reranker=False,
            max_governed_candidates=mode.max_retrieved_chunks,
            reason="portal_help_uses_narrow_governed_docs_without_context_expansion",
        ), mode
    if mode.mode == "education_rag" and not complex_query:
        return RagExecutionPolicy(
            mode.mode,
            apply_parent_child=False,
            apply_reranker=True,
            max_governed_candidates=max(mode.max_retrieved_chunks * 4, 12),
            reason="simple_education_skips_parent_child_but_retains_ranking",
        ), mode
    return RagExecutionPolicy(
        mode.mode,
        apply_parent_child=True,
        apply_reranker=True,
        max_governed_candidates=max(mode.max_retrieved_chunks * 5, 15),
        reason="complex_or_record_query_uses_context_expansion_and_ranking",
    ), mode


def govern_candidates(
    chunks: list[Mapping[str, Any]],
    mode: RagModeConfig | None,
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode is None:
        rows = [dict(chunk) for chunk in chunks]
        return rows, {
            "mode": None,
            "kept_count": len(rows),
            "dropped_count": 0,
            "reason": "no_rag_mode_filter_not_applicable",
        }
    filtered: FilterResult = filter_chunks_by_mode(chunks, mode)
    kept = filtered.kept_chunks[:limit] if limit else filtered.kept_chunks
    trace = filtered.to_dict()
    trace["applied_before_generation"] = True
    trace["limited_to"] = limit
    trace["kept_after_limit"] = len(kept)
    return kept, trace


__all__ = ["RagExecutionPolicy", "govern_candidates", "plan_rag_execution"]
