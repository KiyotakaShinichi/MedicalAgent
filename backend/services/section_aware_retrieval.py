"""Query-aware section preference for research-paper retrieval.

The base index already preserves section metadata.  This module adds a small,
auditable ranking adjustment instead of rebuilding or duplicating the corpus.
It is experimental until ablation evidence supports promotion.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


SECTION_ALIASES: dict[str, frozenset[str]] = {
    "abstract": frozenset({"abstract", "highlights", "summary", "plain language summary", "key points"}),
    "introduction": frozenset({"introduction", "background"}),
    "methods": frozenset({"methods", "materials and methods", "patients and methods"}),
    "results": frozenset({"results"}),
    "discussion": frozenset({"discussion"}),
    "conclusion": frozenset({"conclusion", "conclusions", "clinical implications"}),
}

_SECTION_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("methods", re.compile(r"\b(methods?|methodology|study\s+design|how\s+(?:was|were|did)|participants?|cohort|protocol)\b", re.I)),
    ("results", re.compile(r"\b(results?|findings?|outcomes?|what\s+(?:did|was|were)\s+(?:the\s+)?study\s+find)\b", re.I)),
    ("discussion", re.compile(r"\b(discussion|interpretation|limitations?|implications?)\b", re.I)),
    ("conclusion", re.compile(r"\b(conclusions?|takeaway|bottom\s+line|authors?\s+conclude)\b", re.I)),
    ("introduction", re.compile(r"\b(introduction|background|rationale|why\s+was\s+the\s+study)\b", re.I)),
    ("abstract", re.compile(r"\b(abstract|which\s+(?:paper|source|study|article)|find\s+the\s+paper|may\s+(?:paper|source)|anong\s+paper|aling\s+source)\b", re.I)),
)


def canonical_section(value: Any) -> str:
    section = re.sub(r"\s+", " ", str(value or "").strip().lower())
    for canonical, aliases in SECTION_ALIASES.items():
        if section in aliases:
            return canonical
    return section or "unknown"


def infer_preferred_sections(query: str) -> tuple[str, ...]:
    matches = [name for name, pattern in _SECTION_CUES if pattern.search(query or "")]
    return tuple(dict.fromkeys(matches))


def rerank_by_section(
    query: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    boost: float = 0.12,
) -> list[dict[str, Any]]:
    preferred = set(infer_preferred_sections(query))
    reranked: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        section = canonical_section(row.get("section"))
        base_score = float(row.get("retrieval_score") or row.get("rerank_score") or 0.0)
        section_delta = boost if section in preferred else 0.0
        if section == "references":
            section_delta -= boost
        row["section_rerank_score"] = round(base_score + section_delta, 6)
        row["section_rerank_delta"] = round(section_delta, 6)
        row["section_rerank_preference"] = sorted(preferred)
        row["section_rerank_original_rank"] = index + 1
        reranked.append(row)
    return sorted(
        reranked,
        key=lambda item: (
            float(item.get("section_rerank_score") or 0.0),
            -int(item.get("section_rerank_original_rank") or 0),
        ),
        reverse=True,
    )


__all__ = ["SECTION_ALIASES", "canonical_section", "infer_preferred_sections", "rerank_by_section"]
