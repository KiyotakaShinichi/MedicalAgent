"""Citation assembly helpers for generated RAG answers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def assemble_citation_ids(chunks: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return stable citation/source identifiers from retrieved chunks.

    This small helper exists so trace, eval, and answer packaging code use the
    same fallback order.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        candidate = str(
            chunk.get("parent_id")
            or chunk.get("source_id")
            or chunk.get("id")
            or chunk.get("chunk_id")
            or ""
        ).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids


__all__ = ["assemble_citation_ids"]
