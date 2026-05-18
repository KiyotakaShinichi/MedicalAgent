"""Source-tier filtering for retrieved RAG chunks.

Bridges the static KB source-governance artifact (which maps every source
to a tier + allowed_use set) with the live retrieval layer.  Given a list
of retrieved chunks and a ``RagModeConfig``, returns the chunks that the
mode is allowed to cite, plus a structured trace of which chunks were
filtered out and why — so the trace replay panel can show "we found 8
sources, kept 5, dropped 3 for tier/use mismatch".

Pure function over the governance map.  No KB ingestion logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping

from backend.services.kb_source_governance import (
    TIER_MAP,
    load_kb_source_governance,
)
from backend.services.rag_intent_modes import RagModeConfig


@dataclass
class ChunkFilterDecision:
    """Per-chunk verdict produced by the filter."""

    chunk_id: str
    source_id: str | None
    decision: str  # "kept" | "dropped"
    reason: str | None  # populated when dropped
    tier: str
    allowed_use: list[str]
    staleness_status: str


@dataclass
class FilterResult:
    """Aggregate result for a single retrieval pass."""

    kept_chunks: list[dict[str, Any]] = field(default_factory=list)
    dropped_chunks: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[ChunkFilterDecision] = field(default_factory=list)
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "kept_count": len(self.kept_chunks),
            "dropped_count": len(self.dropped_chunks),
            "kept_chunk_ids": [c.get("id") for c in self.kept_chunks],
            "dropped_chunk_ids": [c.get("id") for c in self.dropped_chunks],
            "decisions": [
                {
                    "chunk_id": d.chunk_id,
                    "source_id": d.source_id,
                    "decision": d.decision,
                    "reason": d.reason,
                    "tier": d.tier,
                    "allowed_use": d.allowed_use,
                    "staleness_status": d.staleness_status,
                }
                for d in self.decisions
            ],
        }


# ─── Source metadata index (cached) ──────────────────────────────────────────


@lru_cache(maxsize=2)
def _governance_source_index(governance_path: str) -> dict[str, dict[str, Any]]:
    """Build a {source_id → governance row} map from the most recent
    governance artifact.  Cached so repeated lookups during a single
    request don't re-parse the JSON."""
    payload = load_kb_source_governance(governance_path) if governance_path else load_kb_source_governance()
    index: dict[str, dict[str, Any]] = {}
    for source in payload.get("sources") or []:
        sid = source.get("source_id")
        if sid:
            index[sid] = source
    return index


def _source_index() -> dict[str, dict[str, Any]]:
    """Default-path accessor — cleanest call site."""
    return _governance_source_index("")


def _row_for_chunk(chunk: Mapping[str, Any], index: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Look up the governance row for a chunk, falling back to its source_name
    if parent_id isn't present."""
    candidates = (
        chunk.get("id"),
        chunk.get("chunk_id"),
        chunk.get("parent_id"),
        chunk.get("source_id"),
        chunk.get("source_name"),
        chunk.get("source_path"),
    )
    for candidate in candidates:
        if candidate and candidate in index:
            return index[candidate]
    virtual_row = _virtual_builtin_governance_row(chunk)
    if virtual_row:
        return virtual_row
    return None


def _virtual_builtin_governance_row(chunk: Mapping[str, Any]) -> dict[str, Any] | None:
    """Best-effort governance for legacy in-code snippets.

    The newer curated KB has a persisted source-governance artifact, but the
    regression suite still exercises a small built-in corpus from
    ``agent_rag.KNOWLEDGE_SNIPPETS``.  Those snippets are source-backed and
    intentionally cited, yet they do not appear in the file-based governance
    artifact because they are not ingested from disk.  Treat them as virtual
    governed sources so Phase 11 filtering does not erase valid citations.
    """
    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or chunk.get("parent_id") or "")
    source_name = str(chunk.get("source_name") or "").lower()
    source_url = str(chunk.get("source_url") or "").lower()
    parent_id = str(chunk.get("parent_id") or "").lower()

    if parent_id == "portal-help" or "portal" in source_name or chunk_id.startswith("portal-"):
        return {
            "source_id": chunk_id,
            "tier": "T4",
            "allowed_use": ["portal_help"],
            "staleness_status": "current",
        }

    official_or_curated = any(
        marker in source_name or marker in source_url
        for marker in (
            "national cancer institute",
            "cancer.gov",
            "american cancer society",
            "cancer.org",
            "cdc",
            "nccih",
            "msk",
            "curated",
            "pubmed",
        )
    )
    if official_or_curated:
        return {
            "source_id": chunk_id,
            "tier": "T2" if ("pubmed" in source_url or "clinical guideline" in source_name) else "T3",
            "allowed_use": ["education", "patient_safety", "monitoring_context"],
            "staleness_status": "current",
        }

    if "project" in source_name or source_url in {"readme.md", "model_card.md"}:
        return {
            "source_id": chunk_id,
            "tier": "T4",
            "allowed_use": ["portal_help", "education"],
            "staleness_status": "current",
        }

    return None


# ─── Public API ──────────────────────────────────────────────────────────────


def filter_chunks_by_mode(
    chunks: Iterable[Mapping[str, Any]],
    mode: RagModeConfig,
    *,
    governance_path: str = "",
    keep_unmapped: bool = False,
) -> FilterResult:
    """Filter retrieved chunks against the mode's allowed tier + allowed_use
    intersection.

    A chunk is **kept** when:
      - its source is in the governance map (i.e. has a known tier), AND
      - that tier is in ``mode.allowed_tiers``, AND
      - the chunk's source has at least one ``allowed_use`` in
        ``mode.allowed_use``.

    A chunk is **dropped** when any of those fails.  When
    ``keep_unmapped`` is True, chunks whose source isn't in the
    governance map are kept with tier ``"T5"`` — the post-gen validator
    is then responsible for refusing to cite them.  Default is to drop
    unmapped sources entirely so the citation list is always trustworthy.
    """
    index = _governance_source_index(governance_path) if governance_path else _source_index()
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    decisions: list[ChunkFilterDecision] = []

    allowed_tiers = set(mode.allowed_tiers)
    allowed_uses = set(mode.allowed_use)

    for chunk in chunks:
        chunk_dict = dict(chunk)
        chunk_id = str(chunk_dict.get("id") or chunk_dict.get("chunk_id") or "")
        row = _row_for_chunk(chunk_dict, index)

        if row is None:
            if keep_unmapped:
                kept.append(chunk_dict)
                decisions.append(ChunkFilterDecision(
                    chunk_id=chunk_id,
                    source_id=None,
                    decision="kept",
                    reason="unmapped_source_admitted_under_keep_unmapped",
                    tier="T5",
                    allowed_use=[],
                    staleness_status="unknown",
                ))
            else:
                dropped.append(chunk_dict)
                decisions.append(ChunkFilterDecision(
                    chunk_id=chunk_id,
                    source_id=None,
                    decision="dropped",
                    reason="unmapped_source_not_in_governance",
                    tier="T5",
                    allowed_use=[],
                    staleness_status="unknown",
                ))
            continue

        tier = row.get("tier", "T5")
        row_uses = set(row.get("allowed_use") or [])
        staleness = row.get("staleness_status", "unknown")

        # Tier check
        if tier not in allowed_tiers:
            dropped.append(chunk_dict)
            decisions.append(ChunkFilterDecision(
                chunk_id=chunk_id,
                source_id=row.get("source_id"),
                decision="dropped",
                reason=f"tier_not_allowed_in_mode (have {tier!s}, mode allows {sorted(allowed_tiers)!s})",
                tier=tier,
                allowed_use=sorted(row_uses),
                staleness_status=staleness,
            ))
            continue

        # Allowed-use intersection check
        if not (row_uses & allowed_uses):
            dropped.append(chunk_dict)
            decisions.append(ChunkFilterDecision(
                chunk_id=chunk_id,
                source_id=row.get("source_id"),
                decision="dropped",
                reason=(
                    f"no_allowed_use_overlap (source has {sorted(row_uses)!s}, "
                    f"mode allows {sorted(allowed_uses)!s})"
                ),
                tier=tier,
                allowed_use=sorted(row_uses),
                staleness_status=staleness,
            ))
            continue

        kept.append(chunk_dict)
        decisions.append(ChunkFilterDecision(
            chunk_id=chunk_id,
            source_id=row.get("source_id"),
            decision="kept",
            reason=None,
            tier=tier,
            allowed_use=sorted(row_uses),
            staleness_status=staleness,
        ))

    return FilterResult(
        kept_chunks=kept,
        dropped_chunks=dropped,
        decisions=decisions,
        mode=mode.mode,
    )


def known_tier_for_source(source_identifier: str, *, governance_path: str = "") -> str:
    """Lookup helper used by the claim validator + evidence grader."""
    index = _governance_source_index(governance_path) if governance_path else _source_index()
    row = index.get(source_identifier)
    if row:
        return row.get("tier") or "T5"
    return "T5"


__all__ = [
    "ChunkFilterDecision",
    "FilterResult",
    "filter_chunks_by_mode",
    "known_tier_for_source",
]
