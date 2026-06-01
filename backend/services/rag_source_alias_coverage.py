"""Source-alias coverage diagnostic for the RAG retrieval goldset.

Background
~~~~~~~~~~
``backend/services/rag_baseline_comparison.LOGICAL_SOURCE_ALIASES`` is a
hand-maintained dict mapping each goldset ``expected_source_id`` to the
set of KB chunk identifiers / titles that should count as the same
logical source.

`Data/evals/rag/latest_retrieval_failure_analysis.json` flags
``gold_source_id_alias_or_metadata_normalization`` as the dominant
failure category in the harder goldset (10 of 74 cases). The retriever
brings back the *right content* but the eval can't match it to the
gold's expected ID because the alias map is incomplete.

This module is a **read-only diagnostic**: it walks the goldset, walks
the live KB, and emits an artifact listing per-logical-alias coverage
plus *proposed* alias additions discovered via content matching.

Contract
~~~~~~~~
- No KB chunks are added/removed.
- No retrieval ranking changes.
- No goldset cases are added/removed.
- The output JSON is informational only; promoting a proposed alias
  into ``LOGICAL_SOURCE_ALIASES`` is a separate, reviewer-gated step.

Output: ``Data/evals/rag/latest_source_alias_coverage.json``
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_GOLDSET_PATH = Path("Data/evals/rag/retrieval_goldset.jsonl")
DEFAULT_KB_PATH = Path("Data/rag_knowledge_base_chunks.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_source_alias_coverage.json")


# ─── Content-match heuristics ───────────────────────────────────────────
#
# A logical alias key (e.g. "infection-safety") is matched against a KB
# chunk by intersecting the alias key's tokens with the chunk's
# title/source_name/topic tokens.  Tokens are lowercase, alphanumeric,
# stripped of common stopwords.

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "is", "are",
    "for", "with", "as", "at", "by", "from", "this", "that", "during",
    "during", "during", "of", "and", "or", "in", "on", "to", "for",
    "during", "guide", "reference", "patient", "patients", "breast",
    "cancer", "treatment",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _alias_key_tokens(alias_key: str) -> set[str]:
    """Tokens that *characterise* a logical alias key.

    For ``curated-fever-neutropenia`` we want {"fever", "neutropenia"};
    "curated" is stripped because it's a project-internal prefix.
    """
    no_prefix = re.sub(r"^(curated|project|cbc|nci|cdc|acs|msk)[\-_]", "", alias_key)
    return _tokens(no_prefix)


def _kb_chunk_tokens(chunk: Mapping[str, Any]) -> set[str]:
    fields = (
        chunk.get("title"),
        chunk.get("source_name"),
        chunk.get("topic"),
        chunk.get("section"),
    )
    combined: set[str] = set()
    for f in fields:
        combined |= _tokens(str(f) if f else "")
    return combined


# ─── Coverage computation ───────────────────────────────────────────────


def _load_goldset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_kb_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("chunks") or []


def _goldset_alias_keys(goldset: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in goldset:
        for sid in case.get("expected_source_ids") or []:
            counts[str(sid).strip().lower()] += 1
    return dict(counts)


def _aliases_covering(alias_set: set[str], kb_chunk: Mapping[str, Any]) -> bool:
    """True iff the alias_set already contains an identifier matching this chunk."""
    candidates = {
        str(kb_chunk.get("id") or "").lower(),
        str(kb_chunk.get("parent_id") or "").lower(),
        str(kb_chunk.get("source_name") or "").lower(),
        str(kb_chunk.get("title") or "").lower(),
    }
    return bool(candidates & {a.lower() for a in alias_set})


def build_alias_coverage_report(
    *,
    goldset_path: Path = DEFAULT_GOLDSET_PATH,
    kb_path: Path = DEFAULT_KB_PATH,
    min_token_overlap: int = 2,
) -> dict[str, Any]:
    """Return a coverage report. See module docstring for shape."""

    # Local import keeps the diagnostic decoupled from the runtime
    # import surface of the baseline-comparison module.
    from backend.services.rag_baseline_comparison import LOGICAL_SOURCE_ALIASES

    goldset = _load_goldset(goldset_path)
    chunks = _load_kb_chunks(kb_path)
    alias_demand = _goldset_alias_keys(goldset)

    per_alias: list[dict[str, Any]] = []
    proposed_total = 0
    uncovered_alias_keys: list[str] = []

    for alias_key, demand_count in sorted(alias_demand.items(), key=lambda x: -x[1]):
        alias_set = LOGICAL_SOURCE_ALIASES.get(alias_key) or LOGICAL_SOURCE_ALIASES.get(alias_key.lower())
        if not alias_set:
            uncovered_alias_keys.append(alias_key)
            per_alias.append({
                "alias_key": alias_key,
                "goldset_demand_count": demand_count,
                "alias_set_size": 0,
                "kb_parent_ids_in_alias_set": [],
                "proposed_additions_by_content_match": [],
                "match_method": "none — alias key missing from LOGICAL_SOURCE_ALIASES",
            })
            continue

        key_tokens = _alias_key_tokens(alias_key)
        kb_parent_ids_in_set: set[str] = set()
        proposed: list[dict[str, Any]] = []
        for chunk in chunks:
            pid = str(chunk.get("parent_id") or "").lower()
            if not pid:
                continue
            if pid in {a.lower() for a in alias_set}:
                kb_parent_ids_in_set.add(pid)
                continue
            chunk_tokens = _kb_chunk_tokens(chunk)
            overlap = key_tokens & chunk_tokens
            if len(overlap) >= min_token_overlap:
                # De-duplicate proposed entries by parent_id.
                if not any(p["parent_id"] == pid for p in proposed):
                    proposed.append({
                        "parent_id": pid,
                        "title": (chunk.get("title") or "")[:120],
                        "source_name": chunk.get("source_name"),
                        "matched_tokens": sorted(overlap),
                    })

        per_alias.append({
            "alias_key": alias_key,
            "goldset_demand_count": demand_count,
            "alias_set_size": len(alias_set),
            "kb_parent_ids_in_alias_set": sorted(kb_parent_ids_in_set),
            "proposed_additions_by_content_match": proposed,
            "match_method": (
                f"intersect alias-key tokens with KB chunk title/source_name/topic, "
                f"min_token_overlap={min_token_overlap}"
            ),
        })
        proposed_total += len(proposed)

    status = (
        "needs_attention" if uncovered_alias_keys else
        "informational"
    )

    return {
        "schema_version": "1.0",
        "status": status,
        "label": "source_alias_coverage_diagnostic",
        "claim_boundary": (
            "Coverage diagnostic only.  Proposed additions are discovered by content "
            "matching against KB titles and source names; they are NOT auto-applied. "
            "Promotion into LOGICAL_SOURCE_ALIASES requires reviewer judgement and "
            "ADR-style documentation (see docs/adr/0009-source-alias-normalization.md). "
            "Engineering signal only; not a clinical claim."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_goldset_cases": len(goldset),
        "n_kb_chunks": len(chunks),
        "n_alias_keys_demanded_by_goldset": len(alias_demand),
        "n_alias_keys_uncovered": len(uncovered_alias_keys),
        "uncovered_alias_keys": uncovered_alias_keys,
        "n_proposed_additions_total": proposed_total,
        "per_alias": per_alias,
        "contamination_note": (
            "This diagnostic reads the frozen goldset. The diagnostic itself does NOT "
            "change retrieval ranking or eval scoring. If proposed aliases are "
            "promoted into LOGICAL_SOURCE_ALIASES, the next baseline comparison must "
            "explicitly report raw vs alias-normalized recall side by side and the "
            "promotion must be tied to the diagnostic run it was derived from."
        ),
    }


def write_alias_coverage_report(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    report = build_alias_coverage_report()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_GOLDSET_PATH",
    "DEFAULT_KB_PATH",
    "DEFAULT_OUTPUT_PATH",
    "build_alias_coverage_report",
    "write_alias_coverage_report",
]
