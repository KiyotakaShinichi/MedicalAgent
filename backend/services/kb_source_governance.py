"""KB source-governance layer on top of the RAG knowledge base.

The existing `rag_source_registry` exposes the RAW metadata each KB chunk
carries (trust_level, topic, etc.).  This service adds the *engineering
contract* layer on top:

  - Each ``trust_level`` is mapped to a numbered **tier** (T1 highest =
    clinical guidelines + safety policy, T5 lowest = uncategorised / do
    not cite).
  - Each source carries an explicit ``allowed_use`` set so callers know
    which kinds of answers a source can back (e.g. tumor-marker chunks are
    ``monitoring_context_only`` — never standalone diagnosis).
  - ``staleness_status`` is computed from ``ingested_at`` against a TTL so
    a reviewer can see at-a-glance which sources need refresh.

Output artifact: ``Data/evals/rag/latest_kb_source_governance.json``

This is engineering provenance.  A passing governance report does not mean
every source is clinically authoritative — it means the *mapping* between
trust_level / tier / allowed_use is consistent and traceable.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_KB_CHUNKS_PATH = "Data/rag_knowledge_base_chunks.json"
DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_kb_source_governance.json"

# Staleness TTL: chunks ingested more than this many days ago are flagged
# `needs_review`.  365 days is the default checkpoint for patient-facing
# oncology content — clinical advisors usually re-validate yearly.
STALENESS_TTL_DAYS = 365


# ─── Tier + allowed_use mapping ──────────────────────────────────────────────
# This is the single source of truth for how trust_level translates into a
# numbered tier and the kinds of answers a source can back.  Anything not in
# the map falls to T5 / no allowed_use, which forces the validator to refuse
# to cite the source.

TIER_MAP: dict[str, dict[str, Any]] = {
    "clinical_safety_policy": {
        "tier": "T1",
        "rank": 1,
        "description": "Internal clinical-safety policy (refusal rules, escalation rules).",
        "allowed_use": ("patient_safety", "education", "clinician_only"),
    },
    "clinical_guideline_summary": {
        "tier": "T1",
        "rank": 1,
        "description": "Summary of a published clinical guideline (NCCN, ASCO, ESMO equivalent).",
        "allowed_use": ("education", "clinician_only", "monitoring_context"),
    },
    "systematic_review": {
        "tier": "T2",
        "rank": 2,
        "description": "Peer-reviewed systematic review / meta-analysis.",
        "allowed_use": ("education", "monitoring_context"),
    },
    "research_paper": {
        "tier": "T2",
        "rank": 2,
        "description": "Individual peer-reviewed primary research paper.",
        "allowed_use": ("education", "monitoring_context"),
    },
    "patient_education": {
        "tier": "T3",
        "rank": 3,
        "description": "Patient-education content from a recognised oncology organisation.",
        "allowed_use": ("education",),
    },
    "local_source": {
        "tier": "T4",
        "rank": 4,
        "description": "Internal portal documentation or tool-help text.",
        "allowed_use": ("portal_help",),
    },
}

# Tiers ordered from most authoritative to least.
TIER_ORDER = ("T1", "T2", "T3", "T4", "T5")

ALLOWED_USE_VOCABULARY = (
    "patient_safety",
    "education",
    "monitoring_context",
    "portal_help",
    "clinician_only",
)


# ─── Public API ──────────────────────────────────────────────────────────────


def build_kb_source_governance(
    kb_chunks_path: str = DEFAULT_KB_CHUNKS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    *,
    staleness_ttl_days: int = STALENESS_TTL_DAYS,
) -> dict[str, Any]:
    """Walk every chunk in the KB, group by source, attach tier +
    allowed_use + staleness, write the artifact, and return the payload."""
    path = Path(kb_chunks_path)
    if not path.exists():
        return _missing_payload(kb_chunks_path)

    try:
        kb_payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _error_payload(kb_chunks_path, str(exc))

    chunks = kb_payload.get("chunks") or []
    sources_grouped = defaultdict(list)
    for chunk in chunks:
        key = chunk.get("parent_id") or chunk.get("source_name") or chunk.get("source_path") or chunk.get("id")
        sources_grouped[key].append(chunk)

    source_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for source_id, source_chunks in sources_grouped.items():
        source_rows.append(_build_source_row(source_id, source_chunks, now, staleness_ttl_days))

    source_rows.sort(key=lambda r: (r["tier_rank"], -r["chunk_count"]))

    # Tier + allowed_use distributions across the whole KB.
    tier_counts = Counter(r["tier"] for r in source_rows)
    use_counts: Counter[str] = Counter()
    for r in source_rows:
        for use in r["allowed_use"]:
            use_counts[use] += 1
    staleness_counts = Counter(r["staleness_status"] for r in source_rows)

    governance_issues = _detect_governance_issues(source_rows, kb_payload)
    status = _overall_status(tier_counts, governance_issues)

    payload = {
        "schema_version": "kb_source_governance_v1",
        "generated_at": now.isoformat(),
        "status": status,
        "kb_chunks_path": kb_chunks_path,
        "source_count": len(source_rows),
        "chunk_count": len(chunks),
        "tier_distribution":      dict(tier_counts),
        "allowed_use_distribution": dict(use_counts),
        "staleness_distribution": dict(staleness_counts),
        "tier_map": {
            level: {
                "tier": meta["tier"],
                "description": meta["description"],
                "allowed_use": list(meta["allowed_use"]),
            }
            for level, meta in TIER_MAP.items()
        },
        "tier_order": list(TIER_ORDER),
        "allowed_use_vocabulary": list(ALLOWED_USE_VOCABULARY),
        "staleness_ttl_days": staleness_ttl_days,
        "sources": source_rows,
        "governance_issues": governance_issues,
        "interpretation": (
            "T1/T2 sources can back clinician-facing claims when allowed_use "
            "includes the right intent. T3 is patient-education only. T4 is "
            "internal portal docs. T5 (unmapped trust_level) is do-not-cite — "
            "the post-generation validator must strip those citations."
        ),
        "claim_boundary": (
            "Engineering provenance only. Tier mapping is a defensible "
            "default; clinical advisor sign-off is required before treating "
            "T1/T2 chunks as authoritative for any specific clinical claim."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_kb_source_governance(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "kb_source_governance_v1",
            "status": "missing",
            "message": (
                "KB source governance has not been generated yet. Run "
                "`scripts/run_kb_source_governance.py` or POST to "
                "/admin/kb-source-governance."
            ),
            "sources": [],
            "tier_distribution": {},
            "allowed_use_distribution": {},
            "staleness_distribution": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


# ─── Per-source helpers ──────────────────────────────────────────────────────


def _build_source_row(
    source_id: str,
    chunks: list[dict[str, Any]],
    now: datetime,
    ttl_days: int,
) -> dict[str, Any]:
    head = chunks[0]
    trust_level = head.get("trust_level") or "unknown"
    tier_meta = TIER_MAP.get(trust_level) or {
        "tier": "T5",
        "rank": 5,
        "description": f"Uncategorised trust_level '{trust_level}'.",
        "allowed_use": tuple(),
    }
    ingested_at = _parse_ts(head.get("ingested_at"))
    staleness_status, staleness_days = _staleness(ingested_at, now, ttl_days)

    return {
        "source_id": source_id,
        "title": head.get("title") or head.get("source_name") or "Untitled source",
        "source_url": head.get("source_url"),
        "source_path": head.get("source_path"),
        "trust_level": trust_level,
        "tier": tier_meta["tier"],
        "tier_rank": tier_meta["rank"],
        "tier_description": tier_meta["description"],
        "allowed_use": list(tier_meta["allowed_use"]),
        "ingested_at": head.get("ingested_at"),
        "staleness_status": staleness_status,
        "staleness_days": staleness_days,
        "chunk_count": len(chunks),
        "topics": sorted({c.get("topic") for c in chunks if c.get("topic")}),
        "modalities": sorted({
            modality
            for c in chunks
            for modality in (c.get("modality") or [])
        }),
    }


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _staleness(ingested: datetime | None, now: datetime, ttl_days: int) -> tuple[str, int | None]:
    if ingested is None:
        return ("unknown", None)
    delta_days = (now - ingested).days
    if delta_days <= ttl_days // 2:
        return ("current", delta_days)
    if delta_days <= ttl_days:
        return ("aging", delta_days)
    return ("needs_review", delta_days)


def _detect_governance_issues(
    sources: list[dict[str, Any]],
    kb_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Surface obvious mapping gaps a reviewer should fix."""
    issues: list[dict[str, Any]] = []
    t5_sources = [s for s in sources if s["tier"] == "T5"]
    if t5_sources:
        issues.append({
            "severity": "high",
            "code": "uncategorised_trust_level",
            "message": (
                f"{len(t5_sources)} source(s) have an unmapped trust_level "
                "and would be blocked from citation by the post-gen validator."
            ),
            "examples": [s["source_id"] for s in t5_sources[:5]],
        })
    needs_review = [s for s in sources if s["staleness_status"] == "needs_review"]
    if needs_review:
        issues.append({
            "severity": "medium",
            "code": "needs_review",
            "message": (
                f"{len(needs_review)} source(s) exceed the staleness TTL "
                "and should be re-validated by a clinical advisor."
            ),
            "examples": [s["source_id"] for s in needs_review[:5]],
        })
    sources_with_no_use = [s for s in sources if not s["allowed_use"]]
    if sources_with_no_use:
        issues.append({
            "severity": "medium",
            "code": "no_allowed_use",
            "message": (
                f"{len(sources_with_no_use)} source(s) carry no allowed_use "
                "after tier mapping — they cannot back any claim until "
                "trust_level is fixed."
            ),
            "examples": [s["source_id"] for s in sources_with_no_use[:5]],
        })
    return issues


def _overall_status(
    tier_counts: Counter[str],
    issues: list[dict[str, Any]],
) -> str:
    if any(i["severity"] == "high" for i in issues):
        return "needs_attention"
    # We want at least some T1 + T2 sources to call governance "strong".
    if tier_counts.get("T1", 0) > 0 and tier_counts.get("T2", 0) > 0 and not issues:
        return "strong"
    if not issues:
        return "acceptable"
    return "acceptable"


def _missing_payload(kb_chunks_path: str) -> dict[str, Any]:
    return {
        "schema_version": "kb_source_governance_v1",
        "status": "missing",
        "message": f"KB chunks file not found at {kb_chunks_path}.",
        "sources": [],
        "tier_distribution": {},
        "allowed_use_distribution": {},
        "staleness_distribution": {},
    }


def _error_payload(kb_chunks_path: str, error: str) -> dict[str, Any]:
    return {
        "schema_version": "kb_source_governance_v1",
        "status": "error",
        "kb_chunks_path": kb_chunks_path,
        "error": error,
        "sources": [],
        "tier_distribution": {},
        "allowed_use_distribution": {},
        "staleness_distribution": {},
    }


__all__ = [
    "ALLOWED_USE_VOCABULARY",
    "STALENESS_TTL_DAYS",
    "TIER_MAP",
    "TIER_ORDER",
    "build_kb_source_governance",
    "load_kb_source_governance",
]
