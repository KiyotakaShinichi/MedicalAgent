"""Response cache for the patient agent.

Two lookup paths and one write path:

  - :func:`exact_cache_check`     — byte-exact hash match on the
    normalized query.
  - :func:`semantic_cache_check`  — Jaccard similarity on the cached
    rows' semantic_key for the same intent + low_risk + matching KB
    fingerprint.
  - :func:`store_cache`           — write/refresh a row, gated upstream
    by :func:`is_cacheable`.

A cache row is considered fresh only when (a) its cache_schema_version
matches the current ``AGENT_CACHE_SCHEMA_VERSION``, (b) its
``knowledge_fingerprint`` matches the live KB's, and (c) its
``expires_at`` is in the future.  Any mismatch causes a refresh —
expired rows are not deleted; they get rewritten on the next miss.

Extracted from ``agent_rag.py`` as part of the agent_rag.py module
split.  All public functions are re-exported from
``backend.services.agent_rag`` so existing imports keep working.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from backend.models import AgentResponseCache
from backend.services.local_llm import (
    configured_llm_providers,
    decide_cache_with_local_llm,
)


# ─── Constants ───────────────────────────────────────────────────────────────


AGENT_CACHE_TTL_DAYS: int = 30
AGENT_CACHE_SCHEMA_VERSION: str = "agent_response_cache_v4"
SEMANTIC_CACHE_MIN_SIMILARITY: float = 0.86


# Intents that the agent permits caching for.  Everything else is
# treated as patient-specific or uncertain.
_CACHEABLE_INTENTS: frozenset[str] = frozenset({
    "education",
    "portal_help",
    "general_support",
})


# Substrings that make a query patient-specific (and therefore not
# cacheable).  Padded by whitespace on both sides at match time so
# "myth" doesn't match "my".
_PATIENT_SPECIFIC_TERMS: tuple[str, ...] = (
    " my ", " me ", " i ",
    "latest", "my score", "my labs", "my mri", "my treatment",
)


# ─── Timestamp + JSON utilities (cache-adjacent) ─────────────────────────────


def _coerce_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_to_iso(value):
    value = _coerce_utc(value)
    return value.isoformat() if value else None


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _query_hash(normalized_query: str) -> str:
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


# ─── Cacheability decision ───────────────────────────────────────────────────


def is_cacheable(
    query: str,
    intent: str,
    safety: Mapping[str, Any],
    actions=None,
    urgent_flags=None,
) -> bool:
    """Decide whether the agent should write this response to the cache.

    The hard-no rules (actions, urgent flags, safety opt-out, non-cacheable
    intent, patient-specific wording) short-circuit before any LLM
    consultation.  When the deterministic check says "yes", the LLM is
    consulted and a high-confidence vote can flip the decision either way.
    """
    actions = actions or []
    urgent_flags = urgent_flags or []
    if actions or urgent_flags or not safety.get("cache_allowed"):
        return False
    if intent not in _CACHEABLE_INTENTS:
        return False
    padded = f" {query.lower()} "
    if any(term in padded for term in _PATIENT_SPECIFIC_TERMS):
        return False
    llm = decide_cache_with_local_llm(query, deterministic_cacheable=True, intent=intent, safety=safety)
    if llm.get("available") and float(llm.get("confidence") or 0) >= 0.72:
        return bool(llm.get("cacheable"))
    return True


def cache_rejection_reason(
    query: str,
    intent: str,
    safety: Mapping[str, Any],
    actions,
    urgent_flags,
) -> str:
    if actions:
        return "patient_specific_data_entry"
    if urgent_flags:
        return "urgent_query"
    if not safety.get("cache_allowed"):
        return safety.get("scope")
    if intent not in _CACHEABLE_INTENTS:
        return f"intent_not_cacheable:{intent}"
    return "patient_specific_or_uncertain"


# Underscore alias preserves the agent_rag internal call site.
_cache_rejection_reason = cache_rejection_reason


# ─── Lookups ─────────────────────────────────────────────────────────────────


def exact_cache_check(
    db,
    normalized_query: str,
    intent: str | None = None,
    safety_level: str | None = None,
    knowledge_fingerprint: str | None = None,
    now=None,
):
    """Exact-match cache lookup keyed by SHA-256 of the normalized query.

    Returns the cached envelope (with ``status="exact_cache_hit"``) on
    fresh match, or ``None`` on miss / freshness mismatch.
    """
    from backend.services.agent_rag import knowledge_base_fingerprint

    knowledge_fingerprint = knowledge_fingerprint or knowledge_base_fingerprint()
    query_hash = _query_hash(normalized_query)
    row = db.query(AgentResponseCache).filter(AgentResponseCache.query_hash == query_hash).first()
    if not row:
        return None
    if intent is not None and row.intent != intent:
        return None
    if safety_level is not None and row.safety_level != safety_level:
        return None
    freshness = _cache_row_freshness(row, knowledge_fingerprint, now=now)
    if freshness["status"] != "fresh":
        return None
    response = _json_loads(row.response_json)
    if response is None:
        return None
    _mark_cache_hit(db, row, now=now)
    return {
        "status": "exact_cache_hit",
        "cache_id": row.id,
        "response": response,
        "expires_at": _datetime_to_iso(row.expires_at),
        "knowledge_fingerprint": row.knowledge_fingerprint,
        "policy": _cache_row_policy(row),
    }


def semantic_cache_check(
    db,
    semantic_key: str,
    intent: str,
    min_similarity: float = SEMANTIC_CACHE_MIN_SIMILARITY,
    knowledge_fingerprint: str | None = None,
    now=None,
):
    """Jaccard-similarity lookup over cached rows matching ``intent`` +
    ``low_risk`` + the current KB fingerprint.  Returns the best fresh
    match above ``min_similarity`` or ``None``."""
    from backend.services.agent_rag import knowledge_base_fingerprint

    knowledge_fingerprint = knowledge_fingerprint or knowledge_base_fingerprint()
    query_tokens = set(semantic_key.split())
    if not query_tokens:
        return None
    rows = (
        db.query(AgentResponseCache)
        .filter(AgentResponseCache.intent == intent)
        .filter(AgentResponseCache.safety_level == "low_risk")
        .filter(AgentResponseCache.knowledge_fingerprint == knowledge_fingerprint)
        .all()
    )
    best: tuple[float, Any] | None = None
    for row in rows:
        freshness = _cache_row_freshness(row, knowledge_fingerprint, now=now)
        if freshness["status"] != "fresh":
            continue
        row_tokens = set((row.semantic_key or "").split())
        if not row_tokens:
            continue
        score = len(query_tokens & row_tokens) / len(query_tokens | row_tokens)
        if score >= min_similarity and (best is None or score > best[0]):
            best = (score, row)
    if best is None:
        return None
    row = best[1]
    response = _json_loads(row.response_json)
    if response is None:
        return None
    _mark_cache_hit(db, row, now=now)
    response["semantic_cache_similarity"] = round(best[0], 3)
    return {
        "status": "semantic_cache_hit",
        "cache_id": row.id,
        "response": response,
        "expires_at": _datetime_to_iso(row.expires_at),
        "knowledge_fingerprint": row.knowledge_fingerprint,
        "policy": _cache_row_policy(row),
    }


# ─── Write path ──────────────────────────────────────────────────────────────


def store_cache(
    db,
    rewritten: Mapping[str, Any],
    intent: str,
    safety: Mapping[str, Any],
    response: Mapping[str, Any],
    knowledge_fingerprint: str | None = None,
    now=None,
):
    """Write or refresh the cache row for ``rewritten["normalized_query"]``.

    Re-uses any existing row keyed by the same query hash so the
    response is updated in place; resets ``hit_count`` and ``last_hit_at``
    on refresh so cache-hit stats reflect post-refresh reads only.
    """
    from backend.services.agent_rag import knowledge_base_fingerprint

    now = now or datetime.now(timezone.utc)
    knowledge_fingerprint = knowledge_fingerprint or knowledge_base_fingerprint()
    query_hash = _query_hash(rewritten["normalized_query"])
    row = db.query(AgentResponseCache).filter(AgentResponseCache.query_hash == query_hash).first()
    if row is None:
        row = AgentResponseCache(query_hash=query_hash)
        db.add(row)
    else:
        row.hit_count = 0
        row.last_hit_at = None

    row.semantic_key = rewritten["semantic_key"]
    row.intent = intent
    row.safety_level = safety["level"]
    row.normalized_query = rewritten["normalized_query"]
    row.response_json = json.dumps(_cache_response_payload(response), default=str)
    row.source_ids_json = json.dumps([item["id"] for item in response.get("citations") or []])
    row.knowledge_fingerprint = knowledge_fingerprint
    row.cache_schema_version = AGENT_CACHE_SCHEMA_VERSION
    row.cache_policy_json = json.dumps(_cache_policy_snapshot(knowledge_fingerprint), default=str)
    row.expires_at = now + timedelta(days=AGENT_CACHE_TTL_DAYS)
    row.updated_at = now
    db.commit()
    db.refresh(row)
    return row


def _cache_response_payload(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reply":             response.get("reply"),
        "citations":         response.get("citations") or [],
        "intent":            response.get("intent"),
        "safety":            response.get("safety"),
        "retrieval_context": response.get("retrieval_context") or [],
        "generated_at":      response.get("generated_at"),
        "safety_note":       response.get("safety_note"),
        "validation":        response.get("validation"),
    }


def _cache_policy_snapshot(knowledge_fingerprint: str | None) -> dict[str, Any]:
    return {
        "schema_version":          AGENT_CACHE_SCHEMA_VERSION,
        "ttl_days":                AGENT_CACHE_TTL_DAYS,
        "semantic_min_similarity": SEMANTIC_CACHE_MIN_SIMILARITY,
        "knowledge_fingerprint":   knowledge_fingerprint,
        "reuse_scope":             "low_risk_non_patient_specific_agent_answers",
        "llm_cache_adjudication":  configured_llm_providers(),
        "invalidates_on":          ["ttl_expiry", "knowledge_base_fingerprint_change", "safety_policy_rejection"],
    }


# ─── Freshness + bookkeeping ─────────────────────────────────────────────────


def _cache_row_freshness(row, knowledge_fingerprint: str | None, now=None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    expires_at = _coerce_utc(row.expires_at)
    if row.cache_schema_version != AGENT_CACHE_SCHEMA_VERSION:
        reasons.append("cache_schema_version_changed")
    if not row.knowledge_fingerprint:
        reasons.append("missing_knowledge_fingerprint")
    elif row.knowledge_fingerprint != knowledge_fingerprint:
        reasons.append("knowledge_base_fingerprint_changed")
    if expires_at is None:
        reasons.append("missing_expiry")
    elif expires_at <= now:
        reasons.append("expired")
    return {
        "status": "fresh" if not reasons else "stale",
        "reasons": reasons,
    }


def _cache_row_policy(row) -> dict[str, Any]:
    policy = _json_loads(row.cache_policy_json) or {}
    if not policy:
        policy = _cache_policy_snapshot(row.knowledge_fingerprint)
    return {
        **policy,
        "expires_at":  _datetime_to_iso(row.expires_at),
        "last_hit_at": _datetime_to_iso(row.last_hit_at),
        "hit_count":   int(row.hit_count or 0),
    }


def _mark_cache_hit(db, row, now=None) -> None:
    now = now or datetime.now(timezone.utc)
    row.hit_count = int(row.hit_count or 0) + 1
    row.last_hit_at = now
    row.updated_at = now
    db.commit()
    db.refresh(row)


__all__ = [
    "AGENT_CACHE_TTL_DAYS",
    "AGENT_CACHE_SCHEMA_VERSION",
    "SEMANTIC_CACHE_MIN_SIMILARITY",
    "exact_cache_check",
    "semantic_cache_check",
    "is_cacheable",
    "store_cache",
    "cache_rejection_reason",
    "_cache_rejection_reason",
    # Cache-adjacent utilities (re-exported by agent_rag for backward compat).
    "_query_hash",
    "_coerce_utc",
    "_datetime_to_iso",
    "_json_loads",
    # Internal helpers some callers reach into.
    "_cache_row_freshness",
    "_cache_row_policy",
    "_cache_response_payload",
    "_cache_policy_snapshot",
    "_mark_cache_hit",
]
