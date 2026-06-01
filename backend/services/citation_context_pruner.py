"""Citation-context pruner.

Sits between source-governed retrieval and citation assembly. Takes
the already-ranked, already-governance-filtered top-K chunks and
prunes them down to a smaller, higher-citation-precision set without
weakening source governance.

Design rules (from the brief)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Generic**: the scorer reads only chunk metadata that would exist
   at runtime (id/parent_id/source_tier/topic/section/etc.), the
   query string, the rewritten query string, and the intent. It does
   NOT inspect goldset case_ids, expected_source_ids, or anything
   from the eval scaffolding.
2. **Preserves metadata**: every kept chunk is forwarded unchanged.
3. **Refusal-source retention**: when the intent is a refusal/safety
   route, the top-ranked patient-facing safety / boundary source is
   protected from pruning even if its lexical score is mediocre —
   refusal answers without a boundary citation are worse than refusal
   answers with one.
4. **Penalises**: near-duplicate siblings after the first strong hit,
   broad safety-policy chunks when the intent is education, chunks
   with low query overlap AND low retrieval score, chunks tagged
   clinician-only / disallowed_use (defensive — these should already
   be removed by the tier filter).
5. **No bypass of post-gen / claim validators**: this layer prunes the
   *input* to citation assembly. It does NOT change `validate_claims`
   or `apply_post_gen_validator`.

Public API
~~~~~~~~~~
    prune(
        chunks: Sequence[Mapping[str, Any]],
        *,
        query: str,
        rewritten_query: str | None = None,
        intent: str | None = None,
        keep: int = 5,
        refusal_route: bool | None = None,
    ) -> list[dict[str, Any]]

Returns the kept chunks in their pruned order. Length is at most
``keep``; may be shorter when the input has fewer than ``keep`` rows.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


# Intents that should always retain at least one boundary/policy source
# at the top of the cited context. These match the goldset's intent
# vocabulary and the live agent's refusal route names — neither set
# was introduced by this module.
_REFUSAL_INTENTS: frozenset[str] = frozenset({
    "urgent_escalation",
    "genetic_counselor_review",
    "tumor_marker_boundary",
    "pharmacist_or_clinician_review",
    "treatment_refusal",
    "prognosis_refusal",
    "diagnosis_refusal",
    "privacy_refusal",
    "safety_routing",
    "refuse_due_to_safety",
})

# Tokens that mark a chunk as a "boundary / safety policy" source. This
# is a *metadata signal*, not a content sniff: it triggers off the
# chunk's title / source_name / topic.
_BOUNDARY_HINTS: tuple[str, ...] = (
    "boundary", "policy", "safety", "refusal", "review",
    "minimum evidence", "consent", "escalation",
)

# Topic/intent compatibility map. Keys are normalized intent strings;
# values are tokens that, when present in chunk title/topic/section,
# are taken as compatible. This is a small generic helper, not a goldset
# patch — every value here is a generally-applicable medical/portal
# topic word.
_INTENT_TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "education": ("education", "patient", "general", "explain", "learn"),
    "urgent_escalation": ("fever", "neutropenia", "infection", "urgent", "emergency", "bleeding"),
    "genetic_counselor_review": ("genetic", "counseling", "germline", "brca", "panel", "vus"),
    "tumor_marker_boundary": ("tumor marker", "ca 15", "ca15", "ca 27", "ca27", "cea", "biomarker"),
    "pharmacist_or_clinician_review": ("supplement", "interaction", "pharmacist", "medication"),
    "treatment_refusal": ("treatment", "chemotherapy", "regimen", "dose"),
    "diagnosis_refusal": ("diagnosis", "boundary", "claim"),
    "prognosis_refusal": ("prognosis", "survival", "outcome", "boundary"),
    "privacy_refusal": ("privacy", "boundary", "policy"),
    "portal_help": ("portal", "upload", "entry", "workflow", "help"),
    "record_explanation": ("record", "log", "history", "timeline"),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "is", "are",
    "be", "been", "for", "with", "as", "at", "by", "from", "this", "that",
    "it", "what", "does", "do", "i", "my", "me", "we", "you", "should",
    "can", "may", "during", "about", "into", "while", "if", "then", "so",
    "such", "have", "has", "had", "but", "also", "after", "before", "than",
    "very",
})


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 2
    }


def _chunk_text_tokens(chunk: Mapping[str, Any]) -> set[str]:
    fields = (
        chunk.get("title"),
        chunk.get("source_name"),
        chunk.get("topic"),
        chunk.get("section"),
        # Text content if it's reasonably bounded; otherwise skip to
        # keep the scorer fast.
        chunk.get("text") if isinstance(chunk.get("text"), str) and len(chunk.get("text") or "") < 2000 else None,
    )
    out: set[str] = set()
    for f in fields:
        if f:
            out |= _tokens(str(f))
    return out


def _is_boundary_source(chunk: Mapping[str, Any]) -> bool:
    blob = " ".join(
        str(chunk.get(k) or "").lower()
        for k in ("title", "source_name", "topic", "section")
    )
    return any(hint in blob for hint in _BOUNDARY_HINTS)


def _is_clinician_only(chunk: Mapping[str, Any]) -> bool:
    use = str(chunk.get("allowed_use") or chunk.get("audience") or "").lower()
    return "clinician_only" in use or "disallowed" in use


def _is_stale(chunk: Mapping[str, Any]) -> bool:
    staleness = str(chunk.get("staleness_status") or "").lower()
    return "stale" in staleness or "expired" in staleness


def _retrieval_score(chunk: Mapping[str, Any]) -> float:
    for key in ("retrieval_score", "rerank_score", "fusion_score",
                "rrf_score", "score", "dense_score"):
        v = chunk.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _normalize_scores(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    raw = [_retrieval_score(r) for r in rows]
    if not raw:
        return []
    lo, hi = min(raw), max(raw)
    span = hi - lo
    if span <= 0:
        return [1.0 if v > 0 else 0.0 for v in raw]
    return [(v - lo) / span for v in raw]


def _parent_key(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("parent_id") or chunk.get("source_name") or chunk.get("id") or "")


def _composite_score(
    *,
    chunk: Mapping[str, Any],
    norm_retrieval: float,
    query_toks: set[str],
    rewrite_toks: set[str],
    intent_hint_toks: set[str],
    refusal_route: bool,
) -> tuple[float, dict[str, float]]:
    """Return (score, components) — components surfaced for diagnostics."""
    if _is_clinician_only(chunk):
        return -1.0, {"reason": -1.0}

    chunk_toks = _chunk_text_tokens(chunk)
    query_overlap = (
        len(query_toks & chunk_toks) / max(len(query_toks), 1)
        if query_toks else 0.0
    )
    rewrite_overlap = (
        len(rewrite_toks & chunk_toks) / max(len(rewrite_toks), 1)
        if rewrite_toks else query_overlap
    )
    intent_overlap = (
        len(intent_hint_toks & chunk_toks) / max(len(intent_hint_toks), 1)
        if intent_hint_toks else 0.0
    )

    # Tier bonus: T1/T2 > T3 > everything else.
    tier_raw = str(chunk.get("source_tier") or chunk.get("tier") or "").upper()
    tier_bonus = 0.10 if tier_raw in {"T1", "T2"} else (0.05 if tier_raw == "T3" else 0.0)

    boundary = _is_boundary_source(chunk)
    boundary_bonus = 0.0
    if boundary and refusal_route:
        boundary_bonus = 0.20
    elif boundary and not refusal_route:
        # Broad safety-policy chunks in education routes get penalised.
        boundary_bonus = -0.10

    staleness_penalty = -0.15 if _is_stale(chunk) else 0.0

    # If retrieval is weak AND query overlap is weak, the chunk
    # probably isn't supporting any concrete claim.  Apply a small
    # additional penalty so it loses to a higher-overlap chunk even
    # if its raw retrieval score is comparable.
    weak_signal_penalty = -0.10 if (norm_retrieval < 0.25 and query_overlap < 0.05) else 0.0

    score = (
        0.45 * norm_retrieval
        + 0.25 * query_overlap
        + 0.10 * rewrite_overlap
        + 0.10 * intent_overlap
        + tier_bonus
        + boundary_bonus
        + staleness_penalty
        + weak_signal_penalty
    )
    return (
        score,
        {
            "norm_retrieval": round(norm_retrieval, 4),
            "query_overlap": round(query_overlap, 4),
            "rewrite_overlap": round(rewrite_overlap, 4),
            "intent_overlap": round(intent_overlap, 4),
            "tier_bonus": tier_bonus,
            "boundary_bonus": boundary_bonus,
            "staleness_penalty": staleness_penalty,
            "weak_signal_penalty": weak_signal_penalty,
        },
    )


def prune(
    chunks: Sequence[Mapping[str, Any]],
    *,
    query: str,
    rewritten_query: str | None = None,
    intent: str | None = None,
    keep: int = 5,
    refusal_route: bool | None = None,
) -> list[dict[str, Any]]:
    """Prune a ranked list of chunks down to ``keep`` rows.

    The first chunk per parent_id is preferred; later chunks from the
    same parent are kept only if they add *new* lexical coverage and
    the keep budget allows it.  Refusal/safety routes preserve at
    least one boundary source even if its composite score is mediocre.
    """
    if not chunks:
        return []

    intent_key = (intent or "").lower()
    if refusal_route is None:
        refusal_route = intent_key in _REFUSAL_INTENTS

    query_toks = _tokens(query)
    rewrite_toks = _tokens(rewritten_query) if rewritten_query else query_toks
    intent_hint_toks = _tokens(" ".join(_INTENT_TOPIC_HINTS.get(intent_key, ())))

    norms = _normalize_scores(chunks)
    enriched: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (chunk, norm) in enumerate(zip(chunks, norms)):
        score, _components = _composite_score(
            chunk=chunk,
            norm_retrieval=norm,
            query_toks=query_toks,
            rewrite_toks=rewrite_toks,
            intent_hint_toks=intent_hint_toks,
            refusal_route=refusal_route,
        )
        # Stash the original input order so ties resolve deterministically.
        enriched.append((score, idx, dict(chunk)))

    enriched.sort(key=lambda t: (-t[0], t[1]))

    kept: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    covered_tokens: set[str] = set()
    boundary_picked = False

    # First pass: take top scorers, one-per-parent, with marginal-coverage
    # gating once we have at least one strong hit.
    for score, _, chunk in enriched:
        if len(kept) >= keep:
            break
        if score <= 0 and len(kept) >= 1:
            # Negative composite means the chunk is actively bad
            # (clinician-only sentinel, weak signal in education,
            # stale).  Stop once we have at least one hit.
            break
        parent = _parent_key(chunk)
        if parent and parent in seen_parents:
            # Sibling pruning: same parent, only accept if it adds new
            # tokens AND keep budget is at least half consumed.
            chunk_toks = _chunk_text_tokens(chunk)
            new = chunk_toks - covered_tokens
            if len(new) < 3 or len(kept) < keep // 2:
                continue
        kept.append(chunk)
        if parent:
            seen_parents.add(parent)
        covered_tokens |= _chunk_text_tokens(chunk)
        if _is_boundary_source(chunk):
            boundary_picked = True

    # Refusal-source retention: if the route is a refusal/safety route
    # and no boundary source survived pruning, scan the original ranked
    # list for the highest-ranked patient-facing boundary chunk and
    # insert it at position 0, displacing the weakest kept chunk.
    if refusal_route and not boundary_picked:
        for chunk in chunks:
            if (
                _is_boundary_source(chunk)
                and not _is_clinician_only(chunk)
                and not _is_stale(chunk)
            ):
                if kept and len(kept) >= keep:
                    kept.pop()  # drop weakest
                kept.insert(0, dict(chunk))
                break

    return kept


__all__ = ["prune"]
