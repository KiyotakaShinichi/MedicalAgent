"""Query normalization, tokenization, and rewrite/decomposition.

The functions here used to live inline in ``agent_rag.py``.  They were
extracted as part of the agent_rag.py module split because they're
purely about *text*: no DB, no retrieval, no agent state.

Public symbols
~~~~~~~~~~~~~~
- ``normalize_query`` / ``_normalize_query`` — lower-case + punctuation strip
- ``tokenize``       / ``_tokenize``         — stop-word stripped token list
- ``semantic_key``   / ``_semantic_key``     — canonical cache key for a query+intent
- ``rewrite_and_decompose``                  — expansion + sub-query split

The underscore aliases are kept so existing in-module call sites in
``agent_rag.py`` (which use the underscore names ~15 times) continue to
resolve via the agent_rag re-import shim — no call-site rewrites
required for this incremental split.

Behavior is preserved verbatim from the original ``agent_rag.py`` block
including the SYNONYM_EXPANSIONS table and the sub-query splitter that
breaks on ``and``, ``?``, ``,``.
"""
from __future__ import annotations

import re
from typing import Any


# ─── Tables ──────────────────────────────────────────────────────────────────


# Stop words that ``tokenize`` drops.  Tuned for clinical/educational
# queries — kept *small* on purpose so genuine clinical terms (e.g.
# "fever", "pain") survive.
TOKENIZE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "the", "this", "to", "what", "when", "where",
    "why", "with", "you", "your",
})


# Synonym expansion table for the query-rewrite pass.  Each key is
# matched as a whole word against the normalized query; on match, the
# expansion is appended to the expanded query.  Order is irrelevant;
# every match is appended.
SYNONYM_EXPANSIONS: dict[str, str] = {
    "wbc": "white blood cells cbc infection neutropenia",
    "hgb": "hemoglobin anemia cbc",
    "hb":  "hemoglobin anemia cbc",
    "plt": "platelets cbc bleeding",
    "cbc": "complete blood count lab values blood count results",
    "mri": "imaging response breast mri",
    "ct":  "ct imaging report ascites peritoneal clinician review oncology monitoring metastasis wording",
    "ascites": "ct imaging report ascites peritoneal clinician review oncology monitoring metastasis wording",
    "pcr": "pathologic complete response treatment response classification",
    "chemo": "chemotherapy treatment side effects",
    "neutropenia": "neutropenia infection low white blood cells fever",
    "anemia": "anemia hemoglobin low red blood cells fatigue",
    "nadir": "nadir lowest point blood counts chemotherapy",
    "her2": "her2 receptor breast cancer subtype",
    "fever": "fever infection urgent chemotherapy",
}


# Sub-query splitter — same delimiters used by the original inline code.
_SUBQUERY_SPLITTER = re.compile(r"\band\b|\?|,")


# ─── Text helpers ────────────────────────────────────────────────────────────


def normalize_query(query: str) -> str:
    """Lower-case the input and collapse any character that isn't
    alphanumeric / whitespace / ``/`` / ``.`` / ``-`` into a single space.

    Keeps slashes / dots / dashes so terms like ``ki-67``, ``ca 15-3``,
    ``ac-t`` survive.
    """
    return " ".join(re.sub(r"[^a-z0-9\s/.-]", " ", query.lower()).split())


def tokenize(text: str) -> list[str]:
    """Split ``text`` into stop-word-stripped tokens of length ≥ 2."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in TOKENIZE_STOPWORDS and len(token) > 1
    ]


def semantic_key(expanded_query: str, intent: str) -> str:
    """Canonical cache key — sorted, deduped, lowercased tokens of
    ``intent + expanded_query``, capped at 40 tokens.  This is the key
    the semantic-cache lookup uses to find responses for *similar*
    queries that aren't byte-identical."""
    tokens = sorted(set(tokenize(f"{intent} {expanded_query}")))
    return " ".join(tokens[:40])


# Underscore aliases — kept for back-compat with agent_rag.py's internal
# call sites.  Identical to the public names.
_normalize_query = normalize_query
_tokenize = tokenize
_semantic_key = semantic_key


# ─── Rewrite + decomposition ─────────────────────────────────────────────────


def rewrite_and_decompose(query: str, intent: str) -> dict[str, Any]:
    """Expand a raw query and split it into sub-queries.

    Returns a dict with:
      - ``original_query``   — the input verbatim
      - ``normalized_query`` — output of :func:`normalize_query`
      - ``expanded_query``   — normalized query plus intent / synonym extensions
      - ``subqueries``       — up to 4 sub-queries split on ``and``, ``?``, ``,``
      - ``semantic_key``     — output of :func:`semantic_key`

    Behavior preserved from the original ``agent_rag.rewrite_and_decompose``:
    portal_help queries get a fixed bag of portal/upload terms appended
    before the synonym pass; every matching SYNONYM_EXPANSIONS key whose
    term appears as a *whole word* in the normalized query contributes
    its expansion.
    """
    normalized = normalize_query(query)
    expanded = normalized

    if intent == "portal_help":
        expanded = (
            f"{expanded} portal upload guide patient portal cbc labs blood count results "
            "symptoms medications mri imaging report"
        )

    normalized_words = normalized.split()
    for term, expansion in SYNONYM_EXPANSIONS.items():
        if term in normalized_words:
            expanded = f"{expanded} {expansion}"

    parts = [part.strip() for part in _SUBQUERY_SPLITTER.split(normalized) if part.strip()]
    if not parts:
        parts = [normalized]

    return {
        "original_query":   query,
        "normalized_query": normalized,
        "expanded_query":   expanded,
        "subqueries":       parts[:4],
        "semantic_key":     semantic_key(expanded, intent),
    }


__all__ = [
    "TOKENIZE_STOPWORDS",
    "SYNONYM_EXPANSIONS",
    "normalize_query",
    "tokenize",
    "semantic_key",
    "rewrite_and_decompose",
    # Underscore aliases retained for agent_rag.py compatibility.
    "_normalize_query",
    "_tokenize",
    "_semantic_key",
]
