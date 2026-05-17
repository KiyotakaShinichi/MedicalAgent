"""Claim-level citation validation for RAG replies.

Splits a generated reply into sentences, marks which sentences carry a
*medical claim* (vs. framing / conversational filler), and checks each
claim against the retrieved source chunks for textual support.

What "support" means here
-------------------------
For Phase 11 we use **token-overlap** (n-gram intersection) between the
claim sentence and each retrieved chunk's text.  This is a defensible
heuristic — not entailment — but it catches the majority of cases the
validator is meant to defend against:

  - the LLM fabricated a number or threshold that doesn't appear in any
    retrieved chunk → no overlap → unsupported,
  - the LLM cited a chunk that is on a different topic → low overlap →
    weakly_supported,
  - the LLM paraphrased a chunk → high overlap → supported.

A rigorous entailment check (NLI model, semantic similarity) is a
separate scope; the heuristic ships now so the validator is part of the
RAG flow today, and the bar can be raised later without changing the
contract.

Output: a `ClaimVerdict` per sentence, plus an aggregate `summary` with
counts and rates.  The caller (typically `_finalize_result` in
agent_rag) is responsible for acting on the verdicts — usually by
redacting unsupported claims and replacing the reply with a safe
deferral when too many claims are unsupported.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping


# Sentence splitter — kept simple on purpose.  Avoids heavy NLP deps;
# the regex catches ". ", "? ", "! " and end-of-string variants.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# Phrase fragments that mark a sentence as carrying a medical CLAIM
# (not framing / acknowledgement).  Wide on purpose — false positives
# (flagging non-claims) cost less than false negatives (missing a real
# claim and not checking it).
_CLAIM_SIGNALS = (
    "wbc", "anc", "hemoglobin", "platelets", "neutropenia", "neutrophil",
    "tumor", "metastat", "lesion", "biopsy", "biomarker", "her2", "er+",
    "pr+", "brca", "ki-67", "ki67", "tumor marker", "ca 15-3", "ca 27", "cea",
    "chemo", "chemotherapy", "doxorubicin", "paclitaxel", "trastuzumab",
    "tamoxifen", "cyclophosphamide", "carboplatin", "docetaxel",
    "mg", "milligram", "mri", "ct ", "ultrasound", "mammogram",
    "stage i", "stage ii", "stage iii", "stage iv",
    "diagnosis", "prognosis", "survival", "recurrence", "remission",
    "side effect", "side-effect", "interaction", "contraindicated",
    "recommend", "should take", "dose", "dosage", "increase", "decrease",
    "percent", "%", " 5 ", " 10 ", " 20 ", " 50 ",
)


# Stop tokens removed before token-overlap scoring — keeps the overlap
# from being dominated by "the / and / of / for".
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "for", "of", "in", "on", "to", "from", "by", "with",
    "as", "at", "this", "that", "these", "those", "it", "its", "their",
    "your", "you", "we", "our", "i", "me", "my", "they", "them", "he",
    "she", "his", "her", "not", "no", "yes", "can", "could", "should",
    "would", "may", "might", "will", "would", "would", "than", "such",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "again", "further", "more", "most", "other",
    "some", "any", "all", "each", "few", "many", "much",
})


@dataclass
class ClaimVerdict:
    """One sentence-level result from the validator."""

    sentence: str
    is_claim: bool
    support_score: float            # 0.0–1.0 (best overlap across chunks)
    status: str                     # "supported" | "weakly_supported" | "unsupported" | "non_claim"
    supporting_chunk_ids: list[str] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "is_claim": self.is_claim,
            "support_score": round(self.support_score, 4),
            "status": self.status,
            "supporting_chunk_ids": list(self.supporting_chunk_ids),
            "reason": self.reason,
        }


@dataclass
class ClaimValidationResult:
    """Aggregate validator output for one reply."""

    verdicts: list[ClaimVerdict] = field(default_factory=list)
    claim_count: int = 0
    supported_count: int = 0
    weakly_supported_count: int = 0
    unsupported_count: int = 0
    claim_support_rate: float = 0.0       # supported / claim_count
    citation_status: str = "complete"     # "complete" | "partial" | "missing" | "unsupported"

    def to_dict(self) -> dict:
        return {
            "claim_count": self.claim_count,
            "supported_count": self.supported_count,
            "weakly_supported_count": self.weakly_supported_count,
            "unsupported_count": self.unsupported_count,
            "claim_support_rate": round(self.claim_support_rate, 4),
            "citation_status": self.citation_status,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# Thresholds.  Tuned conservatively — the bar to be called "supported"
# is higher than to be called "weakly_supported", and unsupported below.
SUPPORTED_THRESHOLD = 0.30
WEAKLY_SUPPORTED_THRESHOLD = 0.12


def validate_claims(
    reply: str,
    retrieved_chunks: Iterable[Mapping[str, object]],
) -> ClaimValidationResult:
    """Run the full pipeline: sentence-split → claim detection → support
    check → aggregate.

    Robust to empty input.  When ``reply`` is empty, returns a result
    with zero claims and ``citation_status="missing"``.
    """
    result = ClaimValidationResult()
    if not reply or not reply.strip():
        result.citation_status = "missing"
        return result

    chunks = [c for c in retrieved_chunks if c and isinstance(c, Mapping)]
    chunk_tokens = [(_chunk_id(c), _tokens(str(c.get("text") or ""))) for c in chunks]

    sentences = _split_sentences(reply)
    for sentence in sentences:
        verdict = _evaluate_sentence(sentence, chunk_tokens)
        result.verdicts.append(verdict)

    # Aggregate.
    claims = [v for v in result.verdicts if v.is_claim]
    result.claim_count = len(claims)
    counts = Counter(v.status for v in claims)
    result.supported_count = counts.get("supported", 0)
    result.weakly_supported_count = counts.get("weakly_supported", 0)
    result.unsupported_count = counts.get("unsupported", 0)
    if result.claim_count:
        result.claim_support_rate = result.supported_count / result.claim_count
    else:
        result.claim_support_rate = 1.0  # vacuously true — no claims to support

    result.citation_status = _aggregate_citation_status(result)
    return result


# ─── Internal helpers ────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    raw = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t and t not in _STOPWORDS and len(t) > 1
    }


def _chunk_id(chunk: Mapping[str, object]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or chunk.get("parent_id") or "")


def _is_claim_sentence(sentence: str) -> bool:
    """Heuristic: a sentence carries a claim when it mentions a clinical
    term, a number, a drug, a recommendation verb, or a percentage."""
    lower = sentence.lower()
    if any(signal in lower for signal in _CLAIM_SIGNALS):
        return True
    # Numeric + unit phrase often indicates a quantitative claim even
    # when no recognised drug/term is present.
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|%|percent|cycle|cycles|day|days|week|weeks)\b", lower):
        return True
    return False


def _evaluate_sentence(
    sentence: str,
    chunk_tokens: list[tuple[str, set[str]]],
) -> ClaimVerdict:
    is_claim = _is_claim_sentence(sentence)
    if not is_claim:
        return ClaimVerdict(
            sentence=sentence,
            is_claim=False,
            support_score=0.0,
            status="non_claim",
        )

    sentence_toks = _tokens(sentence)
    if not sentence_toks:
        return ClaimVerdict(
            sentence=sentence,
            is_claim=True,
            support_score=0.0,
            status="unsupported",
            reason="no_substantive_tokens",
        )

    best_score = 0.0
    supporting: list[str] = []
    for chunk_id, chunk_toks in chunk_tokens:
        if not chunk_toks:
            continue
        overlap = len(sentence_toks & chunk_toks) / max(1, len(sentence_toks))
        if overlap > best_score:
            best_score = overlap
        if overlap >= WEAKLY_SUPPORTED_THRESHOLD:
            supporting.append(chunk_id)

    status = (
        "supported" if best_score >= SUPPORTED_THRESHOLD
        else "weakly_supported" if best_score >= WEAKLY_SUPPORTED_THRESHOLD
        else "unsupported"
    )
    return ClaimVerdict(
        sentence=sentence,
        is_claim=True,
        support_score=best_score,
        status=status,
        supporting_chunk_ids=supporting,
        reason=None if status != "unsupported" else "no_chunk_above_weak_support_threshold",
    )


def _aggregate_citation_status(result: ClaimValidationResult) -> str:
    """Translate per-claim counts into an envelope status."""
    if result.claim_count == 0:
        return "complete"  # no claims = nothing to cite
    if result.unsupported_count == 0 and result.weakly_supported_count == 0:
        return "complete"
    if result.unsupported_count == 0:
        return "partial"
    if result.supported_count == 0 and result.weakly_supported_count == 0:
        return "unsupported"
    return "partial" if result.unsupported_count <= result.claim_count // 2 else "unsupported"


__all__ = [
    "SUPPORTED_THRESHOLD",
    "WEAKLY_SUPPORTED_THRESHOLD",
    "ClaimValidationResult",
    "ClaimVerdict",
    "validate_claims",
]
