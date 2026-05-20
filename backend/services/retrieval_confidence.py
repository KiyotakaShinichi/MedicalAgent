"""Uncertainty-aware retrieval routing.

Takes the retrieval-and-validation outputs of a single turn and emits
an ``answerability_status`` plus its decomposed confidence signals so
the answer composer can choose between:

* answering fully with citations,
* answering with explicit limited-context language,
* refusing for lack of evidence,
* surfacing a conflict for clinician review,
* deferring to clinician,
* refusing due to safety.

This module is intentionally **input-only**: it does not call FAISS,
BM25, or any LLM.  Callers pass the already-scored chunks and the
already-validated claim envelope.  That makes the routing decision
testable in isolation and stable across retrieval backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# Allowed source tiers (mirrors clinical_ontology source tier names
# used by the rest of the project).
HIGH_TRUST_TIERS: frozenset[str] = frozenset({"T1", "T2"})
MID_TRUST_TIERS: frozenset[str] = frozenset({"T3"})

# Status enum — exact spelling matters; downstream eval keys on these.
ANSWERABILITY_STATUS_VALUES: tuple[str, ...] = (
    "answerable_with_citations",
    "answerable_with_limited_context",
    "insufficient_evidence",
    "conflicting_evidence",
    "clinician_review_required",
    "refuse_due_to_safety",
)


@dataclass
class RetrievalConfidence:
    retrieval_confidence: float
    source_tier_confidence: float
    citation_support_confidence: float
    evidence_conflict_flag: bool
    answerability_status: str
    reason: str
    top_score: float
    top_k_evaluated: int
    high_trust_chunks: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    safety_level: str
    safety_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_confidence": round(self.retrieval_confidence, 4),
            "source_tier_confidence": round(self.source_tier_confidence, 4),
            "citation_support_confidence": round(self.citation_support_confidence, 4),
            "evidence_conflict_flag": self.evidence_conflict_flag,
            "answerability_status": self.answerability_status,
            "reason": self.reason,
            "top_score": round(self.top_score, 4),
            "top_k_evaluated": self.top_k_evaluated,
            "high_trust_chunks": self.high_trust_chunks,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "unsupported_claims": self.unsupported_claims,
            "safety_level": self.safety_level,
            "safety_scope": self.safety_scope,
        }


def _retrieval_confidence_score(chunks: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Returns (confidence_in_[0,1], top_score).

    Heuristic: top-1 score above 0.6 → confident; below 0.2 →
    not confident.  Score field is whichever of ``score``, ``rrf_score``,
    or ``rerank_score`` the caller provides; we use ``score`` if
    present, else the max of those keys.
    """
    if not chunks:
        return 0.0, 0.0
    scores: list[float] = []
    for chunk in chunks:
        for key in ("score", "rerank_score", "rrf_score", "dense_score", "fusion_score"):
            val = chunk.get(key)
            if isinstance(val, (int, float)):
                scores.append(float(val))
                break
    if not scores:
        return 0.0, 0.0
    top = max(scores)
    # Map [0, 1] → [0, 1] linearly with a saturation knee at 0.6.
    confidence = min(1.0, max(0.0, top / 0.6))
    return confidence, top


def _source_tier_confidence(chunks: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
    if not chunks:
        return 0.0, 0
    high = sum(1 for c in chunks if str(c.get("source_tier") or c.get("tier") or "").upper() in HIGH_TRUST_TIERS)
    mid = sum(1 for c in chunks if str(c.get("source_tier") or c.get("tier") or "").upper() in MID_TRUST_TIERS)
    confidence = (high + 0.5 * mid) / len(chunks)
    return min(1.0, confidence), high


def _citation_support_confidence(claim_envelope: Mapping[str, Any] | None) -> tuple[float, dict[str, int]]:
    counts = {"supported": 0, "contradicted": 0, "unsupported": 0, "total": 0}
    if not claim_envelope:
        return 0.0, counts
    for verdict in claim_envelope.get("verdicts") or []:
        if verdict.get("is_claim") is False:
            continue
        status = str(verdict.get("status") or "").lower()
        counts["total"] += 1
        if status == "supported":
            counts["supported"] += 1
        elif status == "contradicted":
            counts["contradicted"] += 1
        else:
            counts["unsupported"] += 1
    if counts["total"] == 0:
        return 0.0, counts
    # Penalize contradictions heavily.
    score = (counts["supported"] - counts["contradicted"]) / counts["total"]
    return max(0.0, min(1.0, score)), counts


def _has_evidence_conflict(claim_envelope: Mapping[str, Any] | None) -> bool:
    if not claim_envelope:
        return False
    counts = {"supported": 0, "contradicted": 0}
    for verdict in claim_envelope.get("verdicts") or []:
        if verdict.get("is_claim") is False:
            continue
        status = str(verdict.get("status") or "").lower()
        if status in counts:
            counts[status] += 1
    return counts["supported"] > 0 and counts["contradicted"] > 0


def classify_retrieval_uncertainty(
    *,
    chunks: Sequence[Mapping[str, Any]],
    claim_envelope: Mapping[str, Any] | None,
    safety: Mapping[str, Any] | None,
    intent: str | None = None,
) -> RetrievalConfidence:
    """Decide the answerability_status for a single turn.

    Precedence (highest priority first):

    1. ``refuse_due_to_safety`` — safety scope is high-risk; nothing
       else matters for routing.
    2. ``conflicting_evidence`` — claim validator reports both
       supported and contradicted claims for this draft.
    3. ``clinician_review_required`` — patient-specific intent
       (e.g. record_explanation) AND citation support is low.
    4. ``insufficient_evidence`` — retrieval confidence < 0.3 OR
       support confidence < 0.3 OR no governed T2/T3-or-better basis.
    5. ``answerable_with_limited_context`` — middling confidence on
       at least one axis.
    6. ``answerable_with_citations`` — all three confidence axes
       above their floors.
    """
    safety = safety or {}
    retrieval_conf, top_score = _retrieval_confidence_score(chunks)
    tier_conf, high_trust = _source_tier_confidence(chunks)
    support_conf, claim_counts = _citation_support_confidence(claim_envelope)
    conflict = _has_evidence_conflict(claim_envelope)
    safety_level = str(safety.get("level") or "low_risk")
    safety_scope = str(safety.get("scope") or "education_or_tracking")

    if safety_level == "high_risk":
        status = "refuse_due_to_safety"
        reason = f"safety scope {safety_scope!r} requires refusal regardless of retrieval signal"
    elif conflict:
        status = "conflicting_evidence"
        reason = "claim validator returned both supported and contradicted claims"
    elif intent in {"record_explanation", "record_explanation_rag"} and support_conf < 0.5:
        status = "clinician_review_required"
        reason = "patient-specific record question with low citation support"
    elif retrieval_conf < 0.3 or support_conf < 0.3 or (high_trust == 0 and tier_conf < 0.5):
        status = "insufficient_evidence"
        reason = (
            f"retrieval_conf={retrieval_conf:.2f} support={support_conf:.2f} "
            f"high_trust_chunks={high_trust}"
        )
    elif retrieval_conf < 0.6 or support_conf < 0.6 or tier_conf < 0.5:
        status = "answerable_with_limited_context"
        reason = (
            f"middling confidence: retrieval={retrieval_conf:.2f} "
            f"support={support_conf:.2f} tier={tier_conf:.2f}"
        )
    else:
        status = "answerable_with_citations"
        reason = "all confidence axes above floor"

    return RetrievalConfidence(
        retrieval_confidence=retrieval_conf,
        source_tier_confidence=tier_conf,
        citation_support_confidence=support_conf,
        evidence_conflict_flag=conflict,
        answerability_status=status,
        reason=reason,
        top_score=top_score,
        top_k_evaluated=len(chunks),
        high_trust_chunks=high_trust,
        supported_claims=claim_counts.get("supported", 0),
        contradicted_claims=claim_counts.get("contradicted", 0),
        unsupported_claims=claim_counts.get("unsupported", 0),
        safety_level=safety_level,
        safety_scope=safety_scope,
    )


__all__ = [
    "ANSWERABILITY_STATUS_VALUES",
    "HIGH_TRUST_TIERS",
    "MID_TRUST_TIERS",
    "RetrievalConfidence",
    "classify_retrieval_uncertainty",
]
