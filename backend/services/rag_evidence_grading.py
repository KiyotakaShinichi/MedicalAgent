"""RAG evidence-grading envelope.

Wraps every RAG response with a structured grade that downstream code
can act on:

  - **grade**: ``high`` / ``moderate`` / ``low`` / ``insufficient``
  - **source_basis**: list of source IDs (with tiers) the answer leans on
  - **citation_status**: pulled from `rag_claim_validator`
  - **answer_scope**: ``factual_education`` / ``patient_record_summary``
    / ``safety_routing`` / ``insufficient_evidence`` / ``portal_help``
  - **reasoning**: short human-readable explanation
  - **claim_support_rate**: from the claim validator
  - **tier_distribution_of_basis**: count of basis sources at each tier

The grade is the single signal a caller (patient chat, clinician chat,
trace replay) consumes to decide whether to surface the reply as-is,
mark it as "limited evidence", or replace it with the mode's
insufficient-evidence default.

Engineering provenance only.  A "high" grade does not establish clinical
correctness — it means the engineering contract (claim support + source
tier + retrieved-evidence count) was satisfied.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from backend.services.rag_claim_validator import ClaimValidationResult
from backend.services.rag_intent_modes import RagModeConfig
from backend.services.rag_tier_filter import FilterResult, known_tier_for_source


# Scope mapping per RAG mode.  Insufficient evidence has its own scope
# label so callers don't conflate "we found nothing" with "we found
# something off-topic".
_MODE_SCOPE: dict[str, str] = {
    "education_rag":          "factual_education",
    "record_explanation_rag": "patient_record_summary",
    "clinician_context_rag":  "clinician_review_context",
    "urgent_safety_rag":      "safety_routing",
    "portal_help_rag":        "portal_help",
}


@dataclass
class EvidenceGrade:
    grade: str  # "high" | "moderate" | "low" | "insufficient"
    answer_scope: str
    citation_status: str
    claim_support_rate: float
    source_basis: list[dict[str, str]] = field(default_factory=list)
    tier_distribution_of_basis: dict[str, int] = field(default_factory=dict)
    reasoning: str = ""
    mode: str = ""
    claim_count: int = 0
    supported_count: int = 0
    unsupported_count: int = 0

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "answer_scope": self.answer_scope,
            "citation_status": self.citation_status,
            "claim_support_rate": round(self.claim_support_rate, 4),
            "source_basis": list(self.source_basis),
            "tier_distribution_of_basis": dict(self.tier_distribution_of_basis),
            "reasoning": self.reasoning,
            "mode": self.mode,
            "claim_count": self.claim_count,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
        }


def grade_evidence(
    *,
    mode: RagModeConfig,
    filter_result: FilterResult,
    claim_validation: ClaimValidationResult,
    retrieved_count_before_filter: int,
) -> EvidenceGrade:
    """Produce the envelope.  Pure function — does not touch retrieval
    or generation."""
    kept_chunks = filter_result.kept_chunks
    kept_count = len(kept_chunks)

    # Build source-basis from the kept chunks (best-effort source labels).
    basis: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for chunk in kept_chunks:
        source_id = str(chunk.get("parent_id") or chunk.get("source_id") or chunk.get("source_name") or "")
        if not source_id or source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        basis.append({
            "source_id": source_id,
            "title": str(chunk.get("title") or chunk.get("source_name") or "Untitled source"),
            "tier": known_tier_for_source(source_id) or "T5",
        })

    tier_dist = Counter(item["tier"] for item in basis)

    # The grading ladder, ordered from "most must hold" to "least":
    grade, reasoning = _compute_grade(
        mode=mode,
        kept_count=kept_count,
        retrieved_before=retrieved_count_before_filter,
        claim_validation=claim_validation,
        tier_dist=tier_dist,
    )

    scope = (
        "insufficient_evidence"
        if grade == "insufficient"
        else _MODE_SCOPE.get(mode.mode, "general_support")
    )

    return EvidenceGrade(
        grade=grade,
        answer_scope=scope,
        citation_status=claim_validation.citation_status,
        claim_support_rate=claim_validation.claim_support_rate,
        source_basis=basis,
        tier_distribution_of_basis=dict(tier_dist),
        reasoning=reasoning,
        mode=mode.mode,
        claim_count=claim_validation.claim_count,
        supported_count=claim_validation.supported_count,
        unsupported_count=claim_validation.unsupported_count,
    )


# ─── Grading logic ───────────────────────────────────────────────────────────


def _compute_grade(
    *,
    mode: RagModeConfig,
    kept_count: int,
    retrieved_before: int,
    claim_validation: ClaimValidationResult,
    tier_dist: Counter[str],
) -> tuple[str, str]:
    """The deciding rules, in order:

      1. Mode is no-retrieval (urgent_safety): grade = high if a deterministic
         safety response fired, else moderate.  This branch returns early.
      2. No chunks survived governance filtering: insufficient.
      3. Claim validator reports ``unsupported`` citation status with at
         least one claim: insufficient (the answer made claims that nothing
         retrieved actually supports).
      4. Otherwise:
           - high   when ≥1 T1 source backs ≥1 supported claim AND no
             unsupported claims,
           - moderate when supported_count > 0 AND unsupported_count <= 1,
           - low    when claim_support_rate < 0.5,
           - insufficient when there are no claims to support and the mode
             expected some.
    """
    # Mode 1: urgent_safety doesn't run retrieval.  Caller is responsible
    # for the safety-route response; we don't second-guess it.
    if mode.max_retrieved_chunks == 0:
        return "high", "urgent_safety_route: deterministic response, no retrieval required"

    if kept_count == 0 and retrieved_before == 0:
        return "insufficient", "retrieval returned no chunks at all"
    if kept_count == 0 and retrieved_before > 0:
        return "insufficient", (
            f"all {retrieved_before} retrieved chunks were filtered out by "
            "tier/allowed_use rules — no governance-compatible evidence available"
        )

    if claim_validation.citation_status == "unsupported" and claim_validation.claim_count > 0:
        return "insufficient", (
            f"claim validator flagged {claim_validation.unsupported_count} of "
            f"{claim_validation.claim_count} claims as unsupported by any retrieved chunk"
        )

    if claim_validation.claim_count == 0:
        # No claims means the reply was framing/deferral only.  That can
        # be the *right* answer for portal_help or safety modes, but for
        # education / record explanation it usually means the reply
        # didn't actually answer.
        if mode.mode in {"portal_help_rag"}:
            return "moderate", "reply has no claims (portal help framing-only is acceptable)"
        return "low", "reply contains no substantive claims to grade"

    has_t1 = tier_dist.get("T1", 0) > 0
    if (
        has_t1
        and claim_validation.supported_count >= 1
        and claim_validation.unsupported_count == 0
    ):
        return "high", (
            f"{claim_validation.supported_count} supported claim(s) backed by "
            f"≥1 T1 source, no unsupported claims"
        )
    if (
        claim_validation.supported_count > 0
        and claim_validation.unsupported_count <= 1
    ):
        return "moderate", (
            f"{claim_validation.supported_count} supported / "
            f"{claim_validation.unsupported_count} unsupported claim(s); "
            "no T1 backing OR one borderline claim"
        )
    if claim_validation.claim_support_rate < 0.5:
        return "low", (
            f"only {claim_validation.supported_count} of {claim_validation.claim_count} "
            "claims are well-supported"
        )
    return "moderate", "default moderate (no triggering rule fired)"


__all__ = [
    "EvidenceGrade",
    "grade_evidence",
]
