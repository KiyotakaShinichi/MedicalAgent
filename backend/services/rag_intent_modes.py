"""Intent-aware RAG mode configuration.

The pre-existing intent router classifies each query into one of ~11
intents (education, safety_boundary, portal_help, etc.).  This module
wraps that classification with **RAG execution modes** that describe
*how* retrieval and answer composition should run for each intent:

  - which source tiers + allowed_use sets are valid,
  - what claim boundaries must be enforced after generation,
  - whether the mode is patient-facing or clinician-facing,
  - whether the mode is allowed to cite at all,
  - how aggressively to abstain when evidence is weak.

The five canonical modes per the Phase 11 spec:

  - ``education_rag``        — patient/clinician education questions
                              (e.g. "what does WBC mean?")
  - ``urgent_safety_rag``    — urgent symptoms or safety-boundary
                              questions ("nilalagnat ako, ANC mababa")
  - ``record_explanation_rag`` — explains the patient's own timeline
                              ("why was my CBC flagged last cycle?")
  - ``clinician_context_rag`` — clinician viewing a patient — broader
                              evidence + clinician-only sources allowed
  - ``portal_help_rag``      — "how do I do X in the portal?" — only
                              portal docs

This is engineering scaffolding.  The claim boundaries documented here
are defaults a clinical advisor can review; nothing here should be
treated as a clinical recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# Source tiers come from the existing kb_source_governance artifact.
# We import the constants so a future edit there is reflected here.
from backend.services.kb_source_governance import (
    ALLOWED_USE_VOCABULARY,
    TIER_ORDER,
)


# ─── Mode envelope ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RagModeConfig:
    """Per-mode retrieval + answer-composition policy."""

    mode: str
    description: str
    audience: str  # "patient" | "clinician" | "either"
    allowed_tiers: tuple[str, ...]          # subset of TIER_ORDER
    allowed_use: tuple[str, ...]            # subset of ALLOWED_USE_VOCABULARY
    allow_citations: bool                   # if False, validator strips citations
    insufficient_evidence_default: str      # what to return when grade == insufficient
    banned_claim_categories: tuple[str, ...]  # post-gen validator rules this mode enforces
    max_retrieved_chunks: int               # cap on retrieved evidence
    # When ``True``, the answer composition must include a "discuss with
    # clinician" clause regardless of evidence grade.  Used for any mode
    # that touches a patient-specific decision.
    require_clinician_handoff_clause: bool = False

    def __post_init__(self) -> None:
        for tier in self.allowed_tiers:
            if tier not in TIER_ORDER:
                raise ValueError(f"Unknown tier {tier!r} in mode {self.mode!r}")
        for use in self.allowed_use:
            if use not in ALLOWED_USE_VOCABULARY:
                raise ValueError(f"Unknown allowed_use {use!r} in mode {self.mode!r}")


# Claim categories enforced on every mode by default; specific modes can
# extend.  These map 1-to-1 to rule codes in the post_generation_validator.
COMMON_BANNED_CLAIMS: tuple[str, ...] = (
    "diagnosis_claim",
    "treatment_recommendation",
    "prognosis_estimate",
    "dosage_instruction",
    "genetic_risk_overclaim",
    "tumor_marker_overclaim",
)


# ─── Per-mode configs ────────────────────────────────────────────────────────


MODES: dict[str, RagModeConfig] = {
    "education_rag": RagModeConfig(
        mode="education_rag",
        description=(
            "Patient/clinician education questions.  Answer with cited "
            "evidence from T1/T2 sources; T3 patient-education is allowed "
            "when no higher-tier source covers the question."
        ),
        audience="either",
        allowed_tiers=("T1", "T2", "T3"),
        allowed_use=("education",),
        allow_citations=True,
        insufficient_evidence_default=(
            "I don't have enough sourced evidence to answer that. Please "
            "bring this question up with your oncology team — they can "
            "interpret it in the full context of your treatment."
        ),
        banned_claim_categories=COMMON_BANNED_CLAIMS,
        max_retrieved_chunks=6,
    ),
    "urgent_safety_rag": RagModeConfig(
        mode="urgent_safety_rag",
        description=(
            "Urgent symptoms or safety-boundary questions.  Does NOT run "
            "RAG retrieval; instead routes to the deterministic safety "
            "response so the patient doesn't wait on generation latency."
        ),
        audience="patient",
        allowed_tiers=("T1",),
        allowed_use=("patient_safety",),
        allow_citations=False,
        insufficient_evidence_default=(
            "If you are having severe or sudden symptoms — chest pain, "
            "trouble breathing, heavy bleeding, fever after chemotherapy, "
            "fainting — contact your oncology team or local emergency "
            "services immediately. Do not wait for a portal response."
        ),
        banned_claim_categories=COMMON_BANNED_CLAIMS,
        max_retrieved_chunks=0,  # No retrieval — safety route fires deterministically.
        require_clinician_handoff_clause=True,
    ),
    "record_explanation_rag": RagModeConfig(
        mode="record_explanation_rag",
        description=(
            "Explains the patient's OWN timeline (their CBC, their symptom "
            "history) using citations to the source records, not to the KB. "
            "Education-tier KB sources are allowed as supporting context."
        ),
        audience="patient",
        allowed_tiers=("T1", "T2", "T3"),
        allowed_use=("education", "monitoring_context"),
        allow_citations=True,
        insufficient_evidence_default=(
            "I can see your record but I'm not able to interpret that for "
            "you. Please review this with your oncology team."
        ),
        banned_claim_categories=COMMON_BANNED_CLAIMS,
        max_retrieved_chunks=5,
        require_clinician_handoff_clause=True,
    ),
    "clinician_context_rag": RagModeConfig(
        mode="clinician_context_rag",
        description=(
            "Clinician viewing a patient.  Allowed to cite clinician-only "
            "sources and a broader evidence base, but still cannot make "
            "diagnosis/treatment/prognosis claims on the model's authority."
        ),
        audience="clinician",
        allowed_tiers=("T1", "T2", "T3"),
        allowed_use=("clinician_only", "education", "monitoring_context"),
        allow_citations=True,
        insufficient_evidence_default=(
            "Insufficient sourced evidence to summarise. Review the patient "
            "record directly."
        ),
        banned_claim_categories=COMMON_BANNED_CLAIMS,
        max_retrieved_chunks=8,
    ),
    "portal_help_rag": RagModeConfig(
        mode="portal_help_rag",
        description=(
            "How-do-I questions about the portal itself.  Only internal "
            "portal documentation is allowed as evidence; no medical "
            "content."
        ),
        audience="either",
        allowed_tiers=("T4",),
        allowed_use=("portal_help",),
        allow_citations=True,
        insufficient_evidence_default=(
            "I don't have portal help text for that question yet. The "
            "support team can help — use the contact info in the footer."
        ),
        # Portal help cannot make medical claims at all; enforce the full
        # banned-claim list.
        banned_claim_categories=COMMON_BANNED_CLAIMS,
        max_retrieved_chunks=4,
    ),
}


# ─── Intent → mode mapping ───────────────────────────────────────────────────


# Maps the existing `route_intent` output to one of the 5 RAG modes.
# Intents that don't use RAG at all map to ``None`` so callers can skip
# the retrieval stack entirely.
INTENT_TO_MODE: dict[str, str | None] = {
    "education":                     "education_rag",
    "patient_timeline_monitoring":   "record_explanation_rag",
    "portal_help":                   "portal_help_rag",
    "safety_boundary":               "urgent_safety_rag",
    "treatment_decision_boundary":   "urgent_safety_rag",
    "security_boundary":             None,   # blocked before RAG even fires
    "data_entry_confirmation":       None,   # tool flow, not RAG
    "patient_memory":                None,
    "conversation":                  None,
    "emotional_support":             None,
    "general_support":               None,
}


def select_mode(
    intent: str | None,
    *,
    actor_role: str | None = None,
) -> RagModeConfig | None:
    """Pick the RAG mode for a given intent + actor role.

    Returns ``None`` when the intent doesn't use RAG.  When the actor is
    a clinician AND the base intent would have been education or record
    explanation, the mode is upgraded to ``clinician_context_rag`` so the
    broader evidence base + clinician-only sources become available.
    """
    if intent is None:
        return None
    base_mode_key = INTENT_TO_MODE.get(intent)
    if base_mode_key is None:
        return None
    if actor_role == "clinician" and base_mode_key in {"education_rag", "record_explanation_rag"}:
        return MODES["clinician_context_rag"]
    return MODES[base_mode_key]


def list_modes() -> Mapping[str, RagModeConfig]:
    """Read-only view of the mode registry — used by the admin card."""
    return dict(MODES)


__all__ = [
    "COMMON_BANNED_CLAIMS",
    "INTENT_TO_MODE",
    "MODES",
    "RagModeConfig",
    "list_modes",
    "select_mode",
]
