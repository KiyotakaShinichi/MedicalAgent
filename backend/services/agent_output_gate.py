"""Output-stage guardrails for the patient agent.

Sibling of ``agent_input_gate.py``.  Runs after the answer is composed
and the post-gen safety validator has had its say.  Detects three
classes of failure that the upstream layers can miss:

  1. Unsafe treatment-directive or diagnostic claim wording that
     somehow survived ``contains_diagnostic_or_treatment_claim``.
  2. A reply that quotes retrieved context but emits zero citations
     (only flagged on non-refusal intents — refusals intentionally strip
     citations; see ``agent_answer_composition.generate_answer``).
  3. A high-risk safety reply that forgets to escalate to a clinician /
     emergency / oncology / care team — the deterministic refusal
     checks downstream depend on at least one of those phrases being
     present.

Extracted from ``agent_rag.py`` as part of the agent_rag.py module
split.  ``output_guardrail_check`` is re-exported from
``backend.services.agent_rag`` so existing imports work unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping

from backend.services.agent_answer_composition import REFUSAL_INTENTS


# Reply wordings that fail the output gate as either a treatment
# directive or a diagnostic claim.  Promoted from an inline list so the
# vocabulary is auditable alongside the input-gate ``BLOCKING_ISSUES``.
UNSAFE_OUTPUT_TERMS: tuple[str, ...] = (
    "you should stop",
    "you should start",
    "increase your dose",
    "decrease your dose",
    "skip chemo",
    "you are cancer free",
    "you have metastasis",
)


# Phrases that satisfy the high-risk "must escalate" check.  Any single
# occurrence in the reply text counts.
ESCALATION_PHRASES: tuple[str, ...] = (
    "oncology",
    "emergency",
    "clinician",
    "care team",
)


def output_guardrail_check(result: Mapping[str, Any]) -> dict[str, Any]:
    """Run the output-stage guardrail and return ``{"status": ..., "issues": [...]}``.

    Behavior preserved verbatim from the original inline implementation
    in ``agent_rag.py``.  Issues are appended to whatever the validator
    already recorded under ``result["validation"]["issues"]``; the union
    is deduped and sorted.
    """
    reply = result.get("reply") or ""
    validation = result.get("validation") or {}
    issues = list(validation.get("issues") or [])
    intent = result.get("intent")
    rag_mode = result.get("rag_mode")
    lower_reply = reply.lower()

    if any(term in lower_reply for term in UNSAFE_OUTPUT_TERMS):
        issues.append("unsafe_output_directive_or_diagnosis")

    # On refusal intents, citations are intentionally stripped (see
    # generate_answer); the missing-citations check would otherwise
    # fire on every safety_boundary / treatment_decision_boundary reply
    # that surfaces background education context for display.
    non_evidence_portal_help = intent == "portal_help" and rag_mode == "portal_help_rag"
    if (
        intent not in REFUSAL_INTENTS
        and not non_evidence_portal_help
        and (result.get("retrieval_context") or [])
        and not (result.get("citations") or [])
        and not result.get("deliberate_evidence_abstention")
    ):
        issues.append("missing_citations")

    safety = result.get("safety") or {}
    if safety.get("level") == "high_risk" and not any(term in lower_reply for term in ESCALATION_PHRASES):
        issues.append("missing_high_risk_escalation")

    return {
        "status": "passed" if not issues else "failed",
        "issues": sorted(set(issues)),
    }


__all__ = [
    "UNSAFE_OUTPUT_TERMS",
    "ESCALATION_PHRASES",
    "output_guardrail_check",
]
