"""Safety scope classification for the patient agent.

A single pure function — ``safety_scope_check`` — classifies an inbound
query into one of four safety scopes (urgent / treatment_decision /
diagnostic / low_risk) and returns the policy envelope the downstream
agent consumes.

This was extracted from ``agent_rag.py`` so the 2,076-line god module
becomes navigable and unit-testing the safety vocabulary doesn't have
to import the full retrieval stack.

Public contract preserved
~~~~~~~~~~~~~~~~~~~~~~~~~
Six call sites import ``safety_scope_check`` directly from
``backend.services.agent_rag`` (chat, eval scripts, tests).  The shim
re-export in ``agent_rag`` keeps those imports working without a
breaking-rename diff.
"""

from __future__ import annotations

from typing import Any, Sequence

from backend.services.security_guardrails import (
    detect_multilingual_medical_danger,
)
from backend.services.unsafe_intent_context import (
    classify_unsafe_intent_with_context,
)
from backend.services.multilingual_semantic_safety import (
    classify_multilingual_safety,
)
from backend.services.dep001d_semantic_safety import (
    classify_dep001d_safety as classify_dep001b_safety,
)
from backend.services.safety_policy_action import PolicyAction, scope_for_policy_action


from backend.services.agent_safety_vocab import (
    DECISION_TERMS,
    DIAGNOSTIC_TERMS,
    URGENT_TERMS,
)
from backend.services.agent_safety_rules import (
    _is_non_actionable_symptom_disclosure,
    _is_clearly_nonurgent_symptom_disclosure,
    _is_unambiguous_safe_information_request,
    _deterministic_high_confidence_boundary,
    _is_privacy_process_education,
    _safe_information_envelope,
    _is_general_interaction_education,
    _current_urgent_term_matches,
    _has_explicit_personal_action_request,
    _trusted_low_risk_consensus,
)


# ─── Public API ──────────────────────────────────────────────────────────────


def safety_scope_check(
    query: str,
    urgent_flags: Sequence[str] | None = None,
    previous_user_messages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Classify a query into a safety scope and return the policy envelope.

    Precedence (highest-risk first):
      1. ``urgent`` — an upstream lab/imaging extractor flagged it, OR the
         multilingual medical-danger detector matched, OR an English/Taglish
         urgent term matched.
      2. ``treatment_decision`` — wording requests a clinician-level
         medication / treatment decision.
      3. ``diagnostic`` — wording asks the assistant to confirm or deny a
         diagnosis / prognosis.
      4. ``low_risk`` — default educational or portal-support query.

    The returned envelope shape is preserved verbatim from the original
    in-line implementation in ``agent_rag.py``; ``cache_allowed`` is False
    on every high-risk branch so the response cache never serves a
    safety-routed reply to a future query.
    """
    urgent_flags = list(urgent_flags or [])
    if urgent_flags:
        return {
            "level": "high_risk",
            "scope": "urgent_or_safety_related",
            "cache_allowed": False,
            "message": "An upstream structured safety flag requires urgent human review.",
            "matched_safety_terms": sorted(set(urgent_flags))[:10],
            "safety_source": "deterministic_structured_urgent_flag",
            "policy_action": PolicyAction.URGENT_ESCALATION.value,
        }
    # This deterministic lane is intentionally independent of the semantic
    # intent-family prediction. A model cannot veto an explicit multilingual
    # urgent presentation, and downstream retrieval is never consulted first.
    deterministic_danger = detect_multilingual_medical_danger(query)
    current_urgent_terms = _current_urgent_term_matches(query)
    if deterministic_danger.get("detected") or current_urgent_terms:
        matches = [str(item) for item in deterministic_danger.get("matches") or []]
        matches.extend(current_urgent_terms)
        return {
            "level": "high_risk",
            "scope": "urgent_or_safety_related",
            "cache_allowed": False,
            "message": "Potential urgent presentation detected; route to immediate human or emergency review.",
            "matched_safety_terms": matches[:10],
            "safety_source": "deterministic_multilingual_urgent_signal",
            "policy_action": PolicyAction.URGENT_ESCALATION.value,
            "policy_intent_family": "URGENT_PRESENTATION",
            "unsafe_intent_family": "urgent_deterioration",
            "unsafe_intent_confidence": 1.0,
            "urgent_probability": 1.0,
            "policy_uncertainty": 0.0,
            "context_reused": False,
            "context_turn_count": 0,
            "safe_boundary_request": False,
            "safety_control_failure": None,
        }
    explicit_safe_education = _is_unambiguous_safe_information_request(query)
    deterministic_boundary = _deterministic_high_confidence_boundary(query)
    if deterministic_boundary and not explicit_safe_education:
        return deterministic_boundary
    if explicit_safe_education:
        envelope = _safe_information_envelope()
        if _is_privacy_process_education(query):
            envelope["safety_source"] = "safe_boundary_request"
        return envelope
    try:
        prediction = classify_dep001b_safety(
            query,
            previous_user_messages=tuple(previous_user_messages or ()),
        )
    except Exception as exc:
        return {
            "level": "high_risk",
            "scope": "safety_control_unavailable",
            "cache_allowed": False,
            "message": "The safety classifier failed; do not continue to retrieval or generation.",
            "safety_source": "dep001d_fail_closed_wrapper",
            "policy_action": PolicyAction.FAIL_CLOSED.value,
            "unsafe_intent_family": "UNKNOWN_HIGH_RISK",
            "unsafe_intent_confidence": 1.0,
            "urgent_probability": 0.0,
            "policy_uncertainty": 1.0,
            "context_reused": False,
            "context_turn_count": 0,
            "safety_control_failure": f"classifier_exception:{type(exc).__name__}",
        }
    try:
        action = PolicyAction(prediction.policy_action)
    except ValueError:
        action = PolicyAction.FAIL_CLOSED
    # Preserve the established public metadata contract. The old classifier
    # may resolve only an uncertain SAFE_REDIRECT when it identifies a narrow,
    # deterministic safe-process request. It cannot override an actionable
    # refusal, urgent escalation, or fail-closed decision.
    try:
        legacy_metadata = classify_unsafe_intent_with_context(
            query,
            tuple(previous_user_messages or ()),
        )
    except Exception:
        legacy_metadata = {}
    try:
        auxiliary_semantic = classify_multilingual_safety(
            query,
            previous_user_messages=tuple(previous_user_messages or ()),
        )
    except Exception:
        auxiliary_semantic = None
    deterministic_safe_process = (
        not bool(legacy_metadata.get("is_unsafe"))
        and legacy_metadata.get("safety_source") == "safe_boundary_request"
    )
    low_risk_consensus = _trusted_low_risk_consensus(
        query=query,
        legacy_metadata=legacy_metadata,
        auxiliary_semantic=auxiliary_semantic,
        dep001d_prediction=prediction,
    )
    if (
        action
        in {
            PolicyAction.SAFE_REDIRECT,
            PolicyAction.REFUSE_ACTIONABLE,
            PolicyAction.URGENT_ESCALATION,
        }
        and low_risk_consensus
    ):
        action = (
            PolicyAction.ALLOW_EDUCATIONAL
            if low_risk_consensus in {"safe_educational", "safe_research_lookup"}
            else PolicyAction.ALLOW_WITH_BOUNDARY
        )
    if (
        action in {PolicyAction.SAFE_REDIRECT, PolicyAction.REFUSE_ACTIONABLE}
        and deterministic_safe_process
    ):
        action = PolicyAction.ALLOW_WITH_BOUNDARY
    if (
        action == PolicyAction.SAFE_REDIRECT
        and prediction.intent_family == "ACTIVE_SYMPTOM_MANAGEMENT"
        and not bool(legacy_metadata.get("is_unsafe"))
        and _is_non_actionable_symptom_disclosure(query)
    ):
        action = PolicyAction.ALLOW_WITH_BOUNDARY
    if action == PolicyAction.URGENT_ESCALATION and _is_clearly_nonurgent_symptom_disclosure(query):
        action = PolicyAction.ALLOW_WITH_BOUNDARY
    if (
        action == PolicyAction.SAFE_REDIRECT
        and prediction.intent_family == "INTERACTION_EDUCATION"
        and _is_general_interaction_education(query)
    ):
        action = PolicyAction.ALLOW_EDUCATIONAL
    if (
        action in {PolicyAction.ALLOW_EDUCATIONAL, PolicyAction.ALLOW_WITH_BOUNDARY}
        and not deterministic_safe_process
        and prediction.intent_family
        in {
            "PERSONALIZED_ACTION_REQUEST",
            "TREATMENT_MODIFICATION",
            "MEDICATION_DOSING",
            "ACTIVE_SYMPTOM_MANAGEMENT",
            "PERSONALIZED_INTERACTION_ACTION",
            "TUMOR_MARKER_TREATMENT_DECISION",
        }
        and _has_explicit_personal_action_request(query)
    ):
        action = (
            PolicyAction.SAFE_REDIRECT
            if prediction.intent_family == "ACTIVE_SYMPTOM_MANAGEMENT"
            else PolicyAction.REFUSE_ACTIONABLE
        )
    scope = scope_for_policy_action(action, prediction.intent_family)
    high_risk = action in {
        PolicyAction.SAFE_REDIRECT,
        PolicyAction.REFUSE_ACTIONABLE,
        PolicyAction.URGENT_ESCALATION,
        PolicyAction.FAIL_CLOSED,
    }
    if action == PolicyAction.URGENT_ESCALATION:
        message = (
            "Potential urgent presentation detected; route to immediate human or emergency review."
        )
    elif action == PolicyAction.REFUSE_ACTIONABLE:
        message = "A personalized medical action was requested; do not provide diagnosis, dosing, or treatment direction."
    elif action == PolicyAction.SAFE_REDIRECT:
        message = "The request requires a bounded clarification, refusal, or human-review redirect."
    elif action == PolicyAction.FAIL_CLOSED:
        message = (
            "The safety classifier is unavailable; do not continue to retrieval or generation."
        )
    elif action == PolicyAction.ALLOW_EDUCATIONAL:
        message = "High-confidence general education intent; answer within evidence and clinical boundaries."
    else:
        message = "Low-risk information request; answer with an explicit non-diagnostic boundary."
    legacy_unsafe = bool(legacy_metadata.get("is_unsafe"))
    legacy_safe_boundary = not high_risk and deterministic_safe_process
    if high_risk and legacy_unsafe:
        scope = str(legacy_metadata.get("scope") or scope)
    unsafe_intent_family = (
        str(legacy_metadata.get("unsafe_intent_family") or legacy_metadata.get("family"))
        if legacy_unsafe
        else prediction.intent_family
    )
    if legacy_safe_boundary:
        safety_source = "safe_boundary_request"
    elif high_risk and legacy_unsafe and legacy_metadata.get("context_reused"):
        safety_source = str(legacy_metadata.get("safety_source") or "contextual_unsafe_intent")
    else:
        safety_source = "dep001d_explicit_policy_action"
    return {
        "level": "high_risk" if high_risk else "low_risk",
        "scope": scope,
        "cache_allowed": not high_risk,
        "message": message,
        "safety_source": safety_source,
        "policy_action": action.value,
        "policy_intent_family": prediction.intent_family,
        "unsafe_intent_family": unsafe_intent_family,
        "unsafe_intent_confidence": prediction.unsafe_probability,
        "urgent_probability": prediction.urgent_probability,
        "policy_uncertainty": prediction.uncertainty,
        "context_reused": prediction.context_turn_count > 0,
        "context_turn_count": prediction.context_turn_count,
        "semantic_safety": prediction.to_dict(),
        "auxiliary_semantic_safety": (
            auxiliary_semantic.to_dict() if auxiliary_semantic is not None else None
        ),
        "low_risk_consensus": low_risk_consensus,
        "legacy_safety_metadata": {
            "family": legacy_metadata.get("family"),
            "scope": legacy_metadata.get("scope"),
            "safety_source": legacy_metadata.get("safety_source"),
            "context_reused": bool(legacy_metadata.get("context_reused")),
        },
        "safe_boundary_request": legacy_safe_boundary,
        "safety_control_failure": prediction.failure_reason,
    }


__all__ = [
    "DECISION_TERMS",
    "DIAGNOSTIC_TERMS",
    "URGENT_TERMS",
    "safety_scope_check",
]
