from __future__ import annotations

import pytest

from backend.services.agent_safety import (
    _current_urgent_term_matches,
    _has_explicit_personal_action_request,
    _is_general_interaction_education,
    _is_non_actionable_symptom_disclosure,
    safety_scope_check,
)
from backend.services.safety_policy_action import IntentFamily, PolicyAction, select_policy_action


THRESHOLDS = {
    "unsafe_route_threshold": 0.45,
    "urgent_route_threshold": 0.20,
    "urgent_independent_threshold": 0.85,
    "urgent_family_support_floor": 0.20,
    "intent_family_confidence_threshold": 0.50,
    "uncertainty_route_threshold": 0.80,
}


def decide(unsafe: float, urgent: float, family: IntentFamily, confidence: float = 0.9, uncertainty: float = 0.1, failure: str | None = None) -> PolicyAction:
    return select_policy_action(
        unsafe_probability=unsafe,
        urgent_probability=urgent,
        intent_family=family.value,
        intent_family_confidence=confidence,
        uncertainty=uncertainty,
        thresholds=THRESHOLDS,
        failure_reason=failure,
    ).action


def test_system_failure_has_highest_precedence() -> None:
    assert decide(1.0, 1.0, IntentFamily.URGENT_PRESENTATION, failure="injected") == PolicyAction.FAIL_CLOSED


def test_urgent_precedes_generic_actionable_refusal() -> None:
    assert decide(0.99, 0.90, IntentFamily.TREATMENT_MODIFICATION) == PolicyAction.URGENT_ESCALATION


def test_urgent_family_can_use_supported_threshold() -> None:
    assert decide(0.4, 0.3, IntentFamily.URGENT_PRESENTATION) == PolicyAction.URGENT_ESCALATION


def test_independent_urgent_head_does_not_require_urgent_family() -> None:
    assert decide(0.4, 0.9, IntentFamily.PERSONALIZED_INFORMATION) == PolicyAction.URGENT_ESCALATION


@pytest.mark.parametrize("family", [
    IntentFamily.EDUCATIONAL_GENERAL,
    IntentFamily.SYMPTOM_EDUCATION,
    IntentFamily.INTERACTION_EDUCATION,
    IntentFamily.TUMOR_MARKER_EDUCATION,
])
def test_high_confidence_education_stays_answerable(family: IntentFamily) -> None:
    assert decide(0.1, 0.01, family) == PolicyAction.ALLOW_EDUCATIONAL


@pytest.mark.parametrize("family", [
    IntentFamily.PERSONALIZED_ACTION_REQUEST,
    IntentFamily.TREATMENT_MODIFICATION,
    IntentFamily.MEDICATION_DOSING,
    IntentFamily.PERSONALIZED_INTERACTION_ACTION,
    IntentFamily.TUMOR_MARKER_TREATMENT_DECISION,
])
def test_personalized_action_is_not_answered(family: IntentFamily) -> None:
    assert decide(0.9, 0.01, family) == PolicyAction.REFUSE_ACTIONABLE


def test_active_nonurgent_symptom_management_redirects() -> None:
    assert decide(0.9, 0.01, IntentFamily.ACTIVE_SYMPTOM_MANAGEMENT) == PolicyAction.SAFE_REDIRECT


def test_safe_family_and_unsafe_head_disagreement_does_not_allow() -> None:
    assert decide(0.9, 0.01, IntentFamily.EDUCATIONAL_GENERAL) == PolicyAction.SAFE_REDIRECT


def test_actionable_family_without_unsafe_head_still_refuses() -> None:
    assert decide(0.1, 0.01, IntentFamily.TREATMENT_MODIFICATION) == PolicyAction.REFUSE_ACTIONABLE


def test_high_uncertainty_uses_bounded_redirect() -> None:
    assert decide(0.3, 0.01, IntentFamily.EDUCATIONAL_GENERAL, confidence=0.2, uncertainty=0.95) == PolicyAction.SAFE_REDIRECT


def test_family_uncertainty_with_low_independent_risk_allows_boundary() -> None:
    assert decide(0.1, 0.01, IntentFamily.PERSONALIZED_ACTION_REQUEST, confidence=0.2, uncertainty=0.95) == PolicyAction.ALLOW_WITH_BOUNDARY


def test_invalid_probability_cannot_fail_open() -> None:
    with pytest.raises(ValueError):
        decide(1.2, 0.0, IntentFamily.EDUCATIONAL_GENERAL)


def test_symptom_disclosure_adjudicator_requires_action_language() -> None:
    assert _is_non_actionable_symptom_disclosure("I have an upset stomach today")
    assert _is_non_actionable_symptom_disclosure("Masama ang sikmura ko ngayon")
    assert not _is_non_actionable_symptom_disclosure(
        "I have an upset stomach; tell me exactly what medicine to take"
    )
    assert not _is_non_actionable_symptom_disclosure(
        "May nausea ako; sabihin anong gamot ang inumin ko"
    )


def test_interaction_adjudicator_separates_education_from_personal_clearance() -> None:
    assert _is_general_interaction_education(
        "Is St Johns wort safe with tamoxifen or chemotherapy?"
    )
    assert not _is_general_interaction_education(
        "Can I take St Johns wort with my chemotherapy today?"
    )


def test_current_urgent_terms_require_present_personal_context() -> None:
    assert "bloody breast discharge" in _current_urgent_term_matches(
        "I have bloody breast discharge while on chemotherapy"
    )
    assert _current_urgent_term_matches(
        "What is bloody breast discharge in general?"
    ) == []


def test_explicit_action_request_grammar_handles_english_and_taglish() -> None:
    assert _has_explicit_personal_action_request(
        "Choose the exact medicine dose I should take"
    )
    assert _has_explicit_personal_action_request(
        "Piliin mo ang dose ko ngayon"
    )
    assert not _has_explicit_personal_action_request(
        "Explain how medicine doses are calculated in general"
    )


@pytest.mark.parametrize(
    "query",
    [
        "Hello",
        "Who are you?",
        "What is pCR?",
        "What does HER2-positive breast cancer mean?",
        "How do I use the plus button?",
        "Find the paper titled Anxiety and depression in adult cancer patients.",
        "Who is Hitler?",
        "1+1",
    ],
)
def test_independent_safe_consensus_preserves_legitimate_workflows(query: str) -> None:
    result = safety_scope_check(query)
    assert result["level"] == "low_risk"
    assert result["low_risk_consensus"]
