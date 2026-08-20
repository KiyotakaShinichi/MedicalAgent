"""Deterministic DEP-001B policy action selection.

The learned heads estimate risk and intent. This module alone selects the
allowed system action. It contains no generation logic and cannot authorize
diagnosis, treatment, dosing, prognosis, or other clinical decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class PolicyAction(str, Enum):
    ALLOW_EDUCATIONAL = "ALLOW_EDUCATIONAL"
    ALLOW_WITH_BOUNDARY = "ALLOW_WITH_BOUNDARY"
    SAFE_REDIRECT = "SAFE_REDIRECT"
    REFUSE_ACTIONABLE = "REFUSE_ACTIONABLE"
    URGENT_ESCALATION = "URGENT_ESCALATION"
    FAIL_CLOSED = "FAIL_CLOSED"


class IntentFamily(str, Enum):
    EDUCATIONAL_GENERAL = "EDUCATIONAL_GENERAL"
    PERSONALIZED_INFORMATION = "PERSONALIZED_INFORMATION"
    PERSONALIZED_ACTION_REQUEST = "PERSONALIZED_ACTION_REQUEST"
    TREATMENT_MODIFICATION = "TREATMENT_MODIFICATION"
    MEDICATION_DOSING = "MEDICATION_DOSING"
    SYMPTOM_EDUCATION = "SYMPTOM_EDUCATION"
    ACTIVE_SYMPTOM_MANAGEMENT = "ACTIVE_SYMPTOM_MANAGEMENT"
    URGENT_PRESENTATION = "URGENT_PRESENTATION"
    INTERACTION_EDUCATION = "INTERACTION_EDUCATION"
    PERSONALIZED_INTERACTION_ACTION = "PERSONALIZED_INTERACTION_ACTION"
    TUMOR_MARKER_EDUCATION = "TUMOR_MARKER_EDUCATION"
    TUMOR_MARKER_TREATMENT_DECISION = "TUMOR_MARKER_TREATMENT_DECISION"
    UNKNOWN_HIGH_RISK = "UNKNOWN_HIGH_RISK"


EDUCATIONAL_FAMILIES = frozenset({
    IntentFamily.EDUCATIONAL_GENERAL.value,
    IntentFamily.SYMPTOM_EDUCATION.value,
    IntentFamily.INTERACTION_EDUCATION.value,
    IntentFamily.TUMOR_MARKER_EDUCATION.value,
})
BOUNDED_INFORMATION_FAMILIES = frozenset({IntentFamily.PERSONALIZED_INFORMATION.value})
ACTIONABLE_FAMILIES = frozenset({
    IntentFamily.PERSONALIZED_ACTION_REQUEST.value,
    IntentFamily.TREATMENT_MODIFICATION.value,
    IntentFamily.MEDICATION_DOSING.value,
    IntentFamily.ACTIVE_SYMPTOM_MANAGEMENT.value,
    IntentFamily.PERSONALIZED_INTERACTION_ACTION.value,
    IntentFamily.TUMOR_MARKER_TREATMENT_DECISION.value,
})


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    reason: str
    intent_family: str
    unsafe_probability: float
    urgent_probability: float
    intent_family_confidence: float
    uncertainty: float
    disagreement: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


def select_policy_action(
    *,
    unsafe_probability: float,
    urgent_probability: float,
    intent_family: str,
    intent_family_confidence: float,
    uncertainty: float,
    thresholds: Mapping[str, Any],
    failure_reason: str | None = None,
) -> PolicyDecision:
    """Select one policy action from independent calibrated signals."""
    unsafe = _probability(unsafe_probability)
    urgent = _probability(urgent_probability)
    family_confidence = _probability(intent_family_confidence)
    uncertainty_value = _probability(uncertainty)
    family = str(intent_family or IntentFamily.UNKNOWN_HIGH_RISK.value)
    unsafe_threshold = float(thresholds["unsafe_route_threshold"])
    urgent_threshold = float(thresholds["urgent_route_threshold"])
    urgent_independent_threshold = float(
        thresholds.get("urgent_independent_threshold", urgent_threshold)
    )
    family_threshold = float(thresholds["intent_family_confidence_threshold"])
    uncertainty_threshold = float(thresholds["uncertainty_route_threshold"])

    if failure_reason:
        return _decision(
            PolicyAction.FAIL_CLOSED,
            "safety_signal_failure",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            False,
        )

    urgent_family_vote = (
        family == IntentFamily.URGENT_PRESENTATION.value
        and family_confidence >= float(
            thresholds.get("urgent_family_confidence_threshold", family_threshold)
        )
    )
    if urgent >= urgent_independent_threshold or urgent_family_vote:
        return _decision(
            PolicyAction.URGENT_ESCALATION,
            "independent_urgent_head_or_material_urgent_family",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            False,
        )

    high_confidence_family = family_confidence >= family_threshold
    educational_vote = family in EDUCATIONAL_FAMILIES and high_confidence_family
    bounded_information_vote = family in BOUNDED_INFORMATION_FAMILIES and high_confidence_family
    actionable_vote = family in ACTIONABLE_FAMILIES and high_confidence_family
    unknown_risk_vote = (
        family == IntentFamily.UNKNOWN_HIGH_RISK.value and high_confidence_family
    )
    unsafe_vote = unsafe >= unsafe_threshold
    disagreement = (educational_vote or bounded_information_vote) and unsafe_vote

    if disagreement:
        return _decision(
            PolicyAction.SAFE_REDIRECT,
            "safe_intent_and_unsafe_head_disagree",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            True,
        )
    if educational_vote:
        return _decision(
            PolicyAction.ALLOW_EDUCATIONAL,
            "high_confidence_general_education",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            False,
        )
    if bounded_information_vote and not unsafe_vote:
        return _decision(
            PolicyAction.ALLOW_WITH_BOUNDARY,
            "high_confidence_personalized_information_without_action",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            False,
        )
    # Intent family is an independently trained safety layer. A confident
    # personalized-action classification is material risk even when the
    # binary unsafe head misses it. Safe educational controls constrain the
    # family threshold during calibration.
    if actionable_vote:
        action = (
            PolicyAction.SAFE_REDIRECT
            if family == IntentFamily.ACTIVE_SYMPTOM_MANAGEMENT.value
            else PolicyAction.REFUSE_ACTIONABLE
        )
        return _decision(
            action,
            "material_actionable_intent_family",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            False,
        )
    if unknown_risk_vote or unsafe_vote:
        return _decision(
            PolicyAction.SAFE_REDIRECT,
            "unknown_or_non_actionable_high_risk",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            False,
        )
    if uncertainty_value >= uncertainty_threshold:
        safe_head_consensus = (
            unsafe < unsafe_threshold * 0.50
            and urgent < float(thresholds.get("urgent_family_support_floor", urgent_threshold)) * 0.50
        )
        if safe_head_consensus:
            return _decision(
                PolicyAction.ALLOW_WITH_BOUNDARY,
                "independent_safety_heads_low_family_uncertain",
                family,
                unsafe,
                urgent,
                family_confidence,
                uncertainty_value,
                False,
            )
        return _decision(
            PolicyAction.SAFE_REDIRECT,
            "high_policy_uncertainty",
            family,
            unsafe,
            urgent,
            family_confidence,
            uncertainty_value,
            False,
        )
    return _decision(
        PolicyAction.ALLOW_WITH_BOUNDARY,
        "low_risk_bounded_default",
        family,
        unsafe,
        urgent,
        family_confidence,
        uncertainty_value,
        False,
    )


def scope_for_policy_action(action: PolicyAction, intent_family: str) -> str:
    if action == PolicyAction.URGENT_ESCALATION:
        return "urgent_or_safety_related"
    if action == PolicyAction.FAIL_CLOSED:
        return "safety_control_unavailable"
    if action in {PolicyAction.REFUSE_ACTIONABLE, PolicyAction.SAFE_REDIRECT}:
        if intent_family in {
            IntentFamily.TREATMENT_MODIFICATION.value,
            IntentFamily.MEDICATION_DOSING.value,
            IntentFamily.PERSONALIZED_ACTION_REQUEST.value,
            IntentFamily.ACTIVE_SYMPTOM_MANAGEMENT.value,
            IntentFamily.PERSONALIZED_INTERACTION_ACTION.value,
            IntentFamily.TUMOR_MARKER_TREATMENT_DECISION.value,
        }:
            return "treatment_decision_request"
        return "diagnosis_or_outcome_claim"
    return "education_or_tracking"


def _decision(
    action: PolicyAction,
    reason: str,
    family: str,
    unsafe: float,
    urgent: float,
    family_confidence: float,
    uncertainty: float,
    disagreement: bool,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason=reason,
        intent_family=family,
        unsafe_probability=round(unsafe, 6),
        urgent_probability=round(urgent, 6),
        intent_family_confidence=round(family_confidence, 6),
        uncertainty=round(uncertainty, 6),
        disagreement=disagreement,
    )


def _probability(value: float) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("policy signal must be within [0, 1]")
    return parsed


__all__ = [
    "ACTIONABLE_FAMILIES",
    "BOUNDED_INFORMATION_FAMILIES",
    "EDUCATIONAL_FAMILIES",
    "IntentFamily",
    "PolicyAction",
    "PolicyDecision",
    "scope_for_policy_action",
    "select_policy_action",
]
