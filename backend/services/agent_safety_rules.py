"""Deterministic helper rules used by the agent safety orchestrator."""

from __future__ import annotations

import re
from typing import Any

from backend.services.security_guardrails import normalize_security_text
from backend.services.safety_policy_action import PolicyAction
from backend.services.agent_safety_vocab import (
    DECISION_TERMS,
    URGENT_TERMS,
    RESEARCH_AUTHORITY_OVERCLAIM_PATTERN,
    PERSONAL_SUPPLEMENT_SAFETY_PATTERN,
    ACTIVE_TREATMENT_CUE_PATTERN,
    _CROSS_PERSON_DATA_REQUEST_PATTERN,
    _PRIVATE_RECORD_DISCLOSURE_PATTERN,
    _TUMOR_MARKER_CONCLUSION_PATTERN,
    _TAGLISH_DIAGNOSTIC_REQUEST_PATTERN,
    _MULTILINGUAL_TUMOR_MARKER_CONCLUSION_PATTERN,
)


def _is_non_actionable_symptom_disclosure(query: str) -> bool:
    """Distinguish recording a symptom from asking the agent to treat it."""
    normalized = normalize_security_text(query)
    action_cues = (
        "what should i do",
        "what do i do",
        "tell me what to do",
        "tell me exactly",
        "how should i treat",
        "how do i treat",
        "how to treat",
        "treat it myself",
        "manage it alone",
        "stay home",
        "what medicine",
        "which medicine",
        "what medication",
        "which medication",
        "what dose",
        "how much should i take",
        "give me a remedy",
        "anong gagawin",
        "ano ang gagawin",
        "sabihin anong",
        "sabihin kung ano",
        "paano gagamutin",
        "gamutin mag isa",
        "anong gamot",
        "ano ang gamot",
        "gaano karami",
        "bigyan ako ng lunas",
        "bigyan mo ko ng remedy",
        "ako lang ang mag manage",
        "manatili sa bahay",
    )
    return not any(cue in normalized for cue in action_cues)


def _is_clearly_nonurgent_symptom_disclosure(query: str) -> bool:
    """Counter a semantic urgent false positive only for a narrow benign set."""
    normalized = normalize_security_text(query)
    benign_symptom = bool(
        re.search(
            r"\b(?:nausea|nauseous|fatigue|tired|neuropathy|tingling|mouth sores?)\b",
            normalized,
        )
    )
    severe_modifier = bool(
        re.search(
            r"\b(?:severe|unbearable|worst|cannot|can't|faint|fainted|collapse|"
            r"emergency|dying|blood|bleeding|chest pain|shortness of breath)\b",
            normalized,
        )
    )
    return (
        benign_symptom
        and not severe_modifier
        and not _has_explicit_personal_action_request(normalized)
        and not _current_urgent_term_matches(normalized)
    )


def _is_explicit_boundary_education(query: str) -> bool:
    """Recognize explicit education about a safety limit, never an action request."""
    normalized = normalize_security_text(query)
    educational_frame = bool(
        re.search(
            r"\b(?:what does|what is|why can|why is|why does|explain|in general|generally|"
            r"meaning of|limitations? of|stands for|refers to)\b",
            normalized,
        )
    )
    boundary_frame = bool(
        re.search(
            r"\b(?:cannot|can not|does not|doesn't|not by itself|not alone|"
            r"limitation|in general|stands for|refers to|mean in general)\b",
            normalized,
        )
    ) or bool(
        re.search(
            r"\bnot\b.{0,55}\b(?:by itself|alone)\b"
            r"|\b(?:by itself|alone)\b.{0,20}\bnot\b"
            r"|\b(?:cannot|does not|doesn't)\b.{0,55}"
            r"\b(?:prove|confirm|establish|determine)\b",
            normalized,
        )
    )
    personal_action = _has_explicit_personal_action_request(normalized) or bool(
        re.search(
            r"\b(?:stop|start|change|delay|skip|increase|decrease|replace)\b"
            r".{0,45}\b(?:my|mine|ko|akin)\b"
            r"|\b(?:my|mine|ko|akin)\b.{0,45}"
            r"\b(?:stop|start|change|delay|skip|increase|decrease|replace)\b",
            normalized,
        )
    )
    return educational_frame and boundary_frame and not personal_action


def _is_unambiguous_safe_information_request(query: str) -> bool:
    """Identify narrow definition, limitation, and portal-process questions."""
    normalized = normalize_security_text(query)
    if _has_explicit_personal_action_request(normalized) or _current_urgent_term_matches(
        normalized
    ):
        return False
    if RESEARCH_AUTHORITY_OVERCLAIM_PATTERN.search(normalized):
        return False
    if PERSONAL_SUPPLEMENT_SAFETY_PATTERN.search(
        normalized
    ) and ACTIVE_TREATMENT_CUE_PATTERN.search(normalized):
        return False

    personal_conclusion = bool(
        re.search(
            r"\b(?:can|could|may|should|do|does|tell)\s+(?:i|me|my)\b.{0,90}"
            r"\b(?:conclude|assume|confirm|prove|means?|recurrence|progression|"
            r"metastatic|survive|prognosis)\b"
            r"|\b(?:my|mine|for me|ko|akin)\b.{0,80}"
            r"\b(?:recurrence|progression|metastatic|survival|prognosis|dose|dosage)\b",
            normalized,
        )
    )
    if personal_conclusion:
        return False

    if _is_privacy_process_education(normalized):
        return True
    if _is_explicit_boundary_education(normalized):
        return True
    if re.fullmatch(
        r"(?:in general\s+)?what does dose[- ]dense(?:\s+\w+)? mean\??",
        normalized,
    ):
        return True
    if re.search(
        r"\b(?:how is|how are|what privacy|where (?:can|do) i (?:review|find))\b"
        r".{0,70}\b(?:privacy|data protection|privacy settings?)\b",
        normalized,
    ):
        return True
    return False


def _deterministic_high_confidence_boundary(query: str) -> dict[str, Any] | None:
    """Apply a small fail-closed floor for unambiguous policy-bound requests."""
    normalized = normalize_security_text(query)
    privacy_process_education = _is_privacy_process_education(normalized)
    if (
        _CROSS_PERSON_DATA_REQUEST_PATTERN.search(normalized)
        or _PRIVATE_RECORD_DISCLOSURE_PATTERN.search(normalized)
    ) and not privacy_process_education:
        return _deterministic_boundary_envelope(
            scope="privacy_or_security_boundary",
            family="cross_patient_exfiltration",
            intent_family="UNKNOWN_HIGH_RISK",
            message="Another person's private data cannot be retrieved or disclosed.",
        )
    if RESEARCH_AUTHORITY_OVERCLAIM_PATTERN.search(normalized):
        return _deterministic_boundary_envelope(
            scope="diagnosis_or_outcome_claim",
            family="research_authority_overclaim",
            intent_family="UNKNOWN_HIGH_RISK",
            message="Research cannot establish this prototype as clinically safe or validated.",
        )
    if PERSONAL_SUPPLEMENT_SAFETY_PATTERN.search(
        normalized
    ) and ACTIVE_TREATMENT_CUE_PATTERN.search(normalized):
        return _deterministic_boundary_envelope(
            scope="treatment_decision_request",
            family="supplement_safety_request",
            intent_family="PERSONALIZED_INTERACTION_ACTION",
            message="Personal supplement clearance during treatment requires clinician or pharmacist review.",
        )
    if _TUMOR_MARKER_CONCLUSION_PATTERN.search(
        normalized
    ) or _MULTILINGUAL_TUMOR_MARKER_CONCLUSION_PATTERN.search(normalized):
        return _deterministic_boundary_envelope(
            scope="diagnosis_or_outcome_claim",
            family="tumor_marker_conclusion",
            intent_family="TUMOR_MARKER_TREATMENT_DECISION",
            message="A tumor-marker result cannot establish progression or recurrence by itself.",
        )
    if any(normalize_security_text(term) in normalized for term in DECISION_TERMS):
        return _deterministic_boundary_envelope(
            scope="treatment_decision_request",
            family="treatment_change",
            intent_family="TREATMENT_MODIFICATION",
            message="A personalized treatment or dosage decision was requested.",
        )
    diagnostic_request = bool(
        re.search(
            r"\b(?:do|could|might)\s+i\s+(?:have|be)\b.{0,50}"
            r"\b(?:cancer|recurrence|metastatic|metastasis|brca|mutation)\b"
            r"|\b(?:diagnose me|confirm (?:that )?(?:i have|my)|tell me whether i have)\b"
            r"|\b(?:how long do i have|will i survive|my prognosis|survival chances)\b",
            normalized,
        )
        # Same request in Tagalog/Taglish, which has no "do I have" surface form.
        or _TAGLISH_DIAGNOSTIC_REQUEST_PATTERN.search(normalized)
    )
    if diagnostic_request:
        return _deterministic_boundary_envelope(
            scope="diagnosis_or_outcome_claim",
            family="diagnosis_confirmation",
            intent_family="PERSONALIZED_INFORMATION",
            message="A diagnosis, prognosis, or outcome conclusion was requested.",
        )
    return None


def _is_privacy_process_education(query: str) -> bool:
    """Keep consent, authorization, and privacy-process education answerable."""
    normalized = normalize_security_text(query)
    process_cue = bool(
        re.search(
            r"\b(?:consent|authorization|authorisation|permission|privacy|protect|"
            r"privacy policy|security training|verify before|verification)\b",
            normalized,
        )
    )
    disclosure_demand = bool(
        re.search(
            r"\b(?:give|send|reveal|disclose|show)\s+(?:me|us)\b"
            r"|(?:^|[.!?]\s+|\b(?:then|so|now)\s+)"
            r"(?:please\s+)?(?:export|copy|retrieve|fetch|open|access|give|send|"
            r"reveal|disclose|show)\b",
            normalized,
        )
    )
    return process_cue and not disclosure_demand


def _safe_information_envelope() -> dict[str, Any]:
    return {
        "level": "low_risk",
        "scope": "education_or_tracking",
        "cache_allowed": True,
        "message": "High-confidence general education intent; answer within evidence and clinical boundaries.",
        "safety_source": "deterministic_safe_information_request",
        "policy_action": PolicyAction.ALLOW_EDUCATIONAL.value,
        "policy_intent_family": "SAFE_EDUCATION",
        "unsafe_intent_family": "none",
        "unsafe_intent_confidence": 0.0,
        "urgent_probability": 0.0,
        "policy_uncertainty": 0.0,
        "context_reused": False,
        "context_turn_count": 0,
        "safe_boundary_request": True,
        "safety_control_failure": None,
    }


def _deterministic_boundary_envelope(
    *,
    scope: str,
    family: str,
    intent_family: str,
    message: str,
) -> dict[str, Any]:
    return {
        "level": "high_risk",
        "scope": scope,
        "cache_allowed": False,
        "message": message,
        "safety_source": "deterministic_high_confidence_policy_floor",
        "policy_action": PolicyAction.REFUSE_ACTIONABLE.value,
        "policy_intent_family": intent_family,
        "unsafe_intent_family": family,
        "unsafe_intent_confidence": 1.0,
        "urgent_probability": 0.0,
        "policy_uncertainty": 0.0,
        "context_reused": False,
        "context_turn_count": 0,
        "safe_boundary_request": False,
        "safety_control_failure": None,
    }


def _is_general_interaction_education(query: str) -> bool:
    """Permit interaction education while rejecting personal clearance."""
    normalized = normalize_security_text(query)
    personal_action_cues = (
        "can i take",
        "can i use",
        "should i take",
        "should i use",
        "clear it for me",
        "safe for me",
        "with my treatment",
        "with my chemo",
        "start it today",
        "take it today",
        "use it today",
        "replace",
        "instead of",
        "pwede ko bang inumin",
        "puwede ko bang inumin",
        "dapat ko bang inumin",
        "ligtas para sa akin",
        "kasabay ng chemo ko",
        "simulan ko ngayon",
        "palitan",
        "kapalit ng",
    )
    return not any(cue in normalized for cue in personal_action_cues)


def _current_urgent_term_matches(query: str) -> list[str]:
    normalized = normalize_security_text(query)
    current_cues = (
        "i have",
        "i am having",
        "right now",
        "happening now",
        "my ",
        "may ",
        "ako",
        "ko ngayon",
        "nararanasan ko",
        "nangyayari ngayon",
        "tengo",
        "estoy",
        "ahora",
    )
    if not any(cue in normalized for cue in current_cues):
        return []
    return sorted({term for term in URGENT_TERMS if normalize_security_text(term) in normalized})[
        :10
    ]


def _has_explicit_personal_action_request(query: str) -> bool:
    normalized = normalize_security_text(query)
    action = (
        r"\b(?:choose|select|calculate|compute|tell|give|approve|decide|"
        r"piliin|kuwentahin|kwentahin|sabihin|bigyan|aprubahan|magpasya)\b"
    )
    target = (
        r"\b(?:dose|dosage|medicine|medication|treatment|chemotherapy|chemo|"
        r"therapy|gamot|lunas|remedy)\b"
    )
    return bool(
        re.search(action + r".{0,60}" + target, normalized)
        or re.search(target + r".{0,60}" + action, normalized)
        or re.search(
            r"\b(?:should|can|may|how|what|which|dapat|pwede|puwede|paano|ano)\b"
            r".{0,70}\b(?:i|me|my|ko|akin)\b.{0,70}" + target,
            normalized,
        )
        or re.search(
            r"\b(?:i|me|my|ko|akin)\b.{0,70}"
            r"\b(?:should|can|may|how|what|which|dapat|pwede|puwede|paano|ano)\b"
            r".{0,70}" + target,
            normalized,
        )
    )


def _trusted_low_risk_consensus(
    *,
    query: str,
    legacy_metadata: dict[str, Any],
    auxiliary_semantic: Any,
    dep001d_prediction: Any,
) -> str | None:
    """Preserve established safe workflows only when independent signals agree."""
    if auxiliary_semantic is None or bool(legacy_metadata.get("is_unsafe")):
        return None
    if _has_explicit_personal_action_request(query) or _current_urgent_term_matches(query):
        return None
    if getattr(auxiliary_semantic, "status", None) != "safe":
        return None
    unsafe_probability = float(getattr(auxiliary_semantic, "unsafe_probability", 1.0))
    urgent_probability = float(getattr(auxiliary_semantic, "urgent_probability", 1.0))
    unsafe_threshold = float(getattr(auxiliary_semantic, "unsafe_route_threshold", 0.0))
    urgent_threshold = float(getattr(auxiliary_semantic, "urgent_route_threshold", 0.0))
    if unsafe_probability > min(0.05, unsafe_threshold * 0.20):
        return None
    if urgent_probability >= max(1e-6, urgent_threshold * 0.20):
        return None
    if float(getattr(dep001d_prediction, "urgent_probability", 1.0)) >= float(
        getattr(dep001d_prediction, "urgent_independent_threshold", 0.0)
    ):
        return None
    category = str(getattr(auxiliary_semantic, "risk_category", ""))
    allowed_categories = {
        "safe_conversation",
        "safe_clinical_education_direct",
        "safe_educational",
        "safe_emotional_support",
        "safe_out_of_scope",
        "safe_portal_help",
        "safe_research_lookup",
        "safe_symptom_logging",
    }
    return category if category in allowed_categories else None


def _semantic_scope(status: str, category: str) -> str:
    if status == "urgent" or category in {"urgent_deterioration", "emergency_symptoms"}:
        return "urgent_or_safety_related"
    if category in {
        "unsafe_actionable_medical_request",
        "treatment_modification",
        "medication_dose_request",
        "contraindication_interactions",
        "hypothetical_unsafe_request",
        "role_play_unsafe_request",
        "indirect_euphemistic_request",
        "multi_turn_escalation",
    }:
        return "treatment_decision_request"
    return "diagnosis_or_outcome_claim"


def _semantic_message(status: str) -> str:
    if status == "urgent":
        return "Potential urgent wording detected; route to immediate human or emergency review."
    if status == "uncertain":
        return "Safety intent is uncertain; ask a bounded clarification without retrieving or generating medical guidance."
    if status == "fail_closed":
        return "The safety classifier is unavailable; do not continue to retrieval or generation."
    return "Unsafe medical or policy-bypass intent detected; route to refusal or human review."
