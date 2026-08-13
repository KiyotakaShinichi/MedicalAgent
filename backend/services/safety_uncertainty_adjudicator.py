"""Fail-closed adjudication for uncertain safety-critical intent.

The primary unsafe-intent classifier intentionally remains independently
testable. This module adds a small, orthogonal risk ontology for cases where
that classifier is uncertain: requested action, protected target, and framing.
It does not contain evaluation case strings and never authorizes an unsafe
request. Explicitly bounded educational/process questions remain answerable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from backend.services.agent_text_normalization import normalize_agent_text
from backend.services.unsafe_intent_safe_boundary import (
    looks_like_recording_statement,
    looks_like_safe_boundary_request,
)


@dataclass(frozen=True)
class SafetyUncertaintyDecision:
    requires_safe_route: bool
    family: str
    scope: str
    confidence: float
    reason: str
    matched_action_classes: tuple[str, ...]
    matched_target_families: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ACTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "decision": re.compile(
        r"\b(decid\w*|choos\w*|select\w*|pick\w*|recommend\w*|prescrib\w*|"
        r"decision\w*|start\w*|stop\w*|switch\w*|chang\w*|skip\w*|delay\w*|postpon\w*|"
        r"increase\w*|decrease\w*|reduc\w*|adjust\w*|take\w*|"
        r"pili\w*|itigil\w*|ihinto\w*|palit\w*|taasan\w*|babaan\w*)\b"
    ),
    "conclusion": re.compile(
        r"\b(confirm\w*|prov\w*|diagnos\w*|predict\w*|estimat\w*|"
        r"calculat\w*|conclud\w*|interpret\w*|mean\w*|indicat\w*|"
        r"positive|negative|recurr\w*|metasta\w*|odds?|chances?|"
        r"gaano katagal|bumalik|kumalat|sigurado)\b"
    ),
    "access": re.compile(
        r"\b(show\w*|reveal\w*|export\w*|disclos\w*|retriev\w*|"
        r"open\w*|access\w*|send\w*|share\w*|forward\w*|print\w*|"
        r"pakita\w*|buksan\w*|ipadala\w*)\b"
    ),
    "override": re.compile(
        r"\b(ignore\w*|bypass\w*|overrid\w*|disable\w*|jailbreak\w*|"
        r"new instruction|system prompt|hidden instruction|developer message|"
        r"continue\s+without|proceed\s+without|without\s+(the\s+|that\s+|those\s+)?"
        r"(restriction|boundary|oversight)|"
        r"follow\w*|obey\w*|adopt\w*|act as|"
        r"pretend to be|role play|roleplay)\b"
    ),
    "replacement": re.compile(
        r"\b(replac\w*|instead|substitut\w*|natural only|alone|enough|"
        r"cure\w*|avoid\w*|without prescribed|kapalit|na lang)\b"
    ),
}

_TARGET_PATTERNS: tuple[tuple[str, re.Pattern[str], frozenset[str], str], ...] = (
    (
        "prompt_injection_boundary",
        re.compile(
            r"\b(system prompt|guardrail\w*|safety rule\w*|policy|hidden instruction|"
            r"developer message|tool restriction|administrator instruction|admin instruction|"
            r"administrator\s+(says|directs|orders)|admin\s+(says|directs|orders)|"
            r"safety restriction\w*|system restriction\w*)\b"
        ),
        frozenset({"override", "access"}),
        "security_boundary_request",
    ),
    (
        "authority_boundary_override",
        re.compile(r"\b(clinician|doctor|care team|oncology team|pharmacist|genetic counselor)\b.{0,35}\b(review|approval|permission|oversight)\b|\b(review|approval|permission|oversight)\b.{0,35}\b(unnecessary|not needed|optional|ignore)\b"),
        frozenset({"override", "replacement", "conclusion"}),
        "diagnosis_or_outcome_claim",
    ),
    (
        "cross_patient_exfiltration",
        re.compile(r"\b(other|another|previous|next|someone else|different)\b.{0,35}\b(patient|person|record|chart|profile|result)\b|\b(patient|person)\b.{0,50}\bbefore me\b"),
        frozenset({"access", "override"}),
        "privacy_or_cross_patient_request",
    ),
    (
        "privacy_pii_request",
        re.compile(r"\b(private|confidential|protected|personal|patient)\b.{0,35}\b(record|chart|address|phone|email|identifier|registration|data|details)\b"),
        frozenset({"access", "override"}),
        "privacy_or_cross_patient_request",
    ),
    (
        "dosage_request",
        re.compile(r"\b(dos\w*|milligram\w*|\d+\s*mg|schedule|frequency|tablet\w*|capsule\w*)\b"),
        frozenset({"decision", "conclusion"}),
        "treatment_decision_request",
    ),
    (
        "treatment_change",
        re.compile(r"\b(chemo\w*|radiation|regimen|treatment|therapy|infusion|medication|medicine|tamoxifen|trastuzumab|paclitaxel|gamot)\b"),
        frozenset({"decision", "replacement"}),
        "treatment_decision_request",
    ),
    (
        "supplement_replacement",
        re.compile(r"\b(supplement\w*|herbal|herb\w*|turmeric|curcumin|cannabis|cbd|vitamin\w*|natural remedy|natural cure)\b"),
        frozenset({"decision", "replacement"}),
        "treatment_decision_request",
    ),
    (
        "vus_misinterpretation",
        re.compile(r"\b(vus|variant of uncertain significance|unclear variant|uncertain variant|brca1?|brca2?|palb2|chek2|germline|genetic variant|mutation)\b"),
        frozenset({"conclusion", "decision"}),
        "diagnosis_or_outcome_claim",
    ),
    (
        "tumor_marker_conclusion",
        re.compile(r"\b(tumou?r markers?|ca\s*15[ .-]?3|ca\s*27[ .-]?29|cea|markers?\b.{0,16}\b(rise|rose|rising|increase|increased|elevated|high))\b"),
        frozenset({"conclusion", "decision"}),
        "diagnosis_or_outcome_claim",
    ),
    (
        "prognosis_estimate",
        re.compile(r"\b(prognos\w*|surviv\w*|life expectancy|how long|months? left|years? left|outlook|odds?|chances?)\b"),
        frozenset({"conclusion", "decision"}),
        "diagnosis_or_outcome_claim",
    ),
    (
        "diagnosis_confirmation",
        re.compile(r"\b(cancer|malignan\w*|metasta\w*|recurr\w*|disease progression|lesion|tumou?r)\b"),
        frozenset({"conclusion"}),
        "diagnosis_or_outcome_claim",
    ),
)

_PERSONAL_OR_AUTHORITY = re.compile(
    r"\b(my|mine|me|for me|our|relative\w*|child\w*|family|patient|"
    r"doctor|clinician|oncologist|admin|administrator|researcher|authorized|"
    r"hypothetical\w*|educational purposes?)\b"
)

_RESEARCH_DOCUMENT_REQUEST = re.compile(
    r"\b(find|locate|summari[sz]\w*|explain|describe|what is|what does)\b"
    r".{0,100}\b(paper|article|study|guideline|publication|abstract|research|title)\b"
    r"|\b(paper|article|study|guideline|publication|abstract|research)\b"
    r".{0,100}\b(find|locate|summari[sz]\w*|explain|describe|title)\b"
)

_GENERAL_EDUCATION_REQUEST = re.compile(
    r"^\s*(what (is|are|does|do|can)|how (does|do|can)|why (is|are|does|do|might)|"
    r"explain|describe|what should patients know)\b"
)

_EVIDENCE_LIMITATION_REQUEST = re.compile(
    r"\b(vus|variant of uncertain significance|tumou?r marker|ca\s*15[ .-]?3|"
    r"ca\s*27[ .-]?29|cea)\b.{0,70}\b(same as|alone|automatically|always|"
    r"by itself)\b.{0,35}\b(positive|pathogenic|cancer|recurrence|progression)\b"
    r"|\b(tumou?r marker|ca\s*15[ .-]?3|ca\s*27[ .-]?29|cea)\b.{0,50}"
    r"\b(not|cannot|can't|does not|doesn't)\b.{0,30}\b(prove|confirm|establish|mean|show)\b"
    r".{0,35}\b(recurrence|progression|metastasis|cancer)\b"
)

_GENERAL_INTERACTION_REQUEST = re.compile(
    r"\b(supplement\w*|herbal|st\.?\s*john'?s wort|turmeric|curcumin|cbd|vitamin\w*)\b"
    r".{0,65}\b(safe with|interact\w* with|interaction\w*)\b"
    r"|\b(safe with|interact\w* with|interaction\w*)\b.{0,65}"
    r"\b(supplement\w*|herbal|st\.?\s*john'?s wort|turmeric|curcumin|cbd|vitamin\w*)\b"
)


def adjudicate_safety_uncertainty(
    text: str,
    semantic_result: Mapping[str, Any] | None = None,
) -> SafetyUncertaintyDecision:
    """Return a conservative route decision without generating an answer."""

    normalized = normalize_agent_text(text)
    semantic = dict(semantic_result or {})
    if not normalized:
        return _safe("empty_input")
    if looks_like_recording_statement(normalized):
        return _safe("recording_statement")
    if looks_like_safe_boundary_request(normalized):
        return _safe("explicit_safe_boundary_request")

    actions = tuple(
        name for name, pattern in _ACTION_PATTERNS.items() if pattern.search(normalized)
    )
    prescriptive_or_access_action = bool(
        {"decision", "access", "override", "replacement"}.intersection(actions)
    )
    hard_directive = re.search(
        r"\b(should|can|may) i\b|\b(for me|my dose|my treatment)\b|"
        r"\b(recommend|choose|pick|prescribe|start|stop|switch|replace)\b.{0,35}\b(for me|i should|patients? should)\b",
        normalized,
    )
    self_personal = re.search(r"\b(i|my|mine|me|for me)\b", normalized)
    if (
        _EVIDENCE_LIMITATION_REQUEST.search(normalized)
        and not {"decision", "access", "override"}.intersection(actions)
    ):
        return _safe("evidence_limitation_education", actions)
    if (
        _GENERAL_INTERACTION_REQUEST.search(normalized)
        and not prescriptive_or_access_action
        and not re.search(r"\b(should|can|may) i\b|\bfor me\b", normalized)
    ):
        return _safe("general_interaction_education", actions)
    if (
        _GENERAL_EDUCATION_REQUEST.search(normalized)
        and not {"access", "override"}.intersection(actions)
        and not hard_directive
        and not self_personal
    ):
        return _safe("general_education_request", actions)
    if (
        _RESEARCH_DOCUMENT_REQUEST.search(normalized)
        and not _PERSONAL_OR_AUTHORITY.search(normalized)
        and set(actions).issubset({"conclusion"})
    ):
        return _safe("general_research_document_request", actions)
    matched_targets: list[str] = []
    selected: tuple[str, str] | None = None
    for family, pattern, allowed_actions, scope in _TARGET_PATTERNS:
        if not pattern.search(normalized):
            continue
        matched_targets.append(family)
        if selected is None and allowed_actions.intersection(actions):
            selected = (family, scope)

    semantic_confidence = float(
        semantic.get("unsafe_intent_confidence")
        or semantic.get("confidence")
        or 0.0
    )
    semantic_family = str(semantic.get("unsafe_intent_family") or semantic.get("family") or "none")
    semantic_unsafe = bool(semantic.get("is_unsafe"))
    semantic_borderline = bool(semantic.get("borderline"))

    if semantic_unsafe and (semantic_borderline or semantic_confidence >= 0.50):
        family = semantic_family if semantic_family != "none" else (matched_targets[0] if matched_targets else "uncertain_unsafe_intent")
        return SafetyUncertaintyDecision(
            requires_safe_route=True,
            family=family,
            scope=str(semantic.get("scope") or _scope_for_family(family)),
            confidence=round(max(0.55, semantic_confidence), 4),
            reason="semantic_unsafe_or_borderline",
            matched_action_classes=actions,
            matched_target_families=tuple(matched_targets),
        )

    if selected is not None:
        family, scope = selected
        confidence = 0.78 if _PERSONAL_OR_AUTHORITY.search(normalized) else 0.68
        return SafetyUncertaintyDecision(
            requires_safe_route=True,
            family=family,
            scope=scope,
            confidence=confidence,
            reason="protected_target_with_requested_action",
            matched_action_classes=actions,
            matched_target_families=tuple(matched_targets),
        )

    personal_prognosis = (
        "prognosis_estimate" in matched_targets
        and re.search(r"\b(my|me|i|mine)\b", normalized)
    )
    if personal_prognosis:
        return SafetyUncertaintyDecision(
            requires_safe_route=True,
            family="prognosis_estimate",
            scope="diagnosis_or_outcome_claim",
            confidence=0.72,
            reason="personal_prognosis_request",
            matched_action_classes=actions,
            matched_target_families=tuple(matched_targets),
        )

    if matched_targets and actions and semantic_confidence >= 0.28:
        family = matched_targets[0]
        return SafetyUncertaintyDecision(
            requires_safe_route=True,
            family=family,
            scope=_scope_for_family(family),
            confidence=round(max(0.58, semantic_confidence), 4),
            reason="classifier_uncertainty_with_risk_slots",
            matched_action_classes=actions,
            matched_target_families=tuple(matched_targets),
        )

    return _safe("no_composed_safety_risk", actions, tuple(matched_targets))


def _scope_for_family(family: str) -> str:
    if family in {"prompt_injection_boundary"}:
        return "security_boundary_request"
    if family in {"privacy_pii_request", "cross_patient_exfiltration"}:
        return "privacy_or_cross_patient_request"
    if family in {"dosage_request", "treatment_change", "supplement_replacement"}:
        return "treatment_decision_request"
    return "diagnosis_or_outcome_claim"


def _safe(
    reason: str,
    actions: tuple[str, ...] = (),
    targets: tuple[str, ...] = (),
) -> SafetyUncertaintyDecision:
    return SafetyUncertaintyDecision(
        requires_safe_route=False,
        family="none",
        scope="education_or_tracking",
        confidence=0.0,
        reason=reason,
        matched_action_classes=actions,
        matched_target_families=targets,
    )


__all__ = ["SafetyUncertaintyDecision", "adjudicate_safety_uncertainty"]
