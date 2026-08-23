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

import re
from typing import Any, Sequence

from backend.services.security_guardrails import (
    detect_multilingual_medical_danger,
    normalize_security_text,
)
from backend.services.unsafe_intent_context import (
    classify_unsafe_intent_with_context,
)
from backend.services.multilingual_semantic_safety import (
    classify_multilingual_safety,
)
from backend.services.dep001d_semantic_safety import classify_dep001d_safety as classify_dep001b_safety
from backend.services.safety_policy_action import PolicyAction, scope_for_policy_action


# ─── Vocabulary tables ───────────────────────────────────────────────────────
#
# Kept as module-level frozen sets so a future caller can introspect what
# the safety contract actually catches without re-running the agent.

DECISION_TERMS: tuple[str, ...] = (
    "should i stop",
    "should i start",
    "should i change",
    "should i delay",
    "can i stop",
    "can i start",
    "can i change",
    "can i decrease",
    "can i increase",
    "write me a prescription",
    "prescription for",
    "prescribe me",
    "can i delay",
    "do i need to delay",
    # Note: "what dose" alone is too greedy — it false-positives on
    # "what dose-dense AC means".  Specific-phrasing matches keep the
    # treatment-decision intent without flagging the educational form.
    "what dose of",
    "what dose should",
    "what dose do i",
    "dose reduction",
    "usual dose reduction",
    "change my dose",
    "what treatment should i take",
    "should i change chemo",
    "should i change chemotherapy",
    "should i change treatment",
    "should i change surgery",
    "increase my dose",
    "decrease my dose",
    "increase my",
    "decrease my",
    "stop chemo",
    "stop chemotherapy",
    "delay chemo",
    "delay chemotherapy",
    "delay treatment",
    "delay my next chemo",
    "skip chemo",
    "skip treatment",
    "instead of chemotherapy",
    "instead of chemo",
    "replace chemotherapy",
    "replace chemo",
    "replace my chemotherapy",
    "replace my chemo",
    "use turmeric instead",
    "use supplements instead",
    "herbal instead of chemo",
    "supplement instead of chemo",
    "supplements instead of chemo",
    "supplement instead of chemotherapy",
    "supplements instead of chemotherapy",
    "dapat ko bang itigil",
    "pwede ko bang itigil",
    "pwede ko bang ihinto",
    "itigil chemo",
    "ihinto chemo",
    "itigil chemotherapy",
    "itigil yung chemo",
    "itigil ang chemo",
    "palitan dose",
    "taasan dose",
    "babaan dose",
    "baguhin dosage",
    "baguhin gamot",
    "anong dose",
    "anong dosage",
    "mag reseta",
    "ireseta",
    "debo parar",
    "parar la quimioterapia",
    "que dosis",
    "qué dosis",
    "dosis debo cambiar",
    "停止化疗",
    "改變什麼劑量",
    "改变什么剂量",
    "прекратить химиотерапию",
    "какую дозу",
    "изменить дозу",
)

DIAGNOSTIC_TERMS: tuple[str, ...] = (
    "do i have cancer",
    "do i have breast cancer",
    "do i have brca",
    "do i have brca1",
    "do i have brca2",
    "is this lump cancer",
    "is this cancer",
    "tell me if this is cancer",
    "tell me whether i have breast cancer",
    "based on my mri",
    "lesion in my mri malignant",
    "is the lesion",
    "malignant",
    "biopsy result means cancer",
    "result means cancer",
    "vus means positive",
    "vus mean positive",
    "vus positive",
    "vus in brca mean positive",
    "vus the same as positive",
    "variant of uncertain significance mean cancer",
    "vus means i have the mutation",
    "vus am i positive",
    "positive for cancer risk",
    "vus sa brca positive",
    "brca vus means positive",
    "positive na ba",
    "genetic mutation that means cancer",
    "brca1 mean i will definitely get cancer",
    "will i get cancer",
    "will my children get cancer",
    "will my relatives get cancer",
    "will my family get cancer",
    "confirm that i have hereditary breast cancer",
    "chek2 mean i have cancer",
    "germline result means cancer",
    "hereditary breast cancer",
    "is it metastatic",
    "has my cancer spread",
    "cancer spread to my bones",
    "do i have metastatic",
    "do i have metastatic disease",
    "do i have metastasis",
    "am i cancer free",
    "is my cancer gone",
    "does that mean my cancer came back",
    "does this mean my cancer came back",
    "does that mean recurrence",
    "does this mean recurrence",
    "confirm that my recurrence is back",
    "recurrence is back",
    "recurrence na ba",
    "ca 15-3 mataas",
    "ca 15-3 result proof",
    "rising cea proof",
    "proof of metastasis",
    "ca 27 29 is elevated",
    "ca 27.29 is elevated",
    "tumor marker high",
    "cancer is back",
    "diagnose me",
    "tell me if i have cancer",
    "tell me whether i have cancer",
    "will i survive",
    "will i beat",
    "how long do i have",
    "my prognosis",
    "survival rate",
    "survival chances",
    "may cancer ba ako",
    "metastatic ba",
    "meron na ba akong metastatic",
    "may metastatic ba",
    "kumalat na ba",
    "bumalik na ba",
    "ibig sabihin ba bumalik",
    "gumaling na ba ako",
    "cancer free na ba ako",
    "diagnose mo ako",
    "gaano katagal buhay ko",
    "mabubuhay ba ako",
    "ilang buwan",
    "buwan na lang",
    "prognosis ko",
    "confirma recurrencia",
    "alto confirma recurrencia",
    "cuanto me queda",
    "复发",
    "证明复发",
    "сколько мне осталось жить",
)

URGENT_TERMS: tuple[str, ...] = (
    "fever",
    "chest pain",
    "cannot breathe",
    "shortness of breath",
    "uncontrolled bleeding",
    "heavy bleeding",
    "bleeding now",
    "actively bleeding",
    "cannot stop bleeding",
    "bloody discharge",
    "bloody breast discharge",
    "blood discharge",
    "blood breast discharge",
    "suicidal",
    "self harm",
    "lagnat",
    "nilalagnat",
)

RESEARCH_AUTHORITY_OVERCLAIM_PATTERN = re.compile(
    r"\b(papers?|research|stud(?:y|ies)|trials?|literature|evidence)\b"
    r".{0,100}\b(prove|proves|proved|confirm|confirms|validate|validates|guarantee|establish)\b"
    r".{0,100}\b(this|our|the)\s+(app|prototype|agent|assistant|system|platform|portal)\b"
    r"|\b(this|our|the)\s+(app|prototype|agent|assistant|system|platform|portal)\b"
    r".{0,100}\b(prove|proves|proved|confirm|confirms|validate|validates|guarantee|establish)\b"
    r".{0,100}\b(safe|clinical|real patients?|patient care|benefit|ready)\b",
    re.IGNORECASE,
)

PERSONAL_SUPPLEMENT_SAFETY_PATTERN = re.compile(
    r"\b(safe|safely|okay|ok|pwede|puwede|ligtas)\b.{0,55}"
    r"\b(herb|herbal|supplement|vitamin|turmeric|cannabis|cbd|natural product)\b"
    r"|\b(herb|herbal|supplement|vitamin|turmeric|cannabis|cbd|natural product)\b"
    r".{0,55}\b(safe|safely|okay|ok|pwede|puwede|ligtas)\b",
    re.IGNORECASE,
)

ACTIVE_TREATMENT_CUE_PATTERN = re.compile(
    r"\b(during|while|habang|with|kasabay|on)\b.{0,35}"
    r"\b(chemo|chemotherapy|radiation|treatment|therapy|tamoxifen|trastuzumab|paclitaxel)\b",
    re.IGNORECASE,
)

_CROSS_PERSON_DATA_REQUEST_PATTERN = re.compile(
    r"\b(reveal|show|give|provide|export|open|retrieve|fetch|disclose)\b"
    r".{0,100}\b(contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r".{0,100}\b(other|another|previous|prior|next|person before me|person after me)\b"
    r"|\b(other|another|previous|prior|next|person before me|person after me)\b"
    r".{0,100}\b(person|patient|portal user|account holder)\b"
    r".{0,100}\b(contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r"|\bperson\b.{0,80}\b(?:used|accessed|visited)\b.{0,40}\bportal\b"
    r".{0,50}\b(?:before|after)\s+me\b",
    re.IGNORECASE,
)

_PRIVATE_RECORD_DISCLOSURE_PATTERN = re.compile(
    r"\b(?:disclose|reveal|show|send|give|provide|export|copy|read|recite|"
    r"retrieve|fetch|open|access)\b.{0,90}"
    r"\b(?:private|confidential|patient|person(?:'s)?)\b.{0,55}"
    r"\b(?:contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r"|\b(?:private|confidential|patient|person(?:'s)?)\b.{0,55}"
    r"\b(?:contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r".{0,90}\b(?:disclose|reveal|show|send|give|provide|export|copy|read|"
    r"recite|retrieve|fetch|open|access)\b",
    re.IGNORECASE,
)

_TUMOR_MARKER_CONCLUSION_PATTERN = re.compile(
    r"\b(?:tumou?r markers?|markers?|cea|ca\s*15-?3|ca\s*27\.?29)\b"
    r".{0,100}\b(?:conclude|assume|settle|prove|confirm|establish|means?)\b"
    r".{0,80}\b(?:progress(?:ed|ion)?|recurrence|recurred|metastasis|"
    r"metastatic|spread|treatment failed|therapy failed)\b"
    r"|\b(?:conclude|assume|settle|prove|confirm|establish)\b.{0,80}"
    r"\b(?:progress(?:ed|ion)?|recurrence|recurred|metastasis|metastatic|spread)\b"
    r".{0,100}\b(?:tumou?r markers?|markers?|cea|ca\s*15-?3|ca\s*27\.?29)\b",
    re.IGNORECASE,
)

# Tagalog/Taglish "do I have <disease>?" requests.
#
# The English diagnostic-request rule below keys on the surface form
# "do/could/might I have", which Tagalog cannot produce: possession is
# existential (`meron/mayroon/may ... ako`) and the question is marked by the
# particle `ba`. So "Base sa symptoms ko, meron na ba akong metastatic?" — the
# exact translation of a request the English rule stops — matched nothing and
# was routed as `general_support` / `education_or_tracking`, i.e. a diagnosis
# request from a Taglish speaker crossed the safety boundary that the same
# request in English does not.
#
# Two orders are accepted because Tagalog permits both:
#   particle-first  "meron na ba akong metastatic"
#   particle-last   "kanser na ba ako", "may bukol ba ako"
#
# The interrogative particle `ba` and a first-person/subject marker are both
# required, so purely educational phrasings such as "Ano ba yung metastatic?"
# ("what is metastatic?") stay educational: they carry no possession claim
# about the speaker. Over-triggering here would make the assistant refuse
# ordinary Tagalog health education, which is its own harm.
_TAGLISH_DIAGNOSTIC_REQUEST_PATTERN = re.compile(
    r"(?:\b(?:meron|mayroon|may)\b(?:\s+na)?\s+\bba\b.{0,60}?"
    r"\b(?:cancer|kanser|recurrence|metasta\w*|bukol|tumor|brca|mutation)\b"
    r"|\b(?:meron|mayroon|may)\b.{0,40}?"
    r"\b(?:cancer|kanser|recurrence|metasta\w*|bukol|tumor)\b\s*"
    r"(?:na\s+)?\bba\s+(?:ako|ito|ang)\b"
    r"|\b(?:cancer|kanser|recurrence|metasta\w*|bukol|tumor)\b\s*"
    r"(?:na\s+)?\bba\s+(?:ako|ito)\b)",
    re.IGNORECASE,
)

_MULTILINGUAL_TUMOR_MARKER_CONCLUSION_PATTERN = re.compile(
    r"(?:(?<![a-z0-9])(?:tumou?r markers?|markers?|cea|"
    r"ca(?:\s*is\s*e|is\s*e|\s*2t\s*29|2t\s*29)|"
    r"marcadores? tumorales?)(?![a-z0-9])|(?:肿瘤标志物|癌胚抗原))"
    r".{0,100}(?:(?:\b(?:confirma?|demuestra|prueba|significa)\b)|"
    r"(?:证明|证实|表明|意味着))"
    r".{0,80}(?:(?:\b(?:recurrenci[ae]|progresi[oó]n|met[aá]stasis)\b)|"
    r"(?:复发|进展|转移))",
    re.IGNORECASE,
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
    if action in {
        PolicyAction.SAFE_REDIRECT,
        PolicyAction.REFUSE_ACTIONABLE,
        PolicyAction.URGENT_ESCALATION,
    } and low_risk_consensus:
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
    if (
        action == PolicyAction.URGENT_ESCALATION
        and _is_clearly_nonurgent_symptom_disclosure(query)
    ):
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
        and prediction.intent_family in {
            "PERSONALIZED_ACTION_REQUEST", "TREATMENT_MODIFICATION",
            "MEDICATION_DOSING", "ACTIVE_SYMPTOM_MANAGEMENT",
            "PERSONALIZED_INTERACTION_ACTION", "TUMOR_MARKER_TREATMENT_DECISION",
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
        message = "Potential urgent presentation detected; route to immediate human or emergency review."
    elif action == PolicyAction.REFUSE_ACTIONABLE:
        message = "A personalized medical action was requested; do not provide diagnosis, dosing, or treatment direction."
    elif action == PolicyAction.SAFE_REDIRECT:
        message = "The request requires a bounded clarification, refusal, or human-review redirect."
    elif action == PolicyAction.FAIL_CLOSED:
        message = "The safety classifier is unavailable; do not continue to retrieval or generation."
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


def _is_non_actionable_symptom_disclosure(query: str) -> bool:
    """Distinguish recording a symptom from asking the agent to treat it."""
    normalized = normalize_security_text(query)
    action_cues = (
        "what should i do", "what do i do", "tell me what to do",
        "tell me exactly", "how should i treat", "how do i treat",
        "how to treat", "treat it myself", "manage it alone", "stay home",
        "what medicine", "which medicine", "what medication", "which medication",
        "what dose", "how much should i take", "give me a remedy",
        "anong gagawin", "ano ang gagawin", "sabihin anong", "sabihin kung ano",
        "paano gagamutin", "gamutin mag isa", "anong gamot", "ano ang gamot",
        "gaano karami", "bigyan ako ng lunas", "bigyan mo ko ng remedy",
        "ako lang ang mag manage", "manatili sa bahay",
    )
    return not any(cue in normalized for cue in action_cues)


def _is_clearly_nonurgent_symptom_disclosure(query: str) -> bool:
    """Counter a semantic urgent false positive only for a narrow benign set."""
    normalized = normalize_security_text(query)
    benign_symptom = bool(re.search(
        r"\b(?:nausea|nauseous|fatigue|tired|neuropathy|tingling|mouth sores?)\b",
        normalized,
    ))
    severe_modifier = bool(re.search(
        r"\b(?:severe|unbearable|worst|cannot|can't|faint|fainted|collapse|"
        r"emergency|dying|blood|bleeding|chest pain|shortness of breath)\b",
        normalized,
    ))
    return (
        benign_symptom
        and not severe_modifier
        and not _has_explicit_personal_action_request(normalized)
        and not _current_urgent_term_matches(normalized)
    )


def _is_explicit_boundary_education(query: str) -> bool:
    """Recognize explicit education about a safety limit, never an action request."""
    normalized = normalize_security_text(query)
    educational_frame = bool(re.search(
        r"\b(?:what does|what is|why can|why is|why does|explain|in general|generally|"
        r"meaning of|limitations? of|stands for|refers to)\b",
        normalized,
    ))
    boundary_frame = bool(re.search(
        r"\b(?:cannot|can not|does not|doesn't|not by itself|not alone|"
        r"limitation|in general|stands for|refers to|mean in general)\b",
        normalized,
    )) or bool(re.search(
        r"\bnot\b.{0,55}\b(?:by itself|alone)\b"
        r"|\b(?:by itself|alone)\b.{0,20}\bnot\b"
        r"|\b(?:cannot|does not|doesn't)\b.{0,55}"
        r"\b(?:prove|confirm|establish|determine)\b",
        normalized,
    ))
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
    if _has_explicit_personal_action_request(normalized) or _current_urgent_term_matches(normalized):
        return False
    if RESEARCH_AUTHORITY_OVERCLAIM_PATTERN.search(normalized):
        return False
    if PERSONAL_SUPPLEMENT_SAFETY_PATTERN.search(normalized) and ACTIVE_TREATMENT_CUE_PATTERN.search(normalized):
        return False

    personal_conclusion = bool(re.search(
        r"\b(?:can|could|may|should|do|does|tell)\s+(?:i|me|my)\b.{0,90}"
        r"\b(?:conclude|assume|confirm|prove|means?|recurrence|progression|"
        r"metastatic|survive|prognosis)\b"
        r"|\b(?:my|mine|for me|ko|akin)\b.{0,80}"
        r"\b(?:recurrence|progression|metastatic|survival|prognosis|dose|dosage)\b",
        normalized,
    ))
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
        (
            _CROSS_PERSON_DATA_REQUEST_PATTERN.search(normalized)
            or _PRIVATE_RECORD_DISCLOSURE_PATTERN.search(normalized)
        )
        and not privacy_process_education
    ):
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
    if PERSONAL_SUPPLEMENT_SAFETY_PATTERN.search(normalized) and ACTIVE_TREATMENT_CUE_PATTERN.search(normalized):
        return _deterministic_boundary_envelope(
            scope="treatment_decision_request",
            family="supplement_safety_request",
            intent_family="PERSONALIZED_INTERACTION_ACTION",
            message="Personal supplement clearance during treatment requires clinician or pharmacist review.",
        )
    if (
        _TUMOR_MARKER_CONCLUSION_PATTERN.search(normalized)
        or _MULTILINGUAL_TUMOR_MARKER_CONCLUSION_PATTERN.search(normalized)
    ):
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
    process_cue = bool(re.search(
        r"\b(?:consent|authorization|authorisation|permission|privacy|protect|"
        r"privacy policy|security training|verify before|verification)\b",
        normalized,
    ))
    disclosure_demand = bool(re.search(
        r"\b(?:give|send|reveal|disclose|show)\s+(?:me|us)\b"
        r"|(?:^|[.!?]\s+|\b(?:then|so|now)\s+)"
        r"(?:please\s+)?(?:export|copy|retrieve|fetch|open|access|give|send|"
        r"reveal|disclose|show)\b",
        normalized,
    ))
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
        "can i take", "can i use", "should i take", "should i use",
        "clear it for me", "safe for me", "with my treatment", "with my chemo",
        "start it today", "take it today", "use it today", "replace",
        "instead of", "pwede ko bang inumin", "puwede ko bang inumin",
        "dapat ko bang inumin", "ligtas para sa akin", "kasabay ng chemo ko",
        "simulan ko ngayon", "palitan", "kapalit ng",
    )
    return not any(cue in normalized for cue in personal_action_cues)


def _current_urgent_term_matches(query: str) -> list[str]:
    normalized = normalize_security_text(query)
    current_cues = (
        "i have", "i am having", "right now", "happening now", "my ",
        "may ", "ako", "ko ngayon", "nararanasan ko", "nangyayari ngayon",
        "tengo", "estoy", "ahora",
    )
    if not any(cue in normalized for cue in current_cues):
        return []
    return sorted({term for term in URGENT_TERMS if normalize_security_text(term) in normalized})[:10]


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


__all__ = [
    "DECISION_TERMS",
    "DIAGNOSTIC_TERMS",
    "URGENT_TERMS",
    "safety_scope_check",
]
