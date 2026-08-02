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
    normalize_security_text,
)
from backend.services.unsafe_intent_context import (
    classify_unsafe_intent_with_context,
)


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
    lower = query.lower()
    normalized = normalize_security_text(query)
    haystack = f"{normalized} {lower}"
    urgent_flags = list(urgent_flags or [])
    medical_danger = detect_multilingual_medical_danger(query)

    if (
        urgent_flags
        or medical_danger["detected"]
        or any(term in haystack for term in URGENT_TERMS)
    ):
        return {
            "level": "high_risk",
            "scope": "urgent_or_safety_related",
            "cache_allowed": False,
            "message": "Urgent or safety-related wording detected; answer must route toward clinician/emergency review.",
            "matched_safety_terms": sorted(set(urgent_flags + medical_danger.get("matches", [])))[:10],
            "safety_source": "deterministic",
        }
    if any(term in haystack for term in DECISION_TERMS):
        return {
            "level": "high_risk",
            "scope": "treatment_decision_request",
            "cache_allowed": False,
            "message": "Treatment decision wording detected; assistant must not recommend medication or treatment changes.",
            "safety_source": "deterministic",
        }
    if any(term in haystack for term in DIAGNOSTIC_TERMS):
        return {
            "level": "high_risk",
            "scope": "diagnosis_or_outcome_claim",
            "cache_allowed": False,
            "message": "Diagnosis/outcome confirmation wording detected; assistant must not confirm disease state.",
            "safety_source": "deterministic",
        }
    semantic = classify_unsafe_intent_with_context(
        query,
        tuple(previous_user_messages or ()),
    )
    if (
        semantic.get("is_unsafe")
        and not semantic.get("borderline")
        and float(semantic.get("confidence") or 0.0) >= 0.62
    ):
        return {
            "level": "high_risk",
            "scope": semantic.get("scope") or "diagnosis_or_outcome_claim",
            "cache_allowed": False,
            "message": semantic.get("safe_template") or "Unsafe medical or privacy intent detected; route to safe refusal or review.",
            "safety_source": semantic.get("safety_source") or "semantic_classifier",
            "unsafe_intent_family": semantic.get("unsafe_intent_family"),
            "unsafe_intent_confidence": semantic.get("unsafe_intent_confidence"),
            "over_refusal_risk_flag": semantic.get("over_refusal_risk_flag"),
            "context_reused": semantic.get("context_reused", False),
            "context_turn_count": semantic.get("context_turn_count", 0),
        }
    if (
        semantic.get("family") == "none"
        and semantic.get("safety_source") == "safe_boundary_request"
    ):
        return {
            "level": "low_risk",
            "scope": "education_or_tracking",
            "cache_allowed": True,
            "message": (
                "Explicit prevention, consent, redaction, or non-executing "
                "education request."
            ),
            "safety_source": "safe_boundary_request",
            "safe_boundary_request": True,
            "context_reused": semantic.get("context_reused", False),
            "context_turn_count": semantic.get("context_turn_count", 0),
        }
    return {
        "level": "low_risk",
        "scope": "education_or_tracking",
        "cache_allowed": True,
        "message": "Low-risk educational or portal-support query.",
        "safety_source": "deterministic",
    }


__all__ = [
    "DECISION_TERMS",
    "DIAGNOSTIC_TERMS",
    "URGENT_TERMS",
    "safety_scope_check",
]
