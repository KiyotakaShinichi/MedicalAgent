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
    "what dose",
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
)

DIAGNOSTIC_TERMS: tuple[str, ...] = (
    "do i have cancer",
    "do i have breast cancer",
    "do i have brca",
    "do i have brca1",
    "do i have brca2",
    "will i get cancer",
    "will my relatives get cancer",
    "will my family get cancer",
    "is it metastatic",
    "do i have metastatic",
    "do i have metastatic disease",
    "do i have metastasis",
    "am i cancer free",
    "is my cancer gone",
    "does that mean my cancer came back",
    "does this mean my cancer came back",
    "does that mean recurrence",
    "does this mean recurrence",
    "diagnose me",
    "tell me if i have cancer",
    "tell me whether i have cancer",
    "tell me if this is cancer",
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
    "prognosis ko",
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


def safety_scope_check(query: str, urgent_flags: Sequence[str] | None = None) -> dict[str, Any]:
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
        }
    if any(term in haystack for term in DECISION_TERMS):
        return {
            "level": "high_risk",
            "scope": "treatment_decision_request",
            "cache_allowed": False,
            "message": "Treatment decision wording detected; assistant must not recommend medication or treatment changes.",
        }
    if any(term in haystack for term in DIAGNOSTIC_TERMS):
        return {
            "level": "high_risk",
            "scope": "diagnosis_or_outcome_claim",
            "cache_allowed": False,
            "message": "Diagnosis/outcome confirmation wording detected; assistant must not confirm disease state.",
        }
    return {
        "level": "low_risk",
        "scope": "education_or_tracking",
        "cache_allowed": True,
        "message": "Low-risk educational or portal-support query.",
    }


__all__ = [
    "DECISION_TERMS",
    "DIAGNOSTIC_TERMS",
    "URGENT_TERMS",
    "safety_scope_check",
]
