"""Intent routing for the patient agent.

Decides one of 11 intents for an inbound query (education, portal_help,
emotional_support, conversation, safety_boundary, treatment_decision_boundary,
data_entry_confirmation, patient_memory, patient_timeline_monitoring,
general_support, security_boundary).

Extracted from ``agent_rag.py`` as part of the god-module split.  The
function is pure aside from the optional LLM consult (which itself
short-circuits when no provider is configured).

Public contract preserved
~~~~~~~~~~~~~~~~~~~~~~~~~
``route_intent`` keeps the exact signature + behavior of the original
in-line implementation.  The deterministic-vs-LLM precedence and the
"hard branches" list (intents the LLM cannot override) are byte-identical
to the prior code so existing tests stay green.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

# Note: ``route_intent_with_local_llm`` is resolved at call time via the
# ``backend.services.agent_rag`` module attribute (see ``_llm_router``
# below).  The indirection preserves the back-compat surface: tests
# monkey-patch ``agent_rag.route_intent_with_local_llm`` to disable the
# LLM router, and that patch must remain authoritative for this module.


# ─── Keyword tables ──────────────────────────────────────────────────────────


MEMORY_TERMS: tuple[str, ...] = (
    "remember", "what did i tell", "what did i say",
    "last message", "previous message", "chat history",
)

EMOTIONAL_TERMS: tuple[str, ...] = (
    "anxious", "worried", "sad", "scared", "depressed", "exhausted",
    "can't keep", "cannot keep", "overwhelmed",
)

PORTAL_TERMS: tuple[str, ...] = (
    "upload", "site", "portal", "dashboard", "where can i", "how do i add",
    "enter my", "where do i enter", "how do i enter", "where do i put",
    "put my results", "where should i",
)

TIMELINE_TERMS: tuple[str, ...] = (
    "last 14", "timeline", "cycle", "toxicity", "score", "my treatment plan",
    "my treatment response", "my treatment results", "working", "progress",
)

# Clinical / educational vocabulary.  Order is irrelevant; any match
# routes the query to ``education``.  Kept as a tuple so reviewers can
# scan the full term inventory.
EDUCATION_TERMS: tuple[str, ...] = (
    "pcr", "response", "mri", "ct", "ultrasound", "imaging", "ascites",
    "cbc", "wbc", "anc", "hemoglobin", "platelets",
    "chemo", "chemotherapy", "treatment", "side effect", "breast cancer",
    "triple-negative", "stage iv", "her2", "er/pr", "er ", "pr ",
    "ki-67", "ki67", "brca", "brca1", "brca2", "palb2", "tp53", "pten",
    "chek2", "atm", "genetic counseling", "genetic test", "germline",
    "somatic", "vus", "variant of uncertain", "multigene panel",
    "ca 15-3", "ca 27.29", "cea", "tumor marker", "biomarker",
    "neutropenia", "neuropathy",
    "paclitaxel", "doxorubicin", "cyclophosphamide", "carboplatin",
    "docetaxel", "trastuzumab", "tamoxifen", "infection risk",
    "supplement", "supplements", "antioxidant", "turmeric", "herbal",
    "herb", "vitamin", "st. john", "st john",
    "acupuncture", "acupressure", "nutrition", "exercise", "yoga", "meditation",
)


# Intents the LLM router is *not* allowed to override the deterministic
# branch on.  These are safety- or determinism-critical: refusing to
# escalate them based on LLM uncertainty is more important than picking
# a slightly better label.
LLM_OVERRIDE_BLOCKED: frozenset[str] = frozenset({
    "safety_boundary",
    "treatment_decision_boundary",
    "data_entry_confirmation",
    "conversation",
    "patient_memory",
    "patient_timeline_monitoring",
    "emotional_support",
    "portal_help",
})


# Allow-list for intents the LLM CAN propose.  An LLM intent outside
# this set is ignored regardless of confidence.
LLM_ALLOWED_INTENTS: frozenset[str] = frozenset({
    "security_boundary",
    "safety_boundary",
    "treatment_decision_boundary",
    "data_entry_confirmation",
    "portal_help",
    "patient_timeline_monitoring",
    "education",
    "emotional_support",
    "general_support",
    "conversation",
    "patient_memory",
})


LLM_CONFIDENCE_FLOOR: float = 0.72


# ─── Conversation detectors ──────────────────────────────────────────────────


GREETINGS: frozenset[str] = frozenset({
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
    "kumusta", "kamusta",
    "salamat", "thanks", "thank you",
})

IDENTITY_PATTERNS: tuple[str, ...] = (
    "who are you", "what are you", "what can you do", "what do you do",
    "how can you help", "help me", "can you help",
    "are you a doctor", "are you ai", "are you an ai",
)

SOCIAL_CHECKIN_PATTERNS: tuple[str, ...] = (
    "how are you", "how are u", "how you doing", "how are you doing",
    "are you ok", "what s up", "whats up",
)


def _clean(lower: str) -> str:
    """Strip non-alphanumeric and collapse whitespace.  Mirrors the
    original inline cleaner used by all three conversation detectors."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lower).strip()
    return re.sub(r"\s+", " ", cleaned)


def is_conversation_opening(lower: str) -> bool:
    cleaned = _clean(lower)
    return cleaned in GREETINGS or cleaned.startswith(("hi ", "hello ", "hey "))


def is_identity_or_capability_question(lower: str) -> bool:
    cleaned = _clean(lower)
    return any(pattern in cleaned for pattern in IDENTITY_PATTERNS)


def is_social_checkin(lower: str) -> bool:
    cleaned = _clean(lower)
    return any(pattern in cleaned for pattern in SOCIAL_CHECKIN_PATTERNS)


# Underscore aliases for backward-compatibility with agent_rag.py's
# internal references.
_is_conversation_opening = is_conversation_opening
_is_identity_or_capability_question = is_identity_or_capability_question
_is_social_checkin = is_social_checkin


# ─── Public API ──────────────────────────────────────────────────────────────


def route_intent(
    query: str,
    actions: Iterable[Mapping[str, Any]] | None = None,
    safety: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the query to one of the 11 supported intents.

    Decision order (deterministic first):
      1. Treatment-decision safety scope → ``treatment_decision_boundary``
      2. Urgent or diagnostic safety scope → ``safety_boundary``
      3. Any tool actions already queued → ``data_entry_confirmation``
      4. Greeting / identity / social check-in → ``conversation``
      5. Memory keyword → ``patient_memory``
      6. Emotional keyword → ``emotional_support``
      7. Portal keyword → ``portal_help``
      8. Timeline keyword → ``patient_timeline_monitoring``
      9. Education vocabulary → ``education``
      10. Default → ``general_support``

    The LLM router is then consulted: when it returns a high-confidence
    intent within the allow-list AND the deterministic branch is *not*
    in the override-blocked set, the LLM's choice wins.  Otherwise the
    deterministic branch is returned.
    """
    lower = query.lower()
    actions = list(actions or [])
    safety = safety or {}

    scope = safety.get("scope")
    if scope == "treatment_decision_request":
        deterministic = "treatment_decision_boundary"
    elif scope in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        deterministic = "safety_boundary"
    elif actions:
        deterministic = "data_entry_confirmation"
    elif is_conversation_opening(lower):
        deterministic = "conversation"
    elif is_identity_or_capability_question(lower):
        deterministic = "conversation"
    elif is_social_checkin(lower):
        deterministic = "conversation"
    elif any(term in lower for term in MEMORY_TERMS):
        deterministic = "patient_memory"
    elif any(term in lower for term in EMOTIONAL_TERMS):
        deterministic = "emotional_support"
    elif any(term in lower for term in PORTAL_TERMS):
        deterministic = "portal_help"
    elif any(term in lower for term in TIMELINE_TERMS):
        deterministic = "patient_timeline_monitoring"
    elif any(term in lower for term in EDUCATION_TERMS):
        deterministic = "education"
    else:
        deterministic = "general_support"

    # Latency optimization: skip the LLM router entirely when the
    # deterministic branch is in LLM_OVERRIDE_BLOCKED.  We would not
    # accept the LLM's vote on those branches anyway, so a 3-second
    # Ollama timeout on every greeting / data-entry / safety message
    # is pure waste.  Tests that monkey-patch
    # agent_rag.route_intent_with_local_llm still work because the
    # path that DOES consult the LLM (open-ended intents like
    # general_support / education) goes through the attribute lookup
    # on agent_rag below.
    if deterministic in LLM_OVERRIDE_BLOCKED:
        return deterministic

    # Resolve route_intent_with_local_llm at call time via agent_rag so the
    # test monkey-patch on agent_rag.route_intent_with_local_llm remains
    # authoritative for this module.
    from backend.services import agent_rag  # local import — avoids cycle at module load
    llm = agent_rag.route_intent_with_local_llm(query, deterministic_intent=deterministic, safety=safety)
    candidate = llm.get("intent")
    confident = (
        llm.get("available")
        and candidate in LLM_ALLOWED_INTENTS
        and float(llm.get("confidence") or 0) >= LLM_CONFIDENCE_FLOOR
    )
    if confident:
        return candidate
    return deterministic


__all__ = [
    "MEMORY_TERMS", "EMOTIONAL_TERMS", "PORTAL_TERMS",
    "TIMELINE_TERMS", "EDUCATION_TERMS",
    "GREETINGS", "IDENTITY_PATTERNS", "SOCIAL_CHECKIN_PATTERNS",
    "LLM_ALLOWED_INTENTS", "LLM_OVERRIDE_BLOCKED", "LLM_CONFIDENCE_FLOOR",
    "route_intent",
    "is_conversation_opening",
    "is_identity_or_capability_question",
    "is_social_checkin",
    "_is_conversation_opening",
    "_is_identity_or_capability_question",
    "_is_social_checkin",
]
