"""Compound-intent routing for the patient chat agent.

The single-intent router (``agent_intent_router.route_intent``) collapses
a message to one bucket.  That is the right primitive for the
downstream pipeline (cache / RAG mode / safety branch), but it loses
information when a message carries **more than one** real intention.

Canonical example
~~~~~~~~~~~~~~~~~

    user> "Hi, can you log my symptoms?"

Single-intent routing snaps to ``conversation`` (because "hi" matches
the greeting opener) and the actual tool-request part of the message
is dropped.  This module returns a SEGMENT LIST instead:

    [
      {"intent": "conversation",          "span": "hi",
       "kind": "casual_opener"},
      {"intent": "data_entry_intention",  "span": "log my symptoms",
       "kind": "tool_request",
       "tool_targets": ["save_symptom"]},
    ]

Plus a structured summary the chat layer can use to compose a richer
response (acknowledge the greeting + surface the tool capability).

Claim boundary
~~~~~~~~~~~~~~
This is intent enrichment, not safety policy.  Safety boundary
detection (``agent_safety.safety_scope_check``) and the input
guardrail (``agent_input_gate.input_guardrail_check``) still run first
and ALWAYS win when triggered — a compound message that includes
"should i stop chemo" still routes to ``treatment_decision_boundary``
regardless of how many casual openers it has.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.services.agent_intent_router import (
    EMOTIONAL_TERMS,
    GREETINGS,
    IDENTITY_PATTERNS,
    MEMORY_TERMS,
    PORTAL_TERMS,
    SOCIAL_CHECKIN_PATTERNS,
    TIMELINE_TERMS,
    is_conversation_opening,
    is_identity_or_capability_question,
    is_social_checkin,
)
from backend.services.multilingual_tool_router import (
    normalize_user_text,
    tool_intent_hints_from_text,
)


# ─── Tool-request vocabulary ────────────────────────────────────────────────


# Phrases the user uses to express the INTENT to log data (even when
# the data itself is not in the message — that's why we need this
# separate from the lab / symptom extractors).  Multilingual: English,
# Taglish, light Spanish.
_TOOL_REQUEST_TERMS: tuple[str, ...] = (
    # English — "can you" / "could you" / "please"-style
    "can you log",
    "can you save",
    "can you record",
    "can you track",
    "could you log",
    "could you save",
    "please log",
    "please save",
    "please record",
    "please track",
    "i want to log",
    "i want to save",
    "i want to record",
    "i want to track",
    "i'd like to log",
    "i'd like to save",
    "i would like to log",
    "i would like to save",
    "let me log",
    "let me save",
    "help me log",
    "help me save",
    # English bare imperative (no severity yet)
    "log my symptom",
    "log my symptoms",
    "save my labs",
    "save my cbc",
    "save my mri",
    "save my medication",
    "record my symptom",
    "record my symptoms",
    "track my symptom",
    "track my symptoms",
    "add my symptom",
    "add my symptoms",
    # Taglish — "i-log", "save mo", "i-track", "patulungan mo akong"
    "ilog mo",
    "i-log mo",
    "i log mo",
    "isave mo",
    "i-save mo",
    "i save mo",
    "irecord mo",
    "i-record mo",
    "i record mo",
    "itrack mo",
    "i-track mo",
    "i track mo",
    "patulungan mo akong",
    "patulungan mo ako",
    "tulungan mo akong",
    "tulungan mo ako",
    "tulungan mo akong i-log",
    "puwede mo bang i-log",
    "puwede mo bang isave",
    "pwede mo bang isave",
    "pwede mo bang ilog",
    "pwede mo bang itrack",
    "gusto kong ilog",
    "gusto kong isave",
    "gusto kong itrack",
    # Spanish
    "registrar mi",
    "registrar mis",
    "guardar mi",
    "guardar mis",
    "anotar mi",
    "anotar mis",
)


# Phrases that ASK for explanation / education even when wrapped in
# casual openers.  Used to detect "hi, can you explain pCR?".
_EDUCATION_REQUEST_TERMS: tuple[str, ...] = (
    "explain",
    "what is",
    "what does",
    "tell me about",
    "ano ang",
    "ano ba ang",
    "ano ibig sabihin",
    "what does it mean",
    "que es",
    "que significa",
)


# Phrases the user uses to ask the system "what can you do" / "what
# tools" — these go through the conversation lane but the chat layer
# may want to enumerate the saved actions / available tools.
_CAPABILITY_REQUEST_TERMS: tuple[str, ...] = (
    "what can you do",
    "what can you log",
    "what can i log",
    "what can i save",
    "what tools",
    "ano ang kaya mong gawin",
    "ano ang pwede mong i-log",
    "ano ang pwede mo i-save",
)


# ─── Compound-intent envelope ────────────────────────────────────────────────


@dataclass
class IntentSegment:
    intent: str        # e.g. "conversation", "data_entry_intention", "education_request"
    kind: str          # "casual_opener" | "tool_request" | "education_request" | ...
    span: str          # raw matched substring from the user message
    tool_targets: list[str] = field(default_factory=list)  # save_symptom, save_complete_cbc, ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent":       self.intent,
            "kind":         self.kind,
            "span":         self.span,
            "tool_targets": list(self.tool_targets),
        }


@dataclass
class CompoundIntent:
    segments: list[IntentSegment] = field(default_factory=list)
    primary_intent: str = "general_support"
    is_compound: bool = False
    has_casual_opener: bool = False
    has_tool_request: bool = False
    has_education_request: bool = False
    has_capability_request: bool = False
    tool_request_targets: list[str] = field(default_factory=list)
    suggested_acknowledgment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments":              [s.to_dict() for s in self.segments],
            "primary_intent":        self.primary_intent,
            "is_compound":           self.is_compound,
            "has_casual_opener":     self.has_casual_opener,
            "has_tool_request":      self.has_tool_request,
            "has_education_request": self.has_education_request,
            "has_capability_request": self.has_capability_request,
            "tool_request_targets":  list(self.tool_request_targets),
            "suggested_acknowledgment": self.suggested_acknowledgment,
        }


# ─── Heuristics ──────────────────────────────────────────────────────────────


# Sentence / clause splitter.  We split on punctuation + conjunctions
# so "hi, can you log my symptoms?" yields ["hi", "can you log my symptoms"].
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:[.?!]|\band then\b|\bthen\b|\balso\b|\band\b|\bat\s+saka\b|\bpati\b|,|;)\s*",
    flags=re.IGNORECASE,
)


def _clean_clause(text: str) -> str:
    return text.strip().strip(".,;:?! ").strip()


def _greeting_span(clause: str) -> str | None:
    """Return the matched greeting substring (the bare clause when it
    is a known greeting, or the prefix when the clause starts with a
    greeting word)."""
    stripped = re.sub(r"[^a-z0-9\s]", " ", clause.lower()).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    if stripped in GREETINGS:
        return clause
    for greeting in GREETINGS:
        if stripped.startswith(greeting + " ") or stripped.startswith(greeting):
            return greeting
    if is_conversation_opening(clause.lower()):
        return clause
    return None


def _matched_phrase(text: str, vocabulary: tuple[str, ...]) -> str | None:
    """Return the longest vocabulary phrase that appears in ``text``,
    or None.  Longest-first match avoids ``"log my"`` shadowing
    ``"log my symptoms"``."""
    lower = text.lower()
    for phrase in sorted(vocabulary, key=len, reverse=True):
        if phrase in lower:
            return phrase
    return None


# Generic verb-noun patterns the explicit phrase table doesn't enumerate.
# Each matches a verb+possessive frame: "log my X" / "save my X" /
# "track my X" / "record my X" / Taglish "i-log ko ang X" / Spanish
# "registrar mi X".  The captured noun is the tool target hint.
_GENERIC_TOOL_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, flags=re.IGNORECASE) for p in (
        r"\b(?:log|save|record|track|add)\s+(?:my|the|some|this)\s+([a-z]+)",
        r"\b(?:can|could|please)\s+(?:you\s+)?(?:log|save|record|track|add)\b",
        r"\bi[-\s]?log\s+(?:mo|ko)?\s*(?:ang|yung|the)?\s*",
        r"\bi[-\s]?save\s+(?:mo|ko)?\s*(?:ang|yung|the)?\s*",
        r"\bi[-\s]?record\s+(?:mo|ko)?\s*(?:ang|yung|the)?\s*",
        r"\bi[-\s]?track\s+(?:mo|ko)?\s*(?:ang|yung|the)?\s*",
        r"\bregistrar\s+mi(?:s)?\b",
        r"\bguardar\s+mi(?:s)?\b",
        r"\banotar\s+mi(?:s)?\b",
    )
)


def _matched_generic_tool_request(text: str) -> str | None:
    """Fallback for tool-request detection — returns the matched span
    when one of the verb-noun patterns above hits."""
    for pattern in _GENERIC_TOOL_REQUEST_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


_TOOL_TARGETS_BY_PHRASE: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("symptom", "symptoms", "sintomas", "sintomas ko", "sintoma",
         "sintoma ko", "sintomas mo", "sintomas ko ngayon"),
        "save_symptom",
    ),
    (
        ("cbc", "labs", "wbc", "hemoglobin", "platelets", "lab", "lab values",
         "laboratorio", "globulos blancos", "plaquetas"),
        "save_complete_cbc",
    ),
    (
        ("mri", "imaging", "ct", "ultrasound", "report", "mri report",
         "imaging report", "ct report", "ultrasound report"),
        "save_imaging_report",
    ),
    (
        ("medication", "medicine", "gamot", "supplement", "medicines",
         "medicamento", "medicamentos", "medicación", "medicacion",
         "medicines ko", "gamot ko"),
        "save_medication",
    ),
)


def _tool_targets_from_text(text: str) -> list[str]:
    """Map a tool-request span to one or more concrete tool names."""
    lower = text.lower()
    targets: list[str] = []
    for keywords, tool in _TOOL_TARGETS_BY_PHRASE:
        if any(kw in lower for kw in keywords):
            targets.append(tool)
    # If the user said "log my [something]" but none of the keywords
    # matched, default to save_symptom — symptoms are the most common
    # tool path.
    if not targets:
        hints = tool_intent_hints_from_text(text)
        targets = hints
        if not targets:
            targets = ["save_symptom"]
    # Dedupe while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for t in targets:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


# ─── Public API ──────────────────────────────────────────────────────────────


def detect_compound_intents(message: str) -> CompoundIntent:
    """Return the compound-intent envelope for ``message``.

    The result is always a valid ``CompoundIntent`` (never None).  When
    the message is single-intent, ``is_compound`` is False and the
    segment list has one entry.

    The function is deterministic and does not call any LLM.
    """
    if not message or not message.strip():
        return CompoundIntent(
            segments=[IntentSegment(intent="general_support", kind="empty", span="")],
            primary_intent="general_support",
        )

    normalized = normalize_user_text(message)
    clauses = [_clean_clause(c) for c in _CLAUSE_SPLIT_RE.split(message) if _clean_clause(c)]
    if not clauses:
        clauses = [message.strip()]

    segments: list[IntentSegment] = []
    for clause in clauses:
        clause_lower = clause.lower()
        norm_clause = normalize_user_text(clause)

        # 1) Greeting / identity / social check-in as a CASUAL opener
        greet_span = _greeting_span(clause)
        if greet_span:
            segments.append(IntentSegment(
                intent="conversation",
                kind="casual_opener",
                span=greet_span,
            ))
            # Don't continue — a bare "hi" clause won't also be a tool
            # request, and a "hi, can you log X" sentence is already
            # split by the comma upstream.
            continue
        if is_identity_or_capability_question(clause_lower):
            segments.append(IntentSegment(
                intent="conversation",
                kind="identity_or_capability_question",
                span=clause,
            ))
            continue
        if is_social_checkin(clause_lower):
            segments.append(IntentSegment(
                intent="conversation",
                kind="social_checkin",
                span=clause,
            ))
            continue

        # 2) Explicit tool-request wording (phrase table or generic verb-noun)
        tool_phrase = (
            _matched_phrase(norm_clause, _TOOL_REQUEST_TERMS)
            or _matched_phrase(clause_lower, _TOOL_REQUEST_TERMS)
            or _matched_generic_tool_request(clause_lower)
        )
        if tool_phrase:
            segments.append(IntentSegment(
                intent="data_entry_intention",
                kind="tool_request",
                span=clause,
                tool_targets=_tool_targets_from_text(clause),
            ))
            continue

        # 3) Capability question ("what can you log?")
        capability_phrase = _matched_phrase(clause_lower, _CAPABILITY_REQUEST_TERMS)
        if capability_phrase:
            segments.append(IntentSegment(
                intent="conversation",
                kind="capability_request",
                span=clause,
            ))
            continue

        # 4) Education request ("explain pCR")
        education_phrase = _matched_phrase(clause_lower, _EDUCATION_REQUEST_TERMS)
        if education_phrase:
            segments.append(IntentSegment(
                intent="education",
                kind="education_request",
                span=clause,
            ))
            continue

        # 5) Vocabulary fallbacks — emotional / memory / portal / timeline
        if any(term in clause_lower for term in EMOTIONAL_TERMS):
            segments.append(IntentSegment("emotional_support", "emotional_support", clause))
            continue
        if any(term in clause_lower for term in MEMORY_TERMS):
            segments.append(IntentSegment("patient_memory", "memory_request", clause))
            continue
        if any(term in clause_lower for term in PORTAL_TERMS):
            segments.append(IntentSegment("portal_help", "portal_help", clause))
            continue
        if any(term in clause_lower for term in TIMELINE_TERMS):
            segments.append(IntentSegment("patient_timeline_monitoring", "timeline_request", clause))
            continue

        # 6) Default: general support
        segments.append(IntentSegment("general_support", "general_support", clause))

    return _build_envelope(segments, original_message=message)


def _build_envelope(segments: list[IntentSegment], original_message: str) -> CompoundIntent:
    kinds = {s.kind for s in segments}
    has_casual_opener = bool(
        kinds & {"casual_opener", "social_checkin", "identity_or_capability_question"}
    )
    has_tool_request = "tool_request" in kinds
    has_education_request = "education_request" in kinds
    has_capability_request = "capability_request" in kinds

    tool_request_targets: list[str] = []
    for seg in segments:
        if seg.kind == "tool_request":
            for t in seg.tool_targets:
                if t not in tool_request_targets:
                    tool_request_targets.append(t)

    is_compound = len([s for s in segments if s.kind != "general_support"]) >= 2

    # Primary intent precedence (tool > education > capability >
    # emotional / memory / portal / timeline > conversation > general):
    primary_intent = "general_support"
    if has_tool_request:
        primary_intent = "data_entry_intention"
    elif has_education_request:
        primary_intent = "education"
    elif has_capability_request:
        primary_intent = "conversation"
    elif any(s.intent == "emotional_support" for s in segments):
        primary_intent = "emotional_support"
    elif any(s.intent == "patient_memory" for s in segments):
        primary_intent = "patient_memory"
    elif any(s.intent == "portal_help" for s in segments):
        primary_intent = "portal_help"
    elif any(s.intent == "patient_timeline_monitoring" for s in segments):
        primary_intent = "patient_timeline_monitoring"
    elif has_casual_opener:
        primary_intent = "conversation"

    ack = _suggested_acknowledgment(has_casual_opener, has_tool_request, has_education_request)

    return CompoundIntent(
        segments=segments,
        primary_intent=primary_intent,
        is_compound=is_compound,
        has_casual_opener=has_casual_opener,
        has_tool_request=has_tool_request,
        has_education_request=has_education_request,
        has_capability_request=has_capability_request,
        tool_request_targets=tool_request_targets,
        suggested_acknowledgment=ack,
    )


def _suggested_acknowledgment(casual: bool, tool: bool, education: bool) -> str | None:
    if not casual:
        return None
    if tool and education:
        return "Hi! Happy to help — let me grab what you want to log and explain the term."
    if tool:
        return "Hi! Sure, I can help log that — what's the detail?"
    if education:
        return "Hi! Sure, let me explain."
    return None


__all__ = [
    "IntentSegment",
    "CompoundIntent",
    "detect_compound_intents",
]
