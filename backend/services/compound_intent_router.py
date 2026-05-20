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


# ─── LLM-backed multilingual classifier ──────────────────────────────────────


# Cache the LLM verdict per normalized message so repeated identical
# turns don't hit Groq twice in a row.  Capped via _CACHE_MAX_ENTRIES
# (FIFO eviction by simple dict iteration order — adequate for chat
# session-scale traffic; not a high-traffic concurrent cache).
_LLM_VERDICT_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_MAX_ENTRIES: int = 256


# System prompt: deliberately enumerates the allowed kinds + tool
# targets + the primary_intent precedence rules so the LLM returns
# a verdict our merge step can consume directly.  Asks for the
# detected language so the operator's trace replay shows which
# language the model thought the user was speaking.
_LLM_COMPOUND_SYSTEM_PROMPT = (
    "You are a strict multilingual intent classifier for a non-diagnostic "
    "breast-cancer monitoring patient-support assistant.  The user may "
    "write in ANY language (English, Filipino/Taglish, Spanish, Bahasa, "
    "Vietnamese, Chinese, Arabic, Russian, French, Portuguese, German, "
    "Japanese, Korean, Hindi, Thai, Bengali, Urdu, etc., including "
    "mixed-language and code-switched messages).  Identify EVERY intent "
    "present in the message and return STRICT JSON only.\n"
    "\n"
    "Allowed segment.kind values:\n"
    "  - casual_opener      : greeting, identity question, capability question, social check-in, "
    "thanks\n"
    "  - tool_request       : user wants to log/save/track/record patient data (symptom, CBC/lab, "
    "imaging report, medication)\n"
    "  - education_request  : user asks the meaning of a medical term, what something means, "
    "explanation request\n"
    "  - safety_boundary    : treatment-decision request (start/stop/change/delay/dose), "
    "diagnosis confirmation request, prognosis request, urgent symptom (chest pain, heavy "
    "bleeding, fever-after-chemo, self-harm)\n"
    "  - emotional_support  : anxiety, fear, sadness, overwhelm expressed by patient\n"
    "  - memory_request     : user references prior conversation / what they said earlier\n"
    "  - portal_help        : user asks how to use the portal / upload / dashboard\n"
    "  - timeline_request   : user asks about cycle progress / treatment history / monitoring "
    "score\n"
    "  - general_support    : none of the above\n"
    "\n"
    "Allowed tool_targets values (only on tool_request segments): "
    "save_symptom, save_complete_cbc, save_imaging_report, save_medication.\n"
    "\n"
    "primary_intent precedence (highest wins):\n"
    "  1. safety_boundary -> safety_boundary\n"
    "  2. tool_request    -> data_entry_intention\n"
    "  3. education_request -> education\n"
    "  4. emotional_support / memory_request / portal_help / timeline_request -> match\n"
    "  5. casual_opener   -> conversation\n"
    "  6. otherwise       -> general_support\n"
    "\n"
    "DO NOT diagnose, interpret, or recommend anything.  Just CLASSIFY "
    "what the user is asking for.\n"
    "\n"
    "Return JSON: {"
    "\"language\": str (ISO-639-1 or 'mixed' or 'unknown'), "
    "\"segments\": [{\"kind\": str, \"span\": str, \"tool_targets\": [str]}], "
    "\"primary_intent\": str, "
    "\"casual_opener_acknowledgment\": str|null, "
    "\"confidence\": float 0..1"
    "}"
)


# Outer set of valid kinds, kept here so the merge step refuses to
# accept an unknown kind the LLM hallucinates.
_LLM_ALLOWED_KINDS: frozenset[str] = frozenset({
    "casual_opener",
    "tool_request",
    "education_request",
    "safety_boundary",
    "emotional_support",
    "memory_request",
    "portal_help",
    "timeline_request",
    "general_support",
})


_LLM_ALLOWED_TOOL_TARGETS: frozenset[str] = frozenset({
    "save_symptom",
    "save_complete_cbc",
    "save_imaging_report",
    "save_medication",
})


_LLM_ALLOWED_PRIMARY_INTENTS: frozenset[str] = frozenset({
    "data_entry_intention",
    "education",
    "safety_boundary",
    "conversation",
    "emotional_support",
    "patient_memory",
    "portal_help",
    "patient_timeline_monitoring",
    "general_support",
})


_KIND_TO_DEFAULT_INTENT: dict[str, str] = {
    "casual_opener":     "conversation",
    "tool_request":      "data_entry_intention",
    "education_request": "education",
    "safety_boundary":   "safety_boundary",
    "emotional_support": "emotional_support",
    "memory_request":    "patient_memory",
    "portal_help":       "portal_help",
    "timeline_request":  "patient_timeline_monitoring",
    "general_support":   "general_support",
}


def _cache_get(key: str) -> dict[str, Any] | None:
    return _LLM_VERDICT_CACHE.get(key)


def _cache_put(key: str, verdict: dict[str, Any]) -> None:
    if len(_LLM_VERDICT_CACHE) >= _CACHE_MAX_ENTRIES:
        # Evict the oldest entry.  Dicts preserve insertion order.
        try:
            oldest = next(iter(_LLM_VERDICT_CACHE))
            _LLM_VERDICT_CACHE.pop(oldest, None)
        except StopIteration:
            pass
    _LLM_VERDICT_CACHE[key] = verdict


def _invalidate_llm_cache() -> None:
    """Test helper — wipe the per-normalized-message cache."""
    _LLM_VERDICT_CACHE.clear()


def classify_compound_intent_with_llm(message: str) -> dict[str, Any] | None:
    """Multilingual LLM second opinion on the message's compound intent.

    Returns ``None`` when:
      - the message is empty, or
      - the configured LLM adjudicator is unavailable (FAST_MODE on,
        no provider configured, or provider call failed).

    Returns a dict shaped like ``CompoundIntent.to_dict()`` augmented
    with ``language`` and ``llm_confidence`` when the LLM responded.
    """
    if not message or not message.strip():
        return None

    cache_key = normalize_user_text(message) or message.strip().lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from backend.services.local_llm import _adjudicate_json
        import json as _json

        verdict = _adjudicate_json(
            system=_LLM_COMPOUND_SYSTEM_PROMPT,
            prompt=_json.dumps({"message": message}, ensure_ascii=False),
            tier="router",
        )
    except Exception as exc:  # noqa: BLE001 — never crash chat on the classifier
        return None

    if not verdict.get("available"):
        return None

    normalized = _normalize_llm_verdict(verdict)
    if normalized is not None:
        _cache_put(cache_key, normalized)
    return normalized


def _normalize_llm_verdict(verdict: dict[str, Any]) -> dict[str, Any] | None:
    """Clamp the LLM's free-form output to our schema.  Drops unknown
    kinds / tool targets / primary intents instead of trusting them
    blindly.  Returns None when nothing usable survives."""
    raw_segments = verdict.get("segments") or []
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip()
        if kind not in _LLM_ALLOWED_KINDS:
            continue
        tool_targets_raw = raw.get("tool_targets") or []
        tool_targets = [
            str(t) for t in tool_targets_raw
            if isinstance(t, str) and t in _LLM_ALLOWED_TOOL_TARGETS
        ]
        span = str(raw.get("span") or "").strip()[:240]
        segments.append({
            "kind":         kind,
            "intent":       _KIND_TO_DEFAULT_INTENT.get(kind, "general_support"),
            "span":         span,
            "tool_targets": tool_targets,
        })

    if not segments:
        return None

    raw_primary = str(verdict.get("primary_intent") or "").strip()
    primary_intent = raw_primary if raw_primary in _LLM_ALLOWED_PRIMARY_INTENTS else "general_support"

    raw_language = str(verdict.get("language") or "").strip().lower()[:16] or "unknown"

    raw_confidence = verdict.get("confidence")
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    ack = verdict.get("casual_opener_acknowledgment")
    if isinstance(ack, str):
        ack = ack.strip()[:240]
        if not ack:
            ack = None
    else:
        ack = None

    return {
        "language":                     raw_language,
        "llm_confidence":               round(confidence, 3),
        "segments":                     segments,
        "primary_intent":               primary_intent,
        "casual_opener_acknowledgment": ack,
        "model":                        verdict.get("model"),
        "provider":                     verdict.get("provider"),
    }


def merge_compound_intent_with_llm(deterministic: CompoundIntent, llm_verdict: dict[str, Any]) -> CompoundIntent:
    """Merge a deterministic envelope with the LLM verdict.

    Safety floor: the deterministic safety / blocked branches always
    win.  Helpfulness floor: any tool_request OR education_request the
    LLM detected that the deterministic pass missed is added so the
    chat layer can act on it.

    The LLM verdict NEVER demotes a deterministic tool_request to a
    casual_opener — that would create a regression on the existing
    multilingual table.
    """
    if not llm_verdict:
        return deterministic

    # If the deterministic pass already detected a safety boundary
    # (treatment_decision / urgent / diagnostic) we trust it and
    # don't let the LLM change the primary_intent.
    deterministic_kinds = {s.kind for s in deterministic.segments}
    deterministic_safety = "safety_boundary" in deterministic_kinds

    # Build the union of segments.  We dedupe by (kind, normalized span).
    seen_keys: set[tuple[str, str]] = set()
    merged_segments: list[IntentSegment] = []
    for seg in deterministic.segments:
        key = (seg.kind, seg.span.lower().strip())
        seen_keys.add(key)
        merged_segments.append(seg)

    for raw in llm_verdict.get("segments") or []:
        key = (raw["kind"], raw["span"].lower().strip())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_segments.append(IntentSegment(
            intent=raw["intent"],
            kind=raw["kind"],
            span=raw["span"],
            tool_targets=list(raw.get("tool_targets") or []),
        ))

    # Recompute the envelope so primary_intent etc. reflect the
    # merged segment set.
    rebuilt = _build_envelope(merged_segments, original_message=" ".join(s.span for s in merged_segments))

    # Safety-deterministic-wins guard.  If the deterministic pass said
    # safety_boundary, force the primary intent to safety_boundary.
    if deterministic_safety:
        rebuilt.primary_intent = "safety_boundary"

    # Prefer the deterministic acknowledgment when present (already
    # tuned for our voice).  Fall back to the LLM's suggestion.
    if deterministic.suggested_acknowledgment:
        rebuilt.suggested_acknowledgment = deterministic.suggested_acknowledgment
    elif llm_verdict.get("casual_opener_acknowledgment"):
        rebuilt.suggested_acknowledgment = llm_verdict["casual_opener_acknowledgment"]

    return rebuilt


def detect_compound_intents_with_llm(
    message: str,
    *,
    use_llm: bool = True,
) -> tuple[CompoundIntent, dict[str, Any] | None]:
    """Return (merged_envelope, raw_llm_verdict_or_none).

    The merged envelope is the deterministic envelope augmented with
    LLM segments when ``use_llm=True`` and the adjudicator is
    available.  Tests opt out with ``use_llm=False`` for hermetic runs.
    """
    deterministic = detect_compound_intents(message)
    llm_verdict = classify_compound_intent_with_llm(message) if use_llm else None
    if llm_verdict is None:
        return deterministic, None
    return merge_compound_intent_with_llm(deterministic, llm_verdict), llm_verdict


__all__ = [
    "IntentSegment",
    "CompoundIntent",
    "detect_compound_intents",
    "classify_compound_intent_with_llm",
    "merge_compound_intent_with_llm",
    "detect_compound_intents_with_llm",
]
