from __future__ import annotations

import re

from backend.models import ChatMessage
from backend.services.security_guardrails import detect_multilingual_medical_danger, normalize_security_text
from backend.services.support_chat_policy import (
    DOMAIN_SCOPE_TERMS,
    GENERAL_SUPPORT_PATTERNS,
    IMMEDIATE_DANGER_PATTERNS,
    OUT_OF_DOMAIN_PATTERNS,
    SAFETY_LOCATION_FOLLOWUP_PATTERNS,
    URGENT_TERMS,
)

def _is_immediate_danger_statement(message):
    normalized = str(message or "").lower().replace("’", "'").strip()
    return any(re.search(pattern, normalized) for pattern in IMMEDIATE_DANGER_PATTERNS)


def _resolve_safety_location_followup(db, patient_id, message):
    normalized = re.sub(r"\s+", " ", str(message or "").lower().strip())
    if not any(re.fullmatch(pattern, normalized) for pattern in SAFETY_LOCATION_FOLLOWUP_PATTERNS):
        return None

    previous = (
        db.query(ChatMessage)
        .filter(ChatMessage.patient_id == patient_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .first()
    )
    if previous is None:
        return None

    previous_text = str(previous.message or "").lower()
    safety_markers = (
        "emergency services",
        "emergency department",
        "immediate help",
        "feel unsafe",
        "might hurt yourself",
        "or go to",
    )
    if not any(marker in previous_text for marker in safety_markers):
        return None
    return {
        "previous_assistant_message_id": previous.id,
        "reason": "clarifies the location in the preceding safety escalation",
    }


def _immediate_danger_reply():
    return (
        "I'm sorry this feels so frightening. I cannot assess what is happening through this portal. "
        "If you feel in immediate danger or unsafe, contact local emergency services or go to the "
        "nearest emergency department now. If you are not in immediate danger, contact your oncology "
        "care team now and tell them exactly what you are experiencing. If possible, ask someone you "
        "trust to stay with you while you make that contact."
    )


def _safety_location_followup_reply():
    return (
        "I meant the nearest emergency department or local emergency services if you feel in immediate "
        "danger or unsafe. If you are not in immediate danger, contact your oncology care team now and "
        "tell them exactly what you are experiencing. If possible, ask someone you trust to stay with you "
        "while you make that contact."
    )


def _enforce_record_provenance(reply, actions):
    if any(str(action.get("type", "")).startswith("saved_") for action in actions or []):
        return str(reply or "")

    text = str(reply or "")
    patient_authorship_patterns = (
        r"\byou(?:['’]ve| have)?\s+logged\b",
        r"\byou(?:['’]ve| have)?\s+recorded\b",
        r"\byou(?:['’]ve| have)?\s+saved\b",
        r"\byou(?:['’]ve| have)?\s+added\b",
    )

    def neutral_record_phrase(match):
        prefix = "The" if match.group(0)[:1].isupper() else "the"
        return f"{prefix} portal record currently shows"

    for pattern in patient_authorship_patterns:
        text = re.sub(pattern, neutral_record_phrase, text, flags=re.IGNORECASE)
    return text


def _is_out_of_domain_request(message, *, actions, safety, emotional_distress):
    """Keep the patient assistant inside its declared product scope.

    The check deliberately runs after structured extraction and safety routing,
    so a terse lab entry, emotional message, or urgent statement can never be
    rejected merely because it lacks an explicit oncology keyword.
    """

    if actions:
        return False
    safety = safety or {}
    if safety.get("level") in {"high_risk", "blocked"}:
        return False
    if emotional_distress is not None and getattr(emotional_distress, "detected", False):
        return False

    raw_text = str(message or "").strip()
    if re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?(?:\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?)+\s*(?:=|\?)?",
        raw_text,
    ):
        return True

    normalized = normalize_security_text(message)
    if not normalized:
        return False
    scope_candidate = re.sub(
        r"^(?:(?:please answer|random question|quick one|curious lang)\s*:?[ ]*)+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    if any(re.search(pattern, scope_candidate, flags=re.IGNORECASE) for pattern in GENERAL_SUPPORT_PATTERNS):
        return False
    if any(
        (term in scope_candidate if " " in term else re.search(rf"\b{re.escape(term)}\b", scope_candidate))
        for term in DOMAIN_SCOPE_TERMS
    ):
        return False
    if any(re.search(pattern, scope_candidate, flags=re.IGNORECASE) for pattern in OUT_OF_DOMAIN_PATTERNS):
        return True

    # Questions with enough semantic content but no portal/oncology anchor are
    # treated as general-purpose requests. Very short follow-ups remain allowed
    # so "why?" and "what does that mean?" can use conversation context.
    words = re.findall(r"[a-z0-9']+", scope_candidate)
    question_like = bool(re.match(r"^(?:who|what|when|where|which|how|why|tell|explain|write|make|calculate|solve|summarize|recommend|translate|debug|give)\b", scope_candidate))
    return question_like and len(words) >= 3


def _out_of_domain_reply():
    return (
        "I’m NLCare’s breast-cancer monitoring support assistant, so I can’t help with unrelated "
        "history, politics, trivia, calculations, coding, or other general-purpose requests. I can "
        "help you understand records already in this portal, organize symptoms, labs, medications, "
        "and imaging notes, explain general breast-cancer terms with sources, or prepare questions "
        "for your care team. I do not diagnose or recommend treatment."
    )


def _looks_truncated_reply(reply):
    text = str(reply or "").strip()
    if not text:
        return True
    if re.search(r"(?:\bif|\band|\bor|\bbut|\bbecause|\bto|\bthe|\ba|\ban|\bof|\bfor|\bwith|\byour|:)\s*$", text, flags=re.IGNORECASE):
        return True
    if len(text.split()) >= 45 and text[-1] not in ".?!)]\"'":
        return True
    return False


def _ensure_complete_response(reply, fallback_reply):
    text = str(reply or "").strip()
    if not _looks_truncated_reply(text):
        return text
    fallback = str(fallback_reply or "").strip()
    if fallback and not _looks_truncated_reply(fallback):
        return fallback
    # Last-resort cleanup for a malformed provider response. Keep only complete
    # sentences and add a stable scope-safe close rather than showing a fragment.
    matches = list(re.finditer(r"[.?!](?:[\"')\]]+)?(?=\s|$)", text))
    complete = text[: matches[-1].end()].strip() if matches else ""
    close = "I can help organize the relevant record details for your care team."
    return f"{complete} {close}".strip()


def _ensure_complete_safety_reply(reply, safety):
    text = str(reply or "").strip()
    if not text:
        return _immediate_danger_reply()
    safety = safety or {}
    is_high_risk = safety.get("level") == "high_risk" or safety.get("scope") == "urgent_or_safety_related"
    if not is_high_risk:
        return text

    if re.search(r"\b(?:or\s+)?go\s+to\s*$", text, flags=re.IGNORECASE):
        return f"{text} the nearest emergency department now."
    if re.search(r"\b(?:or\s+)?contact\s*$", text, flags=re.IGNORECASE):
        return f"{text} local emergency services now."
    if re.search(r"\bor\s*$", text, flags=re.IGNORECASE):
        return f"{text} contact local emergency services now."
    return text


def _detect_urgent_flags(message):
    normalized = normalize_security_text(message)
    flags = [term for term in URGENT_TERMS if term in normalized]
    danger = detect_multilingual_medical_danger(message)
    flags.extend(danger.get("matches") or [])
    return sorted(set(flags))


def _prefer_deterministic_reply(message):
    lower = message.lower()
    deterministic_terms = [
        "last 14",
        "last fourteen",
        "what changed",
        "timeline",
        "tumor board",
        "toxicity",
        "cycle 2",
        "score",
        "model",
        "why",
    ]
    return any(term in lower for term in deterministic_terms)
