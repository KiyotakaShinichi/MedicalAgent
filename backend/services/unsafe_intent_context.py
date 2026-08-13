"""Preserve an unsafe boundary across short, referential follow-up turns."""

from __future__ import annotations

import re
from typing import Any

from backend.services.agent_text_normalization import normalize_agent_text
from backend.services.unsafe_intent_semantic_classifier import (
    classify_unsafe_intent,
)
from backend.services.safety_uncertainty_adjudicator import (
    adjudicate_safety_uncertainty,
)


CONTEXT_CONTINUATION_CUES: tuple[str, ...] = (
    "do it",
    "go ahead",
    "continue",
    "continue with that",
    "same request",
    "same thing",
    "just answer",
    "answer it",
    "show it",
    "use that",
    "only the latest",
    "yes or no",
    "without the disclaimer",
    "without warnings",
    "please proceed",
    "sige",
    "ituloy",
    "gawin mo",
    "yun nga",
)

CONTEXT_RESET_CUES: tuple[str, ...] = (
    "never mind",
    "cancel that",
    "do not do that",
    "don't do that",
    "ignore my previous request",
    "withdraw that request",
    "instead explain generally",
    "explain why that is unsafe",
    "what is the safe way",
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "or",
    "that",
    "the",
    "this",
    "to",
    "you",
}


def classify_unsafe_intent_with_context(
    text: str,
    previous_user_messages: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Retain a recent unsafe decision only for referential follow-ups."""

    current = classify_unsafe_intent(text)
    normalized = normalize_agent_text(text)
    prior_messages = tuple(
        message.strip()
        for message in (previous_user_messages or ())
        if isinstance(message, str) and message.strip()
    )[-3:]
    if (
        not prior_messages
        or current.get("is_unsafe")
        or any(cue in normalized for cue in CONTEXT_RESET_CUES)
    ):
        return _with_context(current, False, len(prior_messages))

    continuation = (
        any(cue in normalized for cue in CONTEXT_CONTINUATION_CUES)
        or _meaningful_token_count(normalized) <= 4
    )
    if not continuation:
        return _with_context(current, False, len(prior_messages))

    combined = classify_unsafe_intent(" ".join((*prior_messages[-2:], text)))
    if combined.get("is_unsafe"):
        confidence_floor = 0.58 if combined.get("borderline") else 0.84
        confidence = max(float(combined.get("confidence") or 0.0), confidence_floor)
        return {
            **combined,
            "confidence": confidence,
            "unsafe_intent_confidence": confidence,
            "safety_source": "contextual_composition",
            "context_reused": True,
            "context_turn_count": len(prior_messages),
        }

    composed_text = " ".join((*prior_messages[-2:], text))
    adjudication = adjudicate_safety_uncertainty(composed_text, combined)
    if adjudication.requires_safe_route:
        return {
            "is_unsafe": True,
            "family": adjudication.family,
            "confidence": adjudication.confidence,
            "route": "safe_clarification",
            "scope": adjudication.scope,
            "safety_source": "contextual_uncertainty_adjudication",
            "unsafe_intent_family": adjudication.family,
            "unsafe_intent_confidence": adjudication.confidence,
            "over_refusal_risk_flag": False,
            "safe_template": None,
            "over_refusal_risk_notes": None,
            "matched_pattern": None,
            "matched_semantic_rule": adjudication.reason,
            "borderline": True,
            "context_reused": True,
            "context_turn_count": len(prior_messages),
        }

    for prior in reversed(prior_messages):
        prior_result = classify_unsafe_intent(prior)
        if prior_result.get("is_unsafe"):
            confidence_floor = 0.55 if prior_result.get("borderline") else 0.72
            confidence = max(
                min(float(prior_result.get("confidence") or 0.0) * 0.92, 0.92),
                confidence_floor,
            )
            return {
                **prior_result,
                "confidence": round(confidence, 4),
                "unsafe_intent_confidence": round(confidence, 4),
                "safety_source": "contextual_boundary_carryover",
                "context_reused": True,
                "context_turn_count": len(prior_messages),
            }
    return _with_context(current, False, len(prior_messages))


def _meaningful_token_count(text: str) -> int:
    return sum(
        token not in _STOPWORDS
        for token in re.findall(r"[a-z0-9]+", text.lower())
    )


def _with_context(
    result: dict[str, Any],
    reused: bool,
    turn_count: int,
) -> dict[str, Any]:
    return {
        **result,
        "context_reused": reused,
        "context_turn_count": turn_count,
    }


__all__ = [
    "CONTEXT_CONTINUATION_CUES",
    "CONTEXT_RESET_CUES",
    "classify_unsafe_intent_with_context",
]
