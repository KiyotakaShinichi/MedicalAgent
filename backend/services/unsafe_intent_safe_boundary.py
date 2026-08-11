"""Benign-process overrides for unsafe-intent routing.

These helpers keep requests about redaction, consent, recorded prescriptions,
and care-team questions answerable even when they quote dangerous vocabulary.
They never authorize access, treatment decisions, or medical conclusions.
"""

from __future__ import annotations

import re


_RECORDING_STATEMENT_PREFIXES = (
    "treatment note",
    "record that",
    "log that",
    "save that",
    "for my record",
)
_RECORDED_TREATMENT_EVENT_CUES = (
    "cycle was delayed",
    "cycle delayed",
    "treatment was delayed",
    "treatment was held",
    "infusion was held",
    "care team delayed",
    "care team held",
    "oncologist delayed",
    "oncologist held",
)
_TREATMENT_DECISION_REQUEST_CUES = (
    "should i",
    "can i",
    "may i",
    "do you recommend",
    "tell me to",
    "authorize",
    "decide for me",
    "next cycle",
    "tomorrow",
    "tonight",
)


def looks_like_recording_statement(normalized: str) -> bool:
    """Keep explicit historical record capture distinct from treatment advice."""
    if any(cue in normalized for cue in _TREATMENT_DECISION_REQUEST_CUES):
        return False
    has_record_prefix = any(prefix in normalized for prefix in _RECORDING_STATEMENT_PREFIXES)
    has_completed_event = any(cue in normalized for cue in _RECORDED_TREATMENT_EVENT_CUES)
    return has_record_prefix and has_completed_event


def looks_like_safe_boundary_request(normalized: str) -> bool:
    """Recognize requests whose objective is understanding or safer process."""
    safe_cues = (
        "anonymized public example",
        "what personal data this app stores",
        "what data this app stores",
    )
    if any(cue in normalized for cue in safe_cues):
        return True

    quoted_attack = (
        any(
            cue in normalized
            for cue in (
                "attack example",
                "security example",
                "security training",
                "awareness example",
            )
        )
        and any(
            cue in normalized
            for cue in (
                "without following",
                "without obeying",
                "without applying",
                "without executing",
                "do not execute",
                "do not follow",
                "do not apply",
            )
        )
    )
    if quoted_attack:
        return True

    protective_patterns = (
        r"\b(do not|don't|never)\b.{0,40}\b(reveal|expose|calculate|compute|estimate|predict|reclassify|conclude|decide|diagnose|confirm|replace|change|stop|skip|execute|apply)\b",
        r"\bwithout\b.{0,35}\b(revealing|exposing|calculating|computing|estimating|predicting|reclassifying|concluding|deciding|diagnosing|confirming|replacing|changing|executing|applying)\b",
        r"\b(i\s+)?(will not|won't|do not want to|don't want to)\b.{0,35}\b(change|stop|skip|replace|alter|reclassify|calculate|estimate)\b",
        r"\bnot asking (you )?to\b.{0,35}\b(confirm|diagnose|decide|calculate|estimate|predict|reveal|access)\b",
    )
    benign_objectives = (
        "explain",
        "what verification steps",
        "what records would",
        "what questions",
        "what should i ask",
        "prepare questions",
        "draft questions",
        "list questions",
        "help me list questions",
        "where can i find",
        "where can i view",
        "show me where",
        "why can",
        "why should",
        "why does",
        "why do",
        "how should staff",
        "how should the team",
        "what safeguards",
        "what steps",
        "what context",
    )
    governance_objectives = (
        "verify consent",
        "confirm consent",
        "authorization before access",
        "authorisation before access",
        "redaction process",
        "redact",
        "privacy procedure",
        "protect private",
        "prevent disclosure",
        "safe way to request",
        "prescriber's instructions",
        "prescriber",
        "prescription instructions",
        "recorded prescription",
        "recorded dose",
        "already recorded",
        "care team",
        "genetic counselor",
        "genetics-trained",
        "pharmacist",
        "clinical context",
        "other evidence",
    )
    unsafe_result_patterns = (
        r"\b(definitely|certainly|just|only)\b.{0,20}\b(have|is|whether|confirm|diagnose|choose|calculate|estimate)\b",
        r"\b(choose|calculate|estimate|confirm|decide|reveal|access)\b.{0,20}\bfor me\b",
        r"\b(use|apply|execute|follow)\b.{0,25}\b(the )?(attack|instruction|command|override)\b",
    )
    has_protective = any(re.search(pattern, normalized, re.IGNORECASE) for pattern in protective_patterns)
    has_benign_objective = any(cue in normalized for cue in benign_objectives) or any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in (
            r"\bwhat\b.{0,40}\bquestions?\b",
            r"\bquestions?\b.{0,25}\b(ask|bring|discuss|prepare|draft|list)\b",
            r"\bhow\b.{0,25}\b(verify|protect|redact|request|review)\b",
        )
    )
    has_governance_objective = any(cue in normalized for cue in governance_objectives)
    has_unsafe_result = any(re.search(pattern, normalized, re.IGNORECASE) for pattern in unsafe_result_patterns)
    disclosure_match = re.search(
        r"\b(display|show|export|send|share|disclose|reveal|expose|unmask|print|list|keep visible)\b.{0,45}\b(confidential|private|protected|registration|identifier|patient id|api key|credential|record|chart|other patient)\b",
        normalized,
        re.IGNORECASE,
    )
    if disclosure_match:
        prefix = normalized[max(0, disclosure_match.start() - 28):disclosure_match.start()]
        explicitly_negated = re.search(
            r"\b(do not|don't|never|without|not asking (you )?to)\s*$",
            prefix,
            re.IGNORECASE,
        )
        has_unsafe_result = has_unsafe_result or explicitly_negated is None
    return has_benign_objective and (has_protective or has_governance_objective) and not has_unsafe_result


__all__ = ["looks_like_recording_statement", "looks_like_safe_boundary_request"]
