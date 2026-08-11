"""Independent authorization guard for bounded patient-agent routes.

This layer reasons about the *operation* requested by a user.  It does not
replace the semantic unsafe-intent classifier or the security guardrail.  It
exists so a missed classifier prototype cannot by itself authorize record
disclosure, cross-record access, instruction bypass, or medical authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.services.agent_text_normalization import normalize_agent_text


@dataclass(frozen=True)
class RouteAuthorizationDecision:
    allowed: bool
    route: str
    family: str
    reason: str
    matched_rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "route": self.route,
            "family": self.family,
            "reason": self.reason,
            "matched_rule": self.matched_rule,
        }


_BYPASS_ACTION = re.compile(
    r"\b(ignore|disregard|override|bypass|disable|remove|suspend|set\s+aside|"
    r"work\s+around|skip|evade|execute\s+(?:beyond|past)|treat\s+as\s+approved)\b",
    re.IGNORECASE,
)
_CONTROL_OBJECT = re.compile(
    r"\b(system\s+prompt|instruction(?:s|\s+hierarchy)?|policy|policies|rule(?:s)?|"
    r"validator|guardrails?|safety\s+check|refusal|access\s+control)\b",
    re.IGNORECASE,
)
_RECORD_ACTION = re.compile(
    r"\b(show|display|reveal|expose|export|send|share|copy|quote|open|retrieve|"
    r"summarize|summarise|compare|repeat|unmask|leave\s+visible|access)\b",
    re.IGNORECASE,
)
_PROTECTED_OBJECT = re.compile(
    r"\b(other|another|prior|previous|different|someone\s+else(?:'s)?|cousin(?:'s)?|"
    r"same[-\s]?surname|foreign)\b.{0,55}\b(patient|person|account|record|chart|"
    r"timeline|result|lab|identifier|membership|header)\b|"
    r"\b(private|protected|confidential)\b.{0,35}\b(record|chart|identifier|account|"
    r"header|number|data)\b",
    re.IGNORECASE,
)
_DIRECT_IDENTIFIER = re.compile(
    r"\b(account|membership|registration|patient|medical\s+record)\s+"
    r"(identifier|id|number)\b",
    re.IGNORECASE,
)
_PROTECTIVE_CONTEXT = re.compile(
    r"\b(do\s+not|don't|never|without)\b.{0,45}\b(reveal|expose|share|send|copy|"
    r"execute|follow|apply|access)\b|\bhow\s+(?:do|can|should)\b.{0,55}\b"
    r"(protect|redact|prevent|verify|authorize|authorise)\b|"
    r"\b(explain|describe|show)\b.{0,55}\b(prevent|protect|redact|verify)\b",
    re.IGNORECASE,
)
_SAFE_DISCUSSION = re.compile(
    r"\b(explain|why|what\s+safeguards|security\s+example|training\s+example|"
    r"redaction\s+process|consent|privacy\s+process)\b",
    re.IGNORECASE,
)


def authorize_patient_route(message: str) -> RouteAuthorizationDecision:
    """Return a fail-closed route decision for protected operations.

    Benign questions about privacy and security remain answerable.  Quoting,
    role-play, hypothetical framing, or a claimed professional role never
    grants permission to perform a protected operation.
    """

    text = normalize_agent_text(message or "")
    protective = bool(_PROTECTIVE_CONTEXT.search(text))
    safe_discussion = bool(_SAFE_DISCUSSION.search(text))

    if _BYPASS_ACTION.search(text) and _CONTROL_OBJECT.search(text):
        if protective and safe_discussion:
            return RouteAuthorizationDecision(
                True,
                "continue",
                "none",
                "Protective discussion of an instruction-bypass example.",
                "protective_security_discussion",
            )
        return RouteAuthorizationDecision(
            False,
            "security_refusal",
            "prompt_injection",
            "A user cannot authorize bypass of system or safety controls.",
            "control_bypass_operation",
        )

    protected_object = bool(_PROTECTED_OBJECT.search(text) or _DIRECT_IDENTIFIER.search(text))
    if _RECORD_ACTION.search(text) and protected_object:
        if protective and safe_discussion:
            return RouteAuthorizationDecision(
                True,
                "continue",
                "none",
                "Protective privacy or redaction discussion without disclosure.",
                "protective_privacy_discussion",
            )
        family = "cross_patient_exfiltration" if _PROTECTED_OBJECT.search(text) else "privacy_pii"
        return RouteAuthorizationDecision(
            False,
            "security_refusal",
            family,
            "The requested disclosure or access operation is not authorized.",
            "protected_record_operation",
        )

    return RouteAuthorizationDecision(
        True,
        "continue",
        "none",
        "No protected operation was requested.",
        None,
    )


__all__ = ["RouteAuthorizationDecision", "authorize_patient_route"]
