"""Input-stage guardrails for the patient agent.

Two responsibilities:

  - :func:`input_guardrail_check` — fuses the security-detector's
    verdict (prompt injection / exfiltration / privacy boundary) with
    the safety-scope envelope and returns a single guardrail decision.
    Sets ``status="failed"`` only on hard-blocking signals so a
    treatment-decision query (high_risk_medical_scope) is recorded
    for safety routing but does not abort the pipeline at the input
    stage.

  - :func:`security_block_reply` — the deterministic refusal returned
    on a failed input guardrail.  Patient-facing; lists the detected
    category(ies) so the user gets a real signal and not just "blocked."

Extracted from ``agent_rag.py`` as part of the agent_rag.py module
split.  Both functions are re-exported from ``backend.services.agent_rag``
so existing call sites work unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping

from backend.services.security_guardrails import detect_prompt_injection_or_exfiltration


# Issues that hard-fail the input guardrail.  Any other issue (notably
# ``high_risk_medical_scope`` from the safety classifier) is recorded
# but does not block at this stage — safety routing happens later.
BLOCKING_ISSUES: frozenset[str] = frozenset({
    "prompt_injection_or_jailbreak",
    "database_or_file_access_attempt",
    "sensitive_data_exfiltration_attempt",
    "privacy_boundary_request",
})


def input_guardrail_check(query: str, safety: Mapping[str, Any]) -> dict[str, Any]:
    """Fuse the security-detector verdict with the safety scope and
    return the input-stage guardrail envelope.

    Returns a dict with ``status`` (passed/failed), ``scope`` (the
    matched safety scope or ``input_guardrail_block`` on failure),
    ``issues`` (sorted unique list), ``message``, and a ``security``
    sub-block carrying the detector's confidence + raw signals.
    """
    security = detect_prompt_injection_or_exfiltration(query)
    issues: list[str] = []
    if security["blocked"]:
        issues.extend(security["issues"])
    if safety.get("level") == "high_risk":
        issues.append(safety.get("scope") or "high_risk_medical_scope")

    status = "failed" if any(issue in BLOCKING_ISSUES for issue in issues) else "passed"
    if status == "failed":
        scope = "input_guardrail_block"
        message = security["message"]
    else:
        scope = safety.get("scope")
        message = "Input guardrail passed."
    return {
        "status":  status,
        "scope":   scope,
        "issues":  sorted(set(issues)),
        "message": message,
        "security": {
            "confidence": security["confidence"],
            "signals":    security["signals"],
        },
    }


def security_block_reply(input_guardrails: Mapping[str, Any]) -> str:
    """Patient-facing deterministic refusal returned when the input
    guardrail fails.  Names the detected category so the user gets a
    real signal."""
    issues = ", ".join(input_guardrails.get("issues") or ["unsafe request"])
    return (
        "I blocked that request for security and privacy reasons. "
        "I cannot reveal system instructions, database contents, secrets, raw internal knowledge-base data, "
        "or any other patient's information. "
        f"Detected category: {issues}. "
        "You can ask general breast cancer treatment-monitoring questions or enter your own symptoms, labs, medications, and uploads. "
        "For medical concerns, contact your oncology care team."
    )


# Back-compat alias — the agent_rag call site uses the underscore form.
_security_block_reply = security_block_reply


__all__ = [
    "BLOCKING_ISSUES",
    "input_guardrail_check",
    "security_block_reply",
    "_security_block_reply",
]
