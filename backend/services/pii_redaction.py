"""Shared, conservative redaction for operational and audit telemetry."""

from __future__ import annotations

import re
from typing import Any

REDACTION_TOKEN = "[redacted]"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,2}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MRN_RE = re.compile(r"\bMRN\s*[:#-]?\s*[A-Za-z0-9]{5,}\b", re.IGNORECASE)
DOB_RE = re.compile(r"\b(?:DOB|Date of Birth)\s*[:#-]?\s*\d{4}-\d{2}-\d{2}\b", re.IGNORECASE)
BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
JWT_RE = re.compile(r"\b[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
API_CREDENTIAL_RE = re.compile(
    r"\b(?:sk|pk|api|webhook)[-_][A-Za-z0-9._~+/=-]{8,}\b",
    re.IGNORECASE,
)
SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:token|api[_-]?key|secret|signature|session|authorization)=)[^&#\s]+",
    re.IGNORECASE,
)

_SENSITIVE_KEY_PARTS = {
    "authorization",
    "cookie",
    "credential",
    "dose",
    "finding",
    "free_text",
    "impression",
    "medication",
    "message",
    "note",
    "password",
    "patient",
    "prompt",
    "report_text",
    "secret",
    "session",
    "side_effect",
    "symptom",
    "token",
    "webhook",
    "variant",
}


def is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if "api_key" in normalized or "apikey" in normalized:
        return True
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact_text(text: Any) -> Any:
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    redacted = EMAIL_RE.sub(REDACTION_TOKEN, text)
    redacted = PHONE_RE.sub(REDACTION_TOKEN, redacted)
    redacted = SSN_RE.sub(REDACTION_TOKEN, redacted)
    redacted = MRN_RE.sub(REDACTION_TOKEN, redacted)
    redacted = DOB_RE.sub(REDACTION_TOKEN, redacted)
    redacted = BEARER_RE.sub(REDACTION_TOKEN, redacted)
    redacted = JWT_RE.sub(REDACTION_TOKEN, redacted)
    redacted = API_CREDENTIAL_RE.sub(REDACTION_TOKEN, redacted)
    redacted = SENSITIVE_QUERY_RE.sub(r"\1[redacted]", redacted)
    return redacted


def redact_payload(payload: Any, *, key: str = "", depth: int = 0) -> Any:
    """Redact secrets and patient prose while retaining bounded metadata.

    This helper is used by both stdout telemetry and the database audit trail.
    It intentionally does not attempt to preserve the contents of fields such
    as ``message``, ``symptom``, ``findings``, or ``notes``. Those values belong
    in their domain tables, not duplicated into operational logs.
    """
    if key and is_sensitive_key(key):
        return REDACTION_TOKEN
    if payload is None:
        return None
    if depth >= 6:
        return "[truncated]"
    if isinstance(payload, str):
        return redact_text(payload)[:500]
    if isinstance(payload, dict):
        return {
            str(child_key): redact_payload(
                value,
                key=str(child_key),
                depth=depth + 1,
            )
            for child_key, value in list(payload.items())[:50]
        }
    if isinstance(payload, (list, tuple)):
        return [redact_payload(value, depth=depth + 1) for value in payload[:50]]
    return payload


__all__ = ["REDACTION_TOKEN", "is_sensitive_key", "redact_payload", "redact_text"]
