import re

REDACTION_TOKEN = "[redacted]"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,2}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MRN_RE = re.compile(r"\bMRN\s*[:#-]?\s*[A-Za-z0-9]{5,}\b", re.IGNORECASE)
DOB_RE = re.compile(r"\b(?:DOB|Date of Birth)\s*[:#-]?\s*\d{4}-\d{2}-\d{2}\b", re.IGNORECASE)


def redact_text(text):
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    redacted = EMAIL_RE.sub(REDACTION_TOKEN, text)
    redacted = PHONE_RE.sub(REDACTION_TOKEN, redacted)
    redacted = SSN_RE.sub(REDACTION_TOKEN, redacted)
    redacted = MRN_RE.sub(REDACTION_TOKEN, redacted)
    redacted = DOB_RE.sub(REDACTION_TOKEN, redacted)
    return redacted


def redact_payload(payload):
    if payload is None:
        return None
    if isinstance(payload, str):
        return redact_text(payload)
    if isinstance(payload, dict):
        return {key: redact_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [redact_payload(value) for value in payload]
    return payload
