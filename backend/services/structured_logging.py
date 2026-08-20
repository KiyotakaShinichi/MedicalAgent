"""Canonical structured logging with conservative sensitive-data redaction."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any


LOGGER = logging.getLogger("nlcare.events")
_SENSITIVE_KEY_PARTS = {
    "authorization", "cookie", "message", "password", "patient", "prompt",
    "secret", "token",
}
_ALLOWED_SEVERITIES = {"debug", "info", "warning", "error", "critical"}


class JsonEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "nlcare_event", None)
        if not isinstance(event, dict):
            event = {
                "schema_version": "structured_event_v2",
                "event_type": "application_log",
                "severity": record.levelname.lower(),
                "component": record.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": {"summary": str(record.getMessage())[:500]},
            }
        return json.dumps(event, sort_keys=True, default=str)


def configure_logging(*, force: bool = False) -> None:
    """Configure one JSON event handler without mutating unrelated loggers."""
    if LOGGER.handlers and not force:
        return
    LOGGER.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonEventFormatter())
    LOGGER.addHandler(handler)
    LOGGER.setLevel(os.environ.get("NLCARE_LOG_LEVEL", "INFO").upper())
    LOGGER.propagate = False


def _sanitize(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if depth >= 4:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, key=str(k), depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth=depth + 1) for item in value[:50]]
    if isinstance(value, str):
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


def new_correlation_id(prefix: str = "req") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def build_event(
    event_type: str,
    *,
    severity: str = "info",
    request_id: str | None = None,
    correlation_id: str | None = None,
    user_role: str | None = None,
    patient_id: str | None = None,
    artifact_id: str | None = None,
    model_version: str | None = None,
    component: str = "application",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_severity = severity.lower() if severity.lower() in _ALLOWED_SEVERITIES else "info"
    return {
        "schema_version": "structured_event_v2",
        "event_type": event_type,
        "severity": normalized_severity,
        "component": component,
        "request_id": request_id or correlation_id or new_correlation_id(),
        "correlation_id": correlation_id or request_id or new_correlation_id("corr"),
        "user_role": user_role,
        "patient_id": "[REDACTED]" if patient_id else None,
        "artifact_id": artifact_id,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": _sanitize(details or {}),
    }


def log_event(event_type: str, **kwargs: Any) -> dict[str, Any]:
    configure_logging()
    event = build_event(event_type, **kwargs)
    level = getattr(logging, str(event["severity"]).upper(), logging.INFO)
    LOGGER.log(level, event_type, extra={"nlcare_event": event})
    return event


__all__ = [
    "JsonEventFormatter", "build_event", "configure_logging", "log_event",
    "new_correlation_id",
]
