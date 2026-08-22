"""Canonical structured logging with conservative sensitive-data redaction.

Built on the standard library only. `logging.config.dictConfig` drives setup,
so the configuration is inspectable data rather than imperative handler
wiring, and no third-party logging dependency is introduced.

Two logging systems exist in this codebase and are not interchangeable:

* this module emits **JSON events to stdout** for operational monitoring;
* `backend.services.app_logging` writes an **auditable database trail**
  (`AppEventLog`) for application/business events.

Both redact before writing. Use this one for anything an operator would grep;
use `app_logging` for anything that must survive as a record.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any


LOGGER = logging.getLogger("nlcare.events")
# Substrings matched against the *key* name, case-insensitively. Chosen to be
# specific enough not to redact ordinary operational fields: a bare "key" would
# blank `cache_key`, `chunk_key`, and `idempotency_key`, which are exactly the
# values an operator needs when debugging. `api_key` was missing until a
# redaction probe caught `{"api_key": "k-secret"}` reaching the log intact.
_SENSITIVE_KEY_PARTS = {
    "access_key", "api_key", "apikey", "authorization", "bearer", "cookie",
    "credential", "message", "password", "patient", "private_key", "prompt",
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


def logging_config() -> dict[str, Any]:
    """The `dictConfig` schema this application logs under.

    Exposed as data so it can be asserted on in tests and read by anyone
    auditing how logs are produced, rather than being implied by a sequence of
    `addHandler` calls.

    `disable_existing_loggers` is False on purpose: turning it on would silence
    library loggers already created at import time, and would break pytest's
    `caplog` capture.
    """
    level = os.environ.get("NLCARE_LOG_LEVEL", "INFO").upper()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "nlcare_json": {"()": f"{__name__}.JsonEventFormatter"},
        },
        "handlers": {
            "nlcare_stdout": {
                "class": "logging.StreamHandler",
                "formatter": "nlcare_json",
            },
        },
        "loggers": {
            LOGGER.name: {
                "handlers": ["nlcare_stdout"],
                "level": level,
                # Events carry their own JSON envelope; propagating would emit
                # each one twice once the root logger also has a handler.
                "propagate": False,
            },
        },
    }


def configure_logging(*, force: bool = False) -> None:
    """Install the JSON event handler. Idempotent.

    Also gives the root logger the same JSON formatter, but **only when it has
    no handlers of its own**. That makes third-party and framework output
    machine-readable in a deployed process, while leaving an embedding
    application's — or pytest's — logging setup untouched.
    """
    # Imported inside the function: this module is listed in [tool.mypy]
    # `files`, and pulling logging.config into the module graph at import time
    # makes mypy resolve backend/services under two module names and abort.
    from logging.config import dictConfig

    if LOGGER.handlers and not force:
        return
    LOGGER.handlers.clear()
    dictConfig(logging_config())

    root = logging.getLogger()
    if not root.handlers:
        root_handler = logging.StreamHandler()
        root_handler.setFormatter(JsonEventFormatter())
        root.addHandler(root_handler)
        root.setLevel(os.environ.get("NLCARE_ROOT_LOG_LEVEL", "WARNING").upper())


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
        # Defence in depth. Key-name matching cannot catch an identifier that
        # arrives inside an innocently-named field — `{"summary": "contact
        # maria@example.com"}` has no sensitive key. `redact_text` is the same
        # value-pattern pass (email/phone/SSN/MRN/DOB) the database audit trail
        # already applies, so both logging paths now enforce one policy.
        # Applied before truncation so a pattern spanning the 500-char cut is
        # still matched.
        from backend.services.pii_redaction import redact_text

        return redact_text(value)[:500]
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
    # Fall back to the in-flight request's id before minting a new one. Without
    # this, any `log_event` call from the service layer invented a fresh id, so
    # the events emitted while handling one request did not correlate with the
    # `http_request_completed` event for that same request — the correlation id
    # was present but useless for anything except the middleware's own events.
    # Imported here, not at module scope, on purpose. This module is one of the
    # files listed in [tool.mypy] `files`, and mypy resolves those by path: a
    # module-level `from backend.services...` import makes it load
    # backend/services/__init__.py under a second module name and abort with
    # "Source file found twice under different module names". Every other
    # mypy-tracked file is likewise free of module-level backend imports.
    from backend.services.request_context import get_request_id

    ambient_request_id = get_request_id()
    resolved_request_id = (
        request_id or correlation_id or ambient_request_id or new_correlation_id()
    )
    return {
        "schema_version": "structured_event_v2",
        "event_type": event_type,
        "severity": normalized_severity,
        "component": component,
        "request_id": resolved_request_id,
        "correlation_id": correlation_id or request_id or ambient_request_id or new_correlation_id("corr"),
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
