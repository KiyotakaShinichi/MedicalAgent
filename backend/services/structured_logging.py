"""Structured event logging helpers for prototype observability."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any


LOGGER = logging.getLogger("nlcare.events")


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
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "structured_event_v1",
        "event_type": event_type,
        "severity": severity,
        "request_id": request_id or correlation_id or new_correlation_id(),
        "correlation_id": correlation_id or request_id or new_correlation_id("corr"),
        "user_role": user_role,
        "patient_id": patient_id,
        "artifact_id": artifact_id,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }


def log_event(event_type: str, **kwargs: Any) -> dict[str, Any]:
    event = build_event(event_type, **kwargs)
    LOGGER.info(json.dumps(event, sort_keys=True))
    return event


__all__ = ["build_event", "log_event", "new_correlation_id"]
