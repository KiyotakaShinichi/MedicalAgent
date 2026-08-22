"""PHI-safe event recording for the evidence release boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, MutableMapping

from backend.services.rag_evidence.types import AuthorizationDecision


def record_event(
    result: MutableMapping[str, Any],
    event_name: str,
    envelope: dict[str, Any],
    decision: AuthorizationDecision,
) -> None:
    events = result.setdefault("evidence_envelope_events", [])
    if not isinstance(events, list):
        raise TypeError("evidence_event_sink_malformed")
    events.append({
        "event": event_name,
        "request_id": envelope.get("request_id"),
        "disposition": decision.disposition.value,
        "reason": decision.reason,
        "evidence_required": bool(envelope.get("evidence_required")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def record_simple_event(
    result: MutableMapping[str, Any],
    event_name: str,
    *,
    request_id: str,
    evidence_required: bool | None,
    reason: str | None = None,
) -> None:
    events = result.setdefault("evidence_envelope_events", [])
    if not isinstance(events, list):
        raise TypeError("evidence_event_sink_malformed")
    event = {
        "event": event_name,
        "request_id": request_id,
        "evidence_required": evidence_required if isinstance(evidence_required, bool) else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if reason:
        event["reason"] = str(reason)[:160]
    events.append(event)
