from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from backend.services.n8n_automation_templates import BLOCKED_PAYLOAD_FIELDS


CLAIM_BOUNDARY = (
    "Signed n8n dispatch is restricted to redacted engineering events. It must not carry PHI, raw patient "
    "messages, raw prompts or responses, private chain-of-thought, or clinical instructions. It is not clinical "
    "validation, healthcare production readiness, or permission to automate patient-facing decisions."
)

ALLOWED_WORKFLOW_IDS = frozenset(
    {
        "release_gate_alert",
        "stale_artifact_ticket",
        "reviewer_intake_reminder",
        "eval_refresh_trigger",
        "trace_quality_digest",
        "pinecone_shadow_report",
        "external_red_team_intake",
        "dependency_security_alert",
        "deployment_health_alert",
        "high_risk_review_alert",
    }
)


def find_blocked_fields(payload: Any, *, prefix: str = "") -> list[str]:
    blocked = set(BLOCKED_PAYLOAD_FIELDS)
    found: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in blocked:
                found.append(path)
            found.extend(find_blocked_fields(value, prefix=path))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            found.extend(find_blocked_fields(value, prefix=path))
    return sorted(set(found))


def build_signed_dispatch(
    *,
    workflow_id: str,
    payload: Mapping[str, Any],
    secret: str,
    timestamp: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    if workflow_id not in ALLOWED_WORKFLOW_IDS:
        raise ValueError(f"Unsupported n8n workflow_id={workflow_id}")
    if not secret:
        raise ValueError("A non-empty signing secret is required")
    blocked_fields = find_blocked_fields(payload)
    if blocked_fields:
        raise ValueError(f"Blocked payload fields present: {blocked_fields}")

    envelope = {
        "schema_version": "nlcare_n8n_event_v1",
        "event_id": event_id or str(uuid.uuid4()),
        "workflow_id": workflow_id,
        "created_at": timestamp or datetime.now(timezone.utc).isoformat(),
        "payload": dict(payload),
        "payload_redacted": True,
        "phi_allowed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    canonical_body = _canonical_json(envelope)
    signature = hmac.new(secret.encode("utf-8"), canonical_body, hashlib.sha256).hexdigest()
    return {
        "envelope": envelope,
        "body": canonical_body.decode("utf-8"),
        "headers": {
            "Content-Type": "application/json",
            "X-NLCare-Event-ID": envelope["event_id"],
            "X-NLCare-Timestamp": envelope["created_at"],
            "X-NLCare-Signature-Algorithm": "hmac-sha256",
            "X-NLCare-Signature": signature,
        },
    }


def verify_signed_dispatch(*, body: str | bytes, signature: str, secret: str) -> bool:
    raw = body.encode("utf-8") if isinstance(body, str) else body
    expected = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def validate_signed_dispatch_envelope(
    *,
    body: str | bytes,
    signature: str,
    secret: str,
    now: datetime | None = None,
    max_age_seconds: int = 300,
    seen_event_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate HMAC, timestamp freshness, redaction, and optional replay state."""
    if not verify_signed_dispatch(body=body, signature=signature, secret=secret):
        return {"valid": False, "reason": "invalid_signature", "event_id": None}
    try:
        raw = body.decode("utf-8") if isinstance(body, bytes) else body
        envelope = json.loads(raw)
        event_id = str(envelope.get("event_id") or "")
        created_at = _parse_timestamp(str(envelope.get("created_at") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"valid": False, "reason": "invalid_envelope", "event_id": None}
    current = now or datetime.now(timezone.utc)
    if not event_id:
        return {"valid": False, "reason": "missing_event_id", "event_id": None}
    if created_at > current + timedelta(seconds=30):
        return {"valid": False, "reason": "future_timestamp", "event_id": event_id}
    if (current - created_at).total_seconds() > max(1, max_age_seconds):
        return {"valid": False, "reason": "expired", "event_id": event_id}
    if envelope.get("phi_allowed") is not False or envelope.get("payload_redacted") is not True:
        return {"valid": False, "reason": "redaction_boundary_failed", "event_id": event_id}
    if find_blocked_fields(envelope.get("payload") or {}):
        return {"valid": False, "reason": "blocked_payload_field", "event_id": event_id}
    if seen_event_ids is not None and event_id in seen_event_ids:
        return {"valid": False, "reason": "replay", "event_id": event_id}
    if seen_event_ids is not None:
        seen_event_ids.add(event_id)
    return {"valid": True, "reason": "accepted", "event_id": event_id, "envelope": envelope}


def dispatch_signed_webhook(
    *,
    workflow_id: str,
    payload: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    transport: Callable[[str, str, Mapping[str, str], float], Mapping[str, Any]] | None = None,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    values = dict(os.environ if env is None else env)
    enabled = _truthy(values.get("N8N_WEBHOOK_DISPATCH_ENABLED"))
    secret = str(values.get("N8N_WEBHOOK_SIGNING_SECRET") or "")
    base_url = str(values.get("N8N_WEBHOOK_BASE_URL") or "").rstrip("/")
    blocked_fields = find_blocked_fields(payload)
    if blocked_fields:
        raise ValueError(f"Blocked payload fields present: {blocked_fields}")

    if not enabled:
        return {
            "status": "disabled_dry_run",
            "sent": False,
            "workflow_id": workflow_id,
            "payload_redacted": True,
            "blocked_payload_fields": [],
            "clinical_validation": False,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    if not base_url or not secret:
        raise ValueError("n8n dispatch requires N8N_WEBHOOK_BASE_URL and N8N_WEBHOOK_SIGNING_SECRET")
    if workflow_id == "high_risk_review_alert" and not _truthy(values.get("NLCARE_ALERT_TEST_RECIPIENT_ONLY")):
        raise ValueError("High-risk alert delivery is restricted to a synthetic test recipient in this prototype")

    _validate_webhook_url(base_url)
    signed = build_signed_dispatch(workflow_id=workflow_id, payload=payload, secret=secret)
    endpoint = f"{base_url}/{workflow_id}"
    sender = transport or _urllib_transport
    response = sender(endpoint, signed["body"], signed["headers"], timeout_seconds)
    return {
        "status": "sent",
        "sent": True,
        "workflow_id": workflow_id,
        "event_id": signed["envelope"]["event_id"],
        "response": dict(response),
        "payload_redacted": True,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_signed_receipt(
    *,
    event_id: str,
    receipt_id: str,
    delivery_status: str,
    secret: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    status = str(delivery_status or "").strip().lower()
    if status not in {"accepted", "delivered", "failed"}:
        raise ValueError(f"Unsupported delivery receipt status={status}")
    if not secret:
        raise ValueError("A non-empty signing secret is required")
    receipt = {
        "schema_version": "nlcare_n8n_receipt_v1",
        "event_id": str(event_id),
        "receipt_id": str(receipt_id),
        "delivery_status": status,
        "occurred_at": timestamp or datetime.now(timezone.utc).isoformat(),
        "payload_redacted": True,
        "phi_allowed": False,
        "clinical_validation": False,
    }
    body = _canonical_json(receipt)
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return {
        "receipt": receipt,
        "body": body.decode("utf-8"),
        "headers": {
            "Content-Type": "application/json",
            "X-NLCare-Receipt-Signature": signature,
            "X-NLCare-Timestamp": receipt["occurred_at"],
        },
    }


def validate_signed_receipt(
    *,
    body: str | bytes,
    signature: str,
    secret: str,
    now: datetime | None = None,
    max_age_seconds: int = 300,
) -> dict[str, Any]:
    raw = body.encode("utf-8") if isinstance(body, str) else body
    expected = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return {"valid": False, "reason": "invalid_signature"}
    try:
        receipt = json.loads(raw.decode("utf-8"))
        occurred_at = _parse_timestamp(str(receipt.get("occurred_at") or ""))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {"valid": False, "reason": "invalid_receipt"}
    current = now or datetime.now(timezone.utc)
    if occurred_at > current + timedelta(seconds=30):
        return {"valid": False, "reason": "future_timestamp"}
    if (current - occurred_at).total_seconds() > max(1, max_age_seconds):
        return {"valid": False, "reason": "expired"}
    required = ("event_id", "receipt_id", "delivery_status")
    if any(not str(receipt.get(field) or "").strip() for field in required):
        return {"valid": False, "reason": "missing_required_field"}
    if receipt.get("delivery_status") not in {"accepted", "delivered", "failed"}:
        return {"valid": False, "reason": "invalid_delivery_status"}
    if receipt.get("phi_allowed") is not False or receipt.get("payload_redacted") is not True:
        return {"valid": False, "reason": "redaction_boundary_failed"}
    if find_blocked_fields(receipt):
        return {"valid": False, "reason": "blocked_payload_field"}
    return {"valid": True, "reason": "accepted", "receipt": receipt}


def _urllib_transport(url: str, body: str, headers: Mapping[str, str], timeout: float) -> Mapping[str, Any]:
    request = Request(url, data=body.encode("utf-8"), headers=dict(headers), method="POST")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is validated before this call.
        return {"status_code": int(response.status), "reason": str(response.reason)}


def _validate_webhook_url(value: str) -> None:
    parsed = urlparse(value)
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and parsed.hostname in local_hosts:
        return
    raise ValueError("n8n webhook URL must use HTTPS, except localhost development URLs")


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "ALLOWED_WORKFLOW_IDS",
    "CLAIM_BOUNDARY",
    "build_signed_dispatch",
    "build_signed_receipt",
    "dispatch_signed_webhook",
    "find_blocked_fields",
    "validate_signed_receipt",
    "validate_signed_dispatch_envelope",
    "verify_signed_dispatch",
]
