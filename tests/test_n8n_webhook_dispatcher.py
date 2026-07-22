from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.services.n8n_webhook_dispatcher import (
    build_signed_dispatch,
    build_signed_receipt,
    dispatch_signed_webhook,
    find_blocked_fields,
    validate_signed_dispatch_envelope,
    validate_signed_receipt,
    verify_signed_dispatch,
    validate_signed_dispatch_envelope_with_keyring,
)


def test_signed_dispatch_verifies_and_tamper_fails():
    signed = build_signed_dispatch(
        workflow_id="release_gate_alert",
        payload={"run_id": "r1", "status": "passed"},
        secret="test-secret",
        timestamp="2026-01-01T00:00:00+00:00",
        event_id="event-1",
    )
    signature = signed["headers"]["X-NLCare-Signature"]
    assert verify_signed_dispatch(body=signed["body"], signature=signature, secret="test-secret")
    assert not verify_signed_dispatch(body=signed["body"] + "x", signature=signature, secret="test-secret")


def test_signing_key_id_supports_bounded_secret_rotation():
    signed = build_signed_dispatch(
        workflow_id="release_gate_alert",
        payload={"status": "warning"},
        secret="new-secret",
        key_id="2026-q3",
    )
    assert signed["headers"]["X-NLCare-Key-ID"] == "2026-q3"
    result = validate_signed_dispatch_envelope_with_keyring(
        body=signed["body"],
        signature=signed["headers"]["X-NLCare-Signature"],
        key_id=signed["headers"]["X-NLCare-Key-ID"],
        secrets={"2026-q2": "old-secret", "2026-q3": "new-secret"},
    )
    assert result["valid"] is True
    unknown = validate_signed_dispatch_envelope_with_keyring(
        body=signed["body"],
        signature=signed["headers"]["X-NLCare-Signature"],
        key_id="unknown",
        secrets={"2026-q3": "new-secret"},
    )
    assert unknown["reason"] == "unknown_key_or_invalid_signature"
    assert signed["envelope"]["phi_allowed"] is False
    assert signed["envelope"]["clinical_validation"] is False


def test_nested_phi_field_is_blocked():
    payload = {"safe": {"nested": [{"raw_patient_message": "do not send"}]}}
    assert find_blocked_fields(payload) == ["safe.nested[0].raw_patient_message"]
    with pytest.raises(ValueError, match="Blocked payload fields"):
        build_signed_dispatch(
            workflow_id="trace_quality_digest",
            payload=payload,
            secret="test-secret",
        )


def test_dispatch_is_disabled_by_default_and_sends_nothing():
    result = dispatch_signed_webhook(
        workflow_id="release_gate_alert",
        payload={"run_id": "r1", "status": "passed"},
        env={},
    )
    assert result["status"] == "disabled_dry_run"
    assert result["sent"] is False
    assert result["clinical_validation"] is False


def test_enabled_dispatch_uses_signed_transport():
    captured = {}

    def transport(url, body, headers, timeout):
        captured.update({"url": url, "body": body, "headers": headers, "timeout": timeout})
        return {"status_code": 200}

    result = dispatch_signed_webhook(
        workflow_id="dependency_security_alert",
        payload={"run_id": "r2", "status": "acceptable"},
        env={
            "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
            "N8N_WEBHOOK_BASE_URL": "http://127.0.0.1:5678/webhook/nlcare",
            "N8N_WEBHOOK_SIGNING_SECRET": "test-secret",
        },
        transport=transport,
    )
    assert result["sent"] is True
    assert captured["url"].endswith("/dependency_security_alert")
    assert verify_signed_dispatch(
        body=captured["body"],
        signature=captured["headers"]["X-NLCare-Signature"],
        secret="test-secret",
    )


def test_nonlocal_plain_http_is_rejected():
    with pytest.raises(ValueError, match="HTTPS"):
        dispatch_signed_webhook(
            workflow_id="release_gate_alert",
            payload={"run_id": "r3"},
            env={
                "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
                "N8N_WEBHOOK_BASE_URL": "http://example.com/webhook",
                "N8N_WEBHOOK_SIGNING_SECRET": "test-secret",
            },
            transport=lambda *_: {"status_code": 200},
        )


def test_envelope_validator_rejects_expiry_and_replay():
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    signed = build_signed_dispatch(
        workflow_id="release_gate_alert",
        payload={"run_id": "r4"},
        secret="test-secret",
        timestamp=(now - timedelta(seconds=10)).isoformat(),
        event_id="event-replay",
    )
    seen = set()
    signature = signed["headers"]["X-NLCare-Signature"]
    accepted = validate_signed_dispatch_envelope(
        body=signed["body"], signature=signature, secret="test-secret", now=now, seen_event_ids=seen
    )
    replay = validate_signed_dispatch_envelope(
        body=signed["body"], signature=signature, secret="test-secret", now=now, seen_event_ids=seen
    )
    expired = validate_signed_dispatch_envelope(
        body=signed["body"], signature=signature, secret="test-secret", now=now + timedelta(minutes=10)
    )
    assert accepted["valid"] is True
    assert replay == {"valid": False, "reason": "replay", "event_id": "event-replay"}
    assert expired["reason"] == "expired"


def test_signed_delivery_receipt_is_fresh_and_redacted():
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    signed = build_signed_receipt(
        event_id="event-1",
        receipt_id="receipt-1",
        delivery_status="delivered",
        secret="test-secret",
        timestamp=now.isoformat(),
    )
    result = validate_signed_receipt(
        body=signed["body"],
        signature=signed["headers"]["X-NLCare-Receipt-Signature"],
        secret="test-secret",
        now=now,
    )
    assert result["valid"] is True
    assert result["receipt"]["phi_allowed"] is False
    assert result["receipt"]["clinical_validation"] is False


def test_high_risk_delivery_requires_synthetic_test_recipient_mode():
    with pytest.raises(ValueError, match="synthetic test recipient"):
        dispatch_signed_webhook(
            workflow_id="high_risk_review_alert",
            payload={"alert_id": 1},
            env={
                "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
                "N8N_WEBHOOK_BASE_URL": "http://127.0.0.1:5678/webhook/nlcare",
                "N8N_WEBHOOK_SIGNING_SECRET": "test-secret",
            },
            transport=lambda *_: {"status_code": 200},
        )
