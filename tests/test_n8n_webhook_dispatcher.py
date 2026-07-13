from __future__ import annotations

import pytest

from backend.services.n8n_webhook_dispatcher import (
    build_signed_dispatch,
    dispatch_signed_webhook,
    find_blocked_fields,
    verify_signed_dispatch,
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
