from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import ChatMessage, HighRiskAlertDeliveryAttempt, HighRiskConversationAlert, Patient
from backend.services.emotional_distress_detection import detect_emotional_distress
from backend.services.high_risk_conversation_alerts import (
    build_alert_automation_status,
    classify_alert_trigger,
    process_due_alert_deliveries,
    queue_and_dispatch_alert,
    record_delivery_receipt,
    requeue_dead_letter_alert,
    serialize_alert,
)
from backend.services.n8n_webhook_dispatcher import build_signed_dispatch


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'alerts.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_mortality_language_creates_local_alert_with_external_delivery_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("N8N_WEBHOOK_DISPATCH_ENABLED", raising=False)
    db = _session(tmp_path)
    try:
        db.add(Patient(id="PX", name="Synthetic Patient", diagnosis="doctor-confirmed"))
        db.flush()
        chat = ChatMessage(patient_id="PX", role="user", message="I will not last", intent="patient_support")
        db.add(chat)
        db.flush()

        alert, action = queue_and_dispatch_alert(
            db,
            patient_id="PX",
            source_chat_message_id=chat.id,
            immediate_danger=True,
            urgent_flags=["immediate_danger_statement"],
            emotional_distress=detect_emotional_distress("I will not last"),
        )
        db.commit()

        assert alert is not None
        assert action is not None
        assert action["type"] == "high_risk_review_alert"
        assert action["external_notification_status"] == "disabled"
        assert "not configured" in action["message"]
        stored = db.query(HighRiskConversationAlert).one()
        assert stored.patient_id == "PX"
        assert stored.notification_status == "disabled"
        assert "I will not last" not in stored.trigger_summary
        assert serialize_alert(stored)["delivery_claim_boundary"]
    finally:
        db.close()


def test_ordinary_support_turn_does_not_create_alert():
    assert classify_alert_trigger(
        immediate_danger=False,
        urgent_flags=[],
        emotional_distress=detect_emotional_distress("What does CBC mean?"),
    ) is None


def test_redacted_alert_dispatch_contract_rejects_patient_identity():
    safe = build_signed_dispatch(
        workflow_id="high_risk_review_alert",
        payload={
            "alert_id": 42,
            "event_type": "high_priority_review_item",
            "priority": "urgent_review",
            "review_path": "/clinician/high-risk-conversation-alerts/42",
            "delivery_scope": "redacted_internal_review_notification",
        },
        secret="test-secret",
    )
    assert safe["envelope"]["phi_allowed"] is False

    try:
        build_signed_dispatch(
            workflow_id="high_risk_review_alert",
            payload={"alert_id": 42, "patient_id": "PX"},
            secret="test-secret",
        )
    except ValueError as exc:
        assert "Blocked payload fields" in str(exc)
    else:
        raise AssertionError("patient_id must be rejected from external alert payloads")


def test_delivery_retry_reaches_dead_letter_without_losing_local_alert(tmp_path):
    db = _session(tmp_path)
    env = {
        "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
        "N8N_WEBHOOK_BASE_URL": "http://127.0.0.1:5678/webhook/nlcare",
        "N8N_WEBHOOK_SIGNING_SECRET": "test-secret",
        "NLCARE_ALERT_TEST_RECIPIENT_ONLY": "true",
        "NLCARE_ALERT_NOTIFICATION_MAX_ATTEMPTS": "2",
        "NLCARE_ALERT_NOTIFICATION_RETRY_BASE_SECONDS": "1",
    }
    try:
        db.add(Patient(id="PX", name="Synthetic Patient", diagnosis="doctor-confirmed"))
        db.flush()
        chat = ChatMessage(patient_id="PX", role="user", message="synthetic crisis phrase", intent="patient_support")
        db.add(chat)
        db.flush()

        def failing_transport(*_):
            raise TimeoutError("private transport detail")

        alert, _ = queue_and_dispatch_alert(
            db,
            patient_id="PX",
            source_chat_message_id=chat.id,
            immediate_danger=True,
            urgent_flags=["immediate_danger_statement"],
            emotional_distress=detect_emotional_distress("I will not last"),
            env=env,
            transport=failing_transport,
        )
        assert alert.notification_status == "retry_scheduled"
        assert alert.notification_error == "TimeoutError"
        result = process_due_alert_deliveries(
            db,
            now=alert.next_notification_retry_at + timedelta(seconds=1),
            env=env,
            transport=failing_transport,
        )
        db.commit()

        assert result["processed"] == 1
        assert alert.notification_status == "dead_lettered"
        assert alert.notification_attempt_count == 2
        assert alert.status == "queued"
        assert db.query(HighRiskAlertDeliveryAttempt).count() == 2
        assert "private transport detail" not in str(serialize_alert(alert))
    finally:
        db.close()


def test_channel_receipt_is_not_clinician_acknowledgement_and_duplicates_do_not_resend(tmp_path):
    db = _session(tmp_path)
    calls = []
    env = {
        "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
        "N8N_WEBHOOK_BASE_URL": "http://127.0.0.1:5678/webhook/nlcare",
        "N8N_WEBHOOK_SIGNING_SECRET": "test-secret",
        "NLCARE_ALERT_TEST_RECIPIENT_ONLY": "true",
    }

    def transport(url, body, headers, timeout):
        calls.append(headers["X-NLCare-Event-ID"])
        return {"status_code": 202}

    try:
        db.add(Patient(id="PX", name="Synthetic Patient", diagnosis="doctor-confirmed"))
        db.flush()
        chat = ChatMessage(patient_id="PX", role="user", message="synthetic crisis phrase", intent="patient_support")
        db.add(chat)
        db.flush()
        kwargs = dict(
            patient_id="PX",
            source_chat_message_id=chat.id,
            immediate_danger=True,
            urgent_flags=["immediate_danger_statement"],
            emotional_distress=detect_emotional_distress("I will not last"),
            env=env,
            transport=transport,
        )
        alert, _ = queue_and_dispatch_alert(db, **kwargs)
        duplicate, duplicate_action = queue_and_dispatch_alert(db, **kwargs)
        assert alert.id == duplicate.id
        assert len(calls) == 1
        assert "did not create or send another" in duplicate_action["message"]

        record_delivery_receipt(
            db,
            event_id=alert.notification_event_id,
            receipt_id="receipt-1",
            delivery_status="accepted",
        )
        record_delivery_receipt(
            db,
            event_id=alert.notification_event_id,
            receipt_id="receipt-1",
            delivery_status="delivered",
        )
        db.commit()

        assert alert.notification_status == "delivered_to_channel"
        assert alert.status == "notified"
        assert alert.acknowledged_at is None
        assert alert.acknowledged_by_role is None
        assert "do not prove" in serialize_alert(alert)["delivery_claim_boundary"]
    finally:
        db.close()


def test_delivery_receipts_reject_backward_transition_and_implausible_clock(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(Patient(id="PR", name="Synthetic Patient", diagnosis="doctor-confirmed"))
        db.flush()
        chat = ChatMessage(patient_id="PR", role="user", message="synthetic crisis phrase", intent="patient_support")
        db.add(chat)
        db.flush()
        now = datetime.now(timezone.utc)
        alert = HighRiskConversationAlert(
            patient_id="PR",
            source_chat_message_id=chat.id,
            idempotency_key="receipt-state-test",
            category="crisis_language",
            severity="critical_review",
            trigger_summary="Synthetic review item.",
            status="notified",
            notification_status="accepted_by_workflow",
            notification_event_id="event-receipt-state",
            notification_attempt_count=1,
            notification_max_attempts=3,
            last_notification_attempt_at=now,
            delivery_receipt_status="awaiting_receipt",
        )
        db.add(alert)
        db.flush()
        record_delivery_receipt(
            db,
            event_id="event-receipt-state",
            receipt_id="receipt-state",
            delivery_status="delivered",
            occurred_at=now + timedelta(seconds=1),
            received_at=now + timedelta(seconds=2),
        )
        with pytest.raises(ValueError, match="Invalid delivery receipt transition"):
            record_delivery_receipt(
                db,
                event_id="event-receipt-state",
                receipt_id="receipt-state",
                delivery_status="accepted",
                occurred_at=now + timedelta(seconds=3),
                received_at=now + timedelta(seconds=3),
            )

        alert.delivery_receipt_id = None
        alert.delivery_receipt_status = "awaiting_receipt"
        alert.delivery_receipt_at = None
        with pytest.raises(ValueError, match="future"):
            record_delivery_receipt(
                db,
                event_id="event-receipt-state",
                receipt_id="receipt-future",
                delivery_status="accepted",
                occurred_at=now + timedelta(hours=1),
                received_at=now,
            )
    finally:
        db.close()


def test_dead_letter_status_and_manual_requeue_remain_nonclinical(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(Patient(id="PX", name="Synthetic Patient", diagnosis="doctor-confirmed"))
        db.flush()
        chat = ChatMessage(patient_id="PX", role="user", message="synthetic crisis phrase", intent="patient_support")
        db.add(chat)
        db.flush()
        alert = HighRiskConversationAlert(
            patient_id="PX",
            source_chat_message_id=chat.id,
            idempotency_key="dead-letter-test",
            category="crisis_language",
            severity="critical_review",
            trigger_summary="Crisis-language route requires prompt human review.",
            status="queued",
            notification_status="dead_lettered",
            notification_attempt_count=3,
            notification_max_attempts=3,
            delivery_receipt_status="failed",
            dead_letter_reason="TimeoutError",
        )
        db.add(alert)
        db.flush()

        before = build_alert_automation_status(db)
        assert before["dead_lettered"] == 1
        assert before["delivery_receipt_is_human_acknowledgement"] is False
        assert before["monitored_emergency_service"] is False
        assert before["attention_threshold_is_clinical_sla"] is False

        requeued = requeue_dead_letter_alert(db, alert_id=alert.id, requested_by_role="admin")
        db.commit()
        assert requeued.notification_status == "retry_scheduled"
        assert requeued.notification_attempt_count == 0
        assert requeued.dead_letter_reason is None
        audit = db.query(HighRiskAlertDeliveryAttempt).filter(
            HighRiskAlertDeliveryAttempt.alert_id == alert.id,
            HighRiskAlertDeliveryAttempt.status == "manual_requeue_requested",
        ).one()
        assert audit.error_code == "requested_by_admin"
    finally:
        db.close()


def test_automation_status_surfaces_age_without_claiming_a_clinical_sla(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(Patient(id="PA", name="Synthetic Patient", diagnosis="doctor-confirmed"))
        db.flush()
        chat = ChatMessage(patient_id="PA", role="user", message="synthetic distress", intent="patient_support")
        db.add(chat)
        db.flush()
        now = datetime.now(timezone.utc)
        db.add(HighRiskConversationAlert(
            patient_id="PA",
            source_chat_message_id=chat.id,
            idempotency_key="age-visibility-test",
            category="crisis_language",
            severity="critical_review",
            trigger_summary="Synthetic review item.",
            status="queued",
            notification_status="disabled",
            delivery_receipt_status="not_received",
            created_at=now - timedelta(minutes=20),
        ))
        db.flush()

        report = build_alert_automation_status(
            db,
            now=now,
            operator_attention_after_seconds=900,
        )

        assert report["open_older_than_attention_threshold"] == 1
        assert report["oldest_open_age_seconds"] >= 1200
        assert report["attention_threshold_is_clinical_sla"] is False
        assert "not a clinical response-time commitment" in report["claim_boundary"]
    finally:
        db.close()
