"""Local review-alert outbox with an optional redacted n8n dispatch.

The local alert is the source of truth. External notification is disabled by
default and never carries a patient identifier or raw conversation text. A
successful webhook response means only that the configured workflow accepted
the event; it does not mean a clinician saw or acted on it.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from sqlalchemy.orm import Session

from backend.models import HighRiskAlertDeliveryAttempt, HighRiskConversationAlert
from backend.services.n8n_webhook_dispatcher import dispatch_signed_webhook


ALERT_WORKFLOW_ID = "high_risk_review_alert"
RECEIPT_STATUSES = frozenset({"accepted", "delivered", "failed"})


def classify_alert_trigger(
    *,
    immediate_danger: bool,
    urgent_flags: list[str] | None,
    emotional_distress: Any | None,
) -> dict[str, str] | None:
    mode = str(getattr(emotional_distress, "response_mode", "") or "")
    category = str(getattr(emotional_distress, "category", "") or "")
    flags = {str(item) for item in (urgent_flags or [])}
    flags.discard("safety_location_followup")

    if mode == "crisis_support":
        return {
            "category": "crisis_language",
            "severity": "critical_review",
            "trigger_summary": "Crisis-language route requires prompt human review.",
        }
    if immediate_danger:
        return {
            "category": "immediate_danger_language",
            "severity": "critical_review",
            "trigger_summary": "Immediate-danger wording requires prompt human review.",
        }
    if mode == "urgent_clinician_review" or category == "mortality_distress":
        return {
            "category": "mortality_or_severe_distress",
            "severity": "urgent_review",
            "trigger_summary": "Mortality or severe-distress wording requires human review.",
        }
    if flags:
        return {
            "category": "urgent_symptom_language",
            "severity": "urgent_review",
            "trigger_summary": "Urgent symptom wording requires human review.",
        }
    return None


def queue_and_dispatch_alert(
    db: Session,
    *,
    patient_id: str,
    source_chat_message_id: int,
    immediate_danger: bool,
    urgent_flags: list[str] | None,
    emotional_distress: Any | None,
    env: Mapping[str, str] | None = None,
    transport: Callable[[str, str, Mapping[str, str], float], Mapping[str, Any]] | None = None,
) -> tuple[HighRiskConversationAlert | None, dict[str, Any] | None]:
    trigger = classify_alert_trigger(
        immediate_danger=immediate_danger,
        urgent_flags=urgent_flags,
        emotional_distress=emotional_distress,
    )
    if trigger is None:
        return None, None

    idempotency_key = hashlib.sha256(
        f"high-risk-alert:{patient_id}:{source_chat_message_id}:{trigger['category']}".encode("utf-8")
    ).hexdigest()
    alert = (
        db.query(HighRiskConversationAlert)
        .filter(HighRiskConversationAlert.idempotency_key == idempotency_key)
        .first()
    )
    created = alert is None
    if created:
        values = dict(os.environ if env is None else env)
        alert = HighRiskConversationAlert(
            patient_id=patient_id,
            source_chat_message_id=source_chat_message_id,
            idempotency_key=idempotency_key,
            category=trigger["category"],
            severity=trigger["severity"],
            trigger_summary=trigger["trigger_summary"],
            status="queued",
            notification_channel="n8n",
            notification_status="disabled",
            notification_max_attempts=_bounded_int(
                values.get("NLCARE_ALERT_NOTIFICATION_MAX_ATTEMPTS"), default=3, minimum=1, maximum=10
            ),
        )
        db.add(alert)
        db.flush()

    if created:
        notice = attempt_alert_delivery(db, alert=alert, env=env, transport=transport)
    else:
        notice = (
            "This high-priority review item was already present in NLCare. Duplicate chat processing "
            "did not create or send another notification. Do not wait for a reply if you feel unsafe "
            "or in immediate danger."
        )

    db.flush()
    action = {
        "type": "high_risk_review_alert",
        "alert_id": alert.id,
        "severity": alert.severity,
        "local_queue_status": alert.status,
        "external_notification_status": alert.notification_status,
        "message": notice,
    }
    return alert, action


def attempt_alert_delivery(
    db: Session,
    *,
    alert: HighRiskConversationAlert,
    env: Mapping[str, str] | None = None,
    transport: Callable[[str, str, Mapping[str, str], float], Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> str:
    """Attempt one redacted delivery and persist bounded operational evidence."""
    current = now or datetime.now(timezone.utc)
    if alert.status == "acknowledged":
        return "The review item is already acknowledged; no additional notification was sent."
    payload = {
        "alert_id": alert.id,
        "event_type": "high_priority_review_item",
        "priority": alert.severity,
        "review_path": f"/clinician/high-risk-conversation-alerts/{alert.id}",
        "delivery_scope": "redacted_internal_review_notification",
        "recipient_scope": "synthetic_test_recipient_only",
    }
    try:
        dispatch = dispatch_signed_webhook(
            workflow_id=ALERT_WORKFLOW_ID,
            payload=payload,
            env=env,
            transport=transport,
            timeout_seconds=2.0,
        )
        if not dispatch.get("sent"):
            alert.notification_status = "disabled"
            alert.notification_error = None
            alert.next_notification_retry_at = None
            return (
                "A high-priority review item was added to this NLCare demo workspace. External "
                "email, SMS, or Viber delivery is not configured, so do not wait for a reply if "
                "you feel unsafe or in immediate danger."
            )

        attempt_number = int(alert.notification_attempt_count or 0) + 1
        alert.notification_attempt_count = attempt_number
        alert.last_notification_attempt_at = current
        alert.status = "notified"
        alert.notification_status = "accepted_by_workflow"
        alert.notification_event_id = str(dispatch.get("event_id") or "") or None
        alert.notification_error = None
        alert.next_notification_retry_at = None
        alert.delivery_receipt_status = "awaiting_receipt"
        alert.delivery_receipt_id = None
        alert.delivery_receipt_at = None
        alert.dead_lettered_at = None
        alert.dead_letter_reason = None
        alert.notified_at = current
        response_status = (dispatch.get("response") or {}).get("status_code")
        _append_attempt(
            db,
            alert=alert,
            attempt_number=attempt_number,
            event_id=alert.notification_event_id,
            status="accepted_by_workflow",
            response_status_code=int(response_status) if response_status is not None else None,
            completed_at=current,
        )
        return (
            "A high-priority review item was added to NLCare, and the configured redacted "
            "notification workflow accepted the event. This does not confirm channel delivery or "
            "that a clinician has seen it, so do not wait for a reply if you feel unsafe or in immediate danger."
        )
    except Exception as exc:  # noqa: BLE001 - preserve the local outbox even when delivery fails
        attempt_number = int(alert.notification_attempt_count or 0) + 1
        alert.notification_attempt_count = attempt_number
        alert.last_notification_attempt_at = current
        error_code = type(exc).__name__
        alert.notification_error = error_code
        _append_attempt(
            db,
            alert=alert,
            attempt_number=attempt_number,
            event_id=None,
            status="failed",
            error_code=error_code,
            completed_at=current,
        )
        if attempt_number >= int(alert.notification_max_attempts or 3):
            alert.notification_status = "dead_lettered"
            alert.next_notification_retry_at = None
            alert.dead_lettered_at = current
            alert.dead_letter_reason = error_code
        else:
            alert.notification_status = "retry_scheduled"
            alert.next_notification_retry_at = current + timedelta(seconds=_retry_delay_seconds(attempt_number, env))
        return (
            "A high-priority review item was added to NLCare, but the external notification workflow "
            "did not confirm acceptance. The engineering outbox retained the item; do not wait for a "
            "reply if you feel unsafe or in immediate danger."
        )


def process_due_alert_deliveries(
    db: Session,
    *,
    now: datetime | None = None,
    limit: int = 50,
    env: Mapping[str, str] | None = None,
    transport: Callable[[str, str, Mapping[str, str], float], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    rows = (
        db.query(HighRiskConversationAlert)
        .filter(
            HighRiskConversationAlert.notification_status == "retry_scheduled",
            HighRiskConversationAlert.next_notification_retry_at <= current,
        )
        .order_by(HighRiskConversationAlert.next_notification_retry_at.asc())
        .limit(max(1, min(limit, 200)))
        .all()
    )
    for alert in rows:
        attempt_alert_delivery(db, alert=alert, env=env, transport=transport, now=current)
    db.flush()
    return {
        "processed": len(rows),
        "alert_ids": [row.id for row in rows],
        "clinical_validation": False,
        "claim_boundary": "Retry processing is engineering delivery evidence, not proof of clinician review.",
    }


def build_alert_automation_status(
    db: Session,
    *,
    now: datetime | None = None,
    operator_attention_after_seconds: int = 900,
) -> dict[str, Any]:
    """Summarise outbox health while keeping delivery and review distinct."""
    current = now or datetime.now(timezone.utc)
    attention_threshold = max(60, min(int(operator_attention_after_seconds), 86400))
    rows = db.query(HighRiskConversationAlert).all()
    notification_counts: dict[str, int] = {}
    local_counts: dict[str, int] = {}
    for row in rows:
        notification_counts[row.notification_status] = notification_counts.get(row.notification_status, 0) + 1
        local_counts[row.status] = local_counts.get(row.status, 0) + 1
    open_rows = [row for row in rows if row.status != "acknowledged"]
    open_ages = [
        max(0, int((_as_utc(current) - _as_utc(row.created_at)).total_seconds()))
        for row in open_rows
        if row.created_at is not None
    ]
    return {
        "schema_version": "high_risk_alert_automation_status_v2_2026_07",
        "total_alerts": len(rows),
        "local_status_counts": local_counts,
        "notification_status_counts": notification_counts,
        "open_local_alerts": len(open_rows),
        "operator_attention_after_seconds": attention_threshold,
        "open_older_than_attention_threshold": sum(age >= attention_threshold for age in open_ages),
        "oldest_open_age_seconds": max(open_ages, default=0),
        "attention_threshold_is_clinical_sla": False,
        "retry_due": sum(
            1 for row in rows
            if row.notification_status == "retry_scheduled"
            and row.next_notification_retry_at is not None
            and _as_utc(row.next_notification_retry_at) <= _as_utc(current)
        ),
        "awaiting_channel_receipt": sum(1 for row in rows if row.delivery_receipt_status == "awaiting_receipt"),
        "dead_lettered": sum(1 for row in rows if row.notification_status == "dead_lettered"),
        "channel_delivered_but_unacknowledged": sum(
            1 for row in rows
            if row.notification_status == "delivered_to_channel" and row.status != "acknowledged"
        ),
        "human_acknowledged": sum(1 for row in rows if row.status == "acknowledged"),
        "delivery_receipt_is_human_acknowledgement": False,
        "monitored_emergency_service": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Queue and channel telemetry are engineering workflow evidence. They do not prove clinician review, "
            "patient contact, clinical action, or emergency coverage. The attention-age threshold is an operator "
            "visibility control, not a clinical response-time commitment."
        ),
    }


def requeue_dead_letter_alert(
    db: Session,
    *,
    alert_id: int,
    requested_by_role: str,
    now: datetime | None = None,
) -> HighRiskConversationAlert:
    """Explicitly requeue one dead letter and preserve an append-only audit row."""
    alert = db.query(HighRiskConversationAlert).filter(HighRiskConversationAlert.id == alert_id).first()
    if alert is None:
        raise LookupError("Review alert not found")
    if alert.status == "acknowledged":
        raise ValueError("Acknowledged alerts cannot be requeued")
    if alert.notification_status != "dead_lettered":
        raise ValueError("Only dead-lettered alerts can be requeued")
    current = now or datetime.now(timezone.utc)
    prior_attempt_count = int(alert.notification_attempt_count or 0)
    _append_attempt(
        db,
        alert=alert,
        attempt_number=prior_attempt_count + 1,
        event_id=None,
        status="manual_requeue_requested",
        error_code=f"requested_by_{requested_by_role}",
        completed_at=current,
    )
    alert.notification_attempt_count = 0
    alert.notification_status = "retry_scheduled"
    alert.next_notification_retry_at = current
    alert.notification_error = None
    alert.dead_lettered_at = None
    alert.dead_letter_reason = None
    db.flush()
    return alert


def record_delivery_receipt(
    db: Session,
    *,
    event_id: str,
    receipt_id: str,
    delivery_status: str,
    occurred_at: datetime | None = None,
) -> HighRiskConversationAlert:
    status = str(delivery_status or "").strip().lower()
    if status not in RECEIPT_STATUSES:
        raise ValueError(f"Unsupported delivery receipt status={status}")
    alert = (
        db.query(HighRiskConversationAlert)
        .filter(HighRiskConversationAlert.notification_event_id == event_id)
        .first()
    )
    if alert is None:
        raise LookupError("Notification event not found")
    if alert.delivery_receipt_id:
        if alert.delivery_receipt_id == receipt_id and alert.delivery_receipt_status == status:
            return alert
        if not (alert.delivery_receipt_id == receipt_id and alert.delivery_receipt_status == "accepted" and status == "delivered"):
            raise ValueError("A different or invalid delivery receipt transition is already recorded for this event")

    current = occurred_at or datetime.now(timezone.utc)
    alert.delivery_receipt_id = receipt_id
    alert.delivery_receipt_status = status
    alert.delivery_receipt_at = current
    attempt = (
        db.query(HighRiskAlertDeliveryAttempt)
        .filter(HighRiskAlertDeliveryAttempt.event_id == event_id)
        .order_by(HighRiskAlertDeliveryAttempt.id.desc())
        .first()
    )
    if attempt is not None:
        attempt.status = f"receipt_{status}"
        attempt.completed_at = current

    if status == "delivered":
        alert.notification_status = "delivered_to_channel"
        alert.next_notification_retry_at = None
    elif status == "accepted":
        alert.notification_status = "accepted_by_channel"
    elif int(alert.notification_attempt_count or 0) >= int(alert.notification_max_attempts or 3):
        alert.notification_status = "dead_lettered"
        alert.dead_lettered_at = current
        alert.dead_letter_reason = "channel_delivery_failed"
    else:
        alert.notification_status = "retry_scheduled"
        alert.next_notification_retry_at = current + timedelta(seconds=_retry_delay_seconds(alert.notification_attempt_count))
    db.flush()
    return alert


def attach_assistant_message(db: Session, alert_id: int, assistant_chat_message_id: int) -> None:
    alert = db.query(HighRiskConversationAlert).filter(HighRiskConversationAlert.id == alert_id).first()
    if alert is not None:
        alert.assistant_chat_message_id = assistant_chat_message_id
        db.flush()


def serialize_alert(alert: HighRiskConversationAlert) -> dict[str, Any]:
    def iso(value):
        return value.isoformat() if value is not None else None

    return {
        "id": alert.id,
        "patient_id": alert.patient_id,
        "source_chat_message_id": alert.source_chat_message_id,
        "assistant_chat_message_id": alert.assistant_chat_message_id,
        "category": alert.category,
        "severity": alert.severity,
        "trigger_summary": alert.trigger_summary,
        "status": alert.status,
        "notification_channel": alert.notification_channel,
        "notification_status": alert.notification_status,
        "notification_event_id": alert.notification_event_id,
        "notification_attempt_count": alert.notification_attempt_count,
        "notification_max_attempts": alert.notification_max_attempts,
        "last_notification_attempt_at": iso(alert.last_notification_attempt_at),
        "next_notification_retry_at": iso(alert.next_notification_retry_at),
        "delivery_receipt_status": alert.delivery_receipt_status,
        "delivery_receipt_id": alert.delivery_receipt_id,
        "delivery_receipt_at": iso(alert.delivery_receipt_at),
        "dead_lettered_at": iso(alert.dead_lettered_at),
        "dead_letter_reason": alert.dead_letter_reason,
        "acknowledged_by_role": alert.acknowledged_by_role,
        "acknowledgement_note": alert.acknowledgement_note,
        "created_at": iso(alert.created_at),
        "notified_at": iso(alert.notified_at),
        "acknowledged_at": iso(alert.acknowledged_at),
        "delivery_claim_boundary": (
            "Workflow acceptance and channel delivery do not prove that a clinician saw or acted on the review item."
        ),
    }


def _append_attempt(
    db: Session,
    *,
    alert: HighRiskConversationAlert,
    attempt_number: int,
    event_id: str | None,
    status: str,
    error_code: str | None = None,
    response_status_code: int | None = None,
    completed_at: datetime | None = None,
) -> None:
    db.add(
        HighRiskAlertDeliveryAttempt(
            alert_id=alert.id,
            attempt_number=attempt_number,
            event_id=event_id,
            status=status,
            error_code=error_code,
            response_status_code=response_status_code,
            completed_at=completed_at,
        )
    )


def _retry_delay_seconds(attempt_number: int, env: Mapping[str, str] | None = None) -> int:
    values = dict(os.environ if env is None else env)
    base = _bounded_int(values.get("NLCARE_ALERT_NOTIFICATION_RETRY_BASE_SECONDS"), default=30, minimum=1, maximum=900)
    return min(base * (2 ** max(0, int(attempt_number) - 1)), 3600)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


__all__ = [
    "ALERT_WORKFLOW_ID",
    "attach_assistant_message",
    "classify_alert_trigger",
    "attempt_alert_delivery",
    "build_alert_automation_status",
    "process_due_alert_deliveries",
    "queue_and_dispatch_alert",
    "record_delivery_receipt",
    "requeue_dead_letter_alert",
    "serialize_alert",
]
