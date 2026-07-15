"""Add durable retry, receipt, dead-letter, and attempt evidence for alerts.

Revision ID: 0009_alert_delivery_controls
Revises: 0008_high_risk_conversation_alerts
Create Date: 2026-07-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0009_alert_delivery_controls"
down_revision = "0008_high_risk_conversation_alerts"
branch_labels = None
depends_on = None

ALERT_TABLE = "high_risk_conversation_alerts"
ATTEMPT_TABLE = "high_risk_alert_delivery_attempts"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_columns = {column["name"] for column in inspector.get_columns(ALERT_TABLE)}
    additions = (
        ("notification_attempt_count", sa.Integer(), False, "0"),
        ("notification_max_attempts", sa.Integer(), False, "3"),
        ("last_notification_attempt_at", sa.DateTime(timezone=True), True, None),
        ("next_notification_retry_at", sa.DateTime(timezone=True), True, None),
        ("delivery_receipt_status", sa.String(), False, "'not_received'"),
        ("delivery_receipt_id", sa.String(), True, None),
        ("delivery_receipt_at", sa.DateTime(timezone=True), True, None),
        ("dead_lettered_at", sa.DateTime(timezone=True), True, None),
        ("dead_letter_reason", sa.Text(), True, None),
    )
    with op.batch_alter_table(ALERT_TABLE) as batch:
        for name, column_type, nullable, default in additions:
            if name not in existing_columns:
                batch.add_column(sa.Column(name, column_type, nullable=nullable, server_default=default))
        for name, columns, unique in (
            ("ix_high_risk_alert_next_retry", ["next_notification_retry_at"], False),
            ("ix_high_risk_alert_receipt_status", ["delivery_receipt_status"], False),
            ("ix_high_risk_alert_receipt_id", ["delivery_receipt_id"], True),
        ):
            if name not in {item["name"] for item in inspector.get_indexes(ALERT_TABLE)}:
                batch.create_index(name, columns, unique=unique)

    if ATTEMPT_TABLE not in inspect(bind).get_table_names():
        op.create_table(
            ATTEMPT_TABLE,
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("alert_id", sa.Integer(), sa.ForeignKey(f"{ALERT_TABLE}.id"), nullable=False),
            sa.Column("attempt_number", sa.Integer(), nullable=False),
            sa.Column("event_id", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("error_code", sa.String(), nullable=True),
            sa.Column("response_status_code", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_high_risk_attempt_alert", ATTEMPT_TABLE, ["alert_id"])
        op.create_index("ix_high_risk_attempt_event", ATTEMPT_TABLE, ["event_id"])
        op.create_index("ix_high_risk_attempt_status", ATTEMPT_TABLE, ["status"])


def downgrade() -> None:
    bind = op.get_bind()
    if ATTEMPT_TABLE in inspect(bind).get_table_names():
        op.drop_table(ATTEMPT_TABLE)
    existing_columns = {column["name"] for column in inspect(bind).get_columns(ALERT_TABLE)}
    with op.batch_alter_table(ALERT_TABLE) as batch:
        for name in (
            "dead_letter_reason",
            "dead_lettered_at",
            "delivery_receipt_at",
            "delivery_receipt_id",
            "delivery_receipt_status",
            "next_notification_retry_at",
            "last_notification_attempt_at",
            "notification_max_attempts",
            "notification_attempt_count",
        ):
            if name in existing_columns:
                batch.drop_column(name)
