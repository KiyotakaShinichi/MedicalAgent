"""Add auditable high-risk support-chat review alerts.

Revision ID: 0008_high_risk_conversation_alerts
Revises: 0007_confirmed_record_writes
Create Date: 2026-07-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0008_high_risk_conversation_alerts"
down_revision = "0007_confirmed_record_writes"
branch_labels = None
depends_on = None

TABLE = "high_risk_conversation_alerts"


def upgrade() -> None:
    bind = op.get_bind()
    if TABLE in inspect(bind).get_table_names():
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.String(), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("source_chat_message_id", sa.Integer(), sa.ForeignKey("chat_messages.id"), nullable=False),
        sa.Column("assistant_chat_message_id", sa.Integer(), sa.ForeignKey("chat_messages.id"), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("trigger_summary", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("notification_channel", sa.String(), nullable=True),
        sa.Column("notification_status", sa.String(), nullable=False, server_default="disabled"),
        sa.Column("notification_event_id", sa.String(), nullable=True),
        sa.Column("notification_error", sa.Text(), nullable=True),
        sa.Column("acknowledged_by_role", sa.String(), nullable=True),
        sa.Column("acknowledgement_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("notified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
    )
    for name, columns, unique in (
        ("ix_high_risk_alert_patient", ["patient_id"], False),
        ("ix_high_risk_alert_source_chat", ["source_chat_message_id"], False),
        ("ix_high_risk_alert_assistant_chat", ["assistant_chat_message_id"], False),
        ("ix_high_risk_alert_idempotency", ["idempotency_key"], True),
        ("ix_high_risk_alert_category", ["category"], False),
        ("ix_high_risk_alert_severity", ["severity"], False),
        ("ix_high_risk_alert_status", ["status"], False),
        ("ix_high_risk_alert_notification_status", ["notification_status"], False),
        ("ix_high_risk_alert_event", ["notification_event_id"], False),
    ):
        op.create_index(name, TABLE, columns, unique=unique)


def downgrade() -> None:
    bind = op.get_bind()
    if TABLE in inspect(bind).get_table_names():
        op.drop_table(TABLE)
