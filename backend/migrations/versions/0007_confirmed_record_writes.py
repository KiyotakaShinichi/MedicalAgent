"""Add confirmed support-chat record write audit table.

Revision ID: 0007_confirmed_record_writes
Revises: 0006_trace_diagnostics_fields
Create Date: 2026-07-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0007_confirmed_record_writes"
down_revision = "0006_trace_diagnostics_fields"
branch_labels = None
depends_on = None

TABLE = "patient_record_write_audits"


def upgrade() -> None:
    bind = op.get_bind()
    if TABLE in inspect(bind).get_table_names():
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.String(), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("record_type", sa.String(), nullable=False),
        sa.Column("record_id", sa.Integer(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("record_fingerprint", sa.String(), nullable=False),
        sa.Column("source_chat_message_id", sa.Integer(), sa.ForeignKey("chat_messages.id"), nullable=True),
        sa.Column("source_message", sa.Text(), nullable=False),
        sa.Column("confirmation_message", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("provenance_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="saved"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("undone_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_record_write_patient", TABLE, ["patient_id"])
    op.create_index("ix_record_write_type", TABLE, ["record_type"])
    op.create_index("ix_record_write_target", TABLE, ["record_id"])
    op.create_index("ix_record_write_idempotency", TABLE, ["idempotency_key"], unique=True)
    op.create_index("ix_record_write_fingerprint", TABLE, ["record_fingerprint"])
    op.create_index("ix_record_write_status", TABLE, ["status"])


def downgrade() -> None:
    bind = op.get_bind()
    if TABLE in inspect(bind).get_table_names():
        op.drop_table(TABLE)
