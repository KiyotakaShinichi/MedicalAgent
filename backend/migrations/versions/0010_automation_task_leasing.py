"""Add durable leases, retries, and delivery receipts for automation tasks.

Revision ID: 0010_automation_task_leasing
Revises: 0009_alert_delivery_controls
Create Date: 2026-07-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0010_automation_task_leasing"
down_revision = "0009_alert_delivery_controls"
branch_labels = None
depends_on = None

TABLE = "async_tasks"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {column["name"] for column in inspector.get_columns(TABLE)}
    additions = (
        ("available_at", sa.DateTime(timezone=True), True, None),
        ("lease_owner", sa.String(), True, None),
        ("lease_token", sa.String(), True, None),
        ("lease_expires_at", sa.DateTime(timezone=True), True, None),
        ("heartbeat_at", sa.DateTime(timezone=True), True, None),
        ("recovery_count", sa.Integer(), False, "0"),
        ("delivery_event_id", sa.String(), True, None),
        ("delivery_receipt_id", sa.String(), True, None),
        ("delivery_receipt_status", sa.String(), False, "'not_applicable'"),
        ("delivery_receipt_at", sa.DateTime(timezone=True), True, None),
    )
    with op.batch_alter_table(TABLE) as batch:
        for name, column_type, nullable, default in additions:
            if name not in existing:
                batch.add_column(sa.Column(name, column_type, nullable=nullable, server_default=default))
        existing_indexes = {item["name"] for item in inspector.get_indexes(TABLE)}
        for name, columns, unique in (
            ("ix_async_tasks_available_at", ["available_at"], False),
            ("ix_async_tasks_lease_owner", ["lease_owner"], False),
            ("ix_async_tasks_lease_token", ["lease_token"], True),
            ("ix_async_tasks_lease_expires_at", ["lease_expires_at"], False),
            ("ix_async_tasks_delivery_event_id", ["delivery_event_id"], True),
            ("ix_async_tasks_delivery_receipt_id", ["delivery_receipt_id"], True),
            ("ix_async_tasks_delivery_receipt_status", ["delivery_receipt_status"], False),
        ):
            if name not in existing_indexes:
                batch.create_index(name, columns, unique=unique)


def downgrade() -> None:
    existing = {column["name"] for column in inspect(op.get_bind()).get_columns(TABLE)}
    with op.batch_alter_table(TABLE) as batch:
        for name in (
            "delivery_receipt_at",
            "delivery_receipt_status",
            "delivery_receipt_id",
            "delivery_event_id",
            "recovery_count",
            "heartbeat_at",
            "lease_expires_at",
            "lease_token",
            "lease_owner",
            "available_at",
        ):
            if name in existing:
                batch.drop_column(name)
