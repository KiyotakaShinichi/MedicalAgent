"""Add structured LLM token-usage telemetry to RAG evaluation logs.

Revision ID: 0011_llm_usage_telemetry
Revises: 0010_automation_task_leasing
Create Date: 2026-07-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0011_llm_usage_telemetry"
down_revision = "0010_automation_task_leasing"
branch_labels = None
depends_on = None

TABLE = "rag_evaluation_logs"


def upgrade() -> None:
    existing = {column["name"] for column in inspect(op.get_bind()).get_columns(TABLE)}
    if "token_usage_json" not in existing:
        with op.batch_alter_table(TABLE) as batch:
            batch.add_column(sa.Column("token_usage_json", sa.Text(), nullable=True))


def downgrade() -> None:
    existing = {column["name"] for column in inspect(op.get_bind()).get_columns(TABLE)}
    if "token_usage_json" in existing:
        with op.batch_alter_table(TABLE) as batch:
            batch.drop_column("token_usage_json")
