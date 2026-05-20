"""Add RAG cost/latency observability fields.

Revision ID: 0005_rag_cost_latency_fields
Revises: 0004_family_history_depth_fields
Create Date: 2026-05-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0005_rag_cost_latency_fields"
down_revision = "0004_family_history_depth_fields"
branch_labels = None
depends_on = None

TABLE = "rag_evaluation_logs"
FIELDS = {
    "stage_latency_json": sa.Text(),
    "model_used": sa.String(),
}


def _columns() -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    existing = _columns()
    if not existing:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        for name, column_type in FIELDS.items():
            if name not in existing:
                batch_op.add_column(sa.Column(name, column_type, nullable=True))
    if "model_used" not in existing:
        op.create_index("ix_rag_evaluation_logs_model_used", TABLE, ["model_used"])


def downgrade() -> None:
    existing = _columns()
    if not existing:
        return
    try:
        op.drop_index("ix_rag_evaluation_logs_model_used", table_name=TABLE)
    except Exception:
        pass
    with op.batch_alter_table(TABLE) as batch_op:
        for name in reversed(tuple(FIELDS)):
            if name in existing:
                batch_op.drop_column(name)
