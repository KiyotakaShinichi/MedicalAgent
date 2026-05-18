"""Add optional family-history depth fields.

Revision ID: 0004_family_history_depth_fields
Revises: 0003_prediction_trace_and_rag_phase11
Create Date: 2026-05-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0004_family_history_depth_fields"
down_revision = "0003_prediction_trace_and_rag_phase11"
branch_labels = None
depends_on = None

TABLE = "family_cancer_history_records"
FIELDS = {
    "bilateral_breast_cancer": sa.String(),
    "multiple_primary_cancers": sa.String(),
    "ancestry_ethnicity": sa.String(),
    "prior_breast_biopsy_atypia": sa.String(),
    "relation_degree": sa.String(),
}


def _columns() -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if TABLE not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(TABLE)}


def upgrade() -> None:
    existing = _columns()
    if not existing:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        for name, col_type in FIELDS.items():
            if name not in existing:
                batch_op.add_column(sa.Column(name, col_type, nullable=True))


def downgrade() -> None:
    existing = _columns()
    if not existing:
        return
    with op.batch_alter_table(TABLE) as batch_op:
        for name in reversed(tuple(FIELDS)):
            if name in existing:
                batch_op.drop_column(name)
