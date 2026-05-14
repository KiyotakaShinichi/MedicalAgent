"""extend clinical_summary_reviews with review_target / reason_category / model_version / rag_version

Revision ID: 0002_clinical_review_extensions
Revises: 0001_baseline
Create Date: 2026-05-14 00:00:01

Why this exists
---------------
The clinician feedback loop accepts seven canonical decisions
(``approved`` / ``edited`` / ``rejected`` / ``unsafe`` / ``missing_evidence``
/ ``wrong_escalation`` / ``needs_followup``) and persists a target type,
a reason category, and the model + RAG versions that produced the AI
output being reviewed. Those four columns live on
``clinical_summary_reviews``.

Previously ``backend/schema_migrations.py`` patched the columns in via
``ALTER TABLE ... ADD COLUMN`` on app startup. That worked, but had no
version history, no rollback path, and a linter occasionally reverted the
column additions. This Alembic revision is the canonical record of the
change going forward.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_clinical_review_extensions"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CLINICAL_REVIEW_TABLE = "clinical_summary_reviews"
NEW_COLUMNS = (
    ("review_target", sa.Column("review_target", sa.String(), nullable=True, server_default="summary")),
    ("reason_category", sa.Column("reason_category", sa.String(), nullable=True)),
    ("model_version", sa.Column("model_version", sa.String(), nullable=True)),
    ("rag_version", sa.Column("rag_version", sa.String(), nullable=True)),
)


def _existing_columns(table: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    """Add the four new columns if they are not already present."""
    existing = _existing_columns(CLINICAL_REVIEW_TABLE)
    with op.batch_alter_table(CLINICAL_REVIEW_TABLE) as batch_op:
        for name, column in NEW_COLUMNS:
            if name not in existing:
                batch_op.add_column(column)


def downgrade() -> None:
    """Drop the four columns. SQLite supports this only via batch ops."""
    with op.batch_alter_table(CLINICAL_REVIEW_TABLE) as batch_op:
        for name, _ in reversed(NEW_COLUMNS):
            try:
                batch_op.drop_column(name)
            except Exception:
                # Safe in dev: dropping a missing column is a no-op.
                pass
