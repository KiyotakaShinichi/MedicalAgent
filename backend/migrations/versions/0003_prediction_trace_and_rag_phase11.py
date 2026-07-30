"""add prediction traces and phase 11 rag trace fields

Revision ID: 0003_prediction_trace_and_rag_phase11
Revises: 0002_clinical_review_extensions
Create Date: 2026-05-17 00:00:03

This revision makes the recent observability work real for non-SQLite demo
databases.  The application could create the table/columns through
Base.metadata.create_all during local development, but hosted databases need
an explicit Alembic history.  It also prevents the RAG trace replay endpoint
from silently returning nulls for Phase 11 fields.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_prediction_trace_and_rag_phase11"
down_revision: Union[str, None] = "0002_clinical_review_extensions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PREDICTION_TRACE_TABLE = "prediction_traces"
RAG_TABLE = "rag_evaluation_logs"


def _table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _existing_columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def _existing_indexes(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return set()
    return {
        str(index["name"])
        for index in inspector.get_indexes(table)
        if index.get("name")
    }


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.alter_column(
            "alembic_version",
            "version_num",
            existing_type=sa.String(length=32),
            type_=sa.String(length=128),
            existing_nullable=False,
        )
    tables = _table_names()
    if PREDICTION_TRACE_TABLE not in tables:
        op.create_table(
            PREDICTION_TRACE_TABLE,
            sa.Column("id", sa.Integer(), primary_key=True, index=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
            sa.Column("patient_id", sa.String(), sa.ForeignKey("patients.id"), nullable=True, index=True),
            sa.Column("request_id", sa.String(), nullable=True, index=True),
            sa.Column("actor_role", sa.String(), nullable=True, index=True),
            sa.Column("question", sa.String(), nullable=False, index=True),
            sa.Column("decision", sa.String(), nullable=False, index=True),
            sa.Column("probability", sa.Float(), nullable=True),
            sa.Column("raw_probability", sa.Float(), nullable=True),
            sa.Column("calibrated", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("confidence", sa.String(), nullable=True),
            sa.Column("evidence_sufficiency", sa.String(), nullable=True, index=True),
            sa.Column("abstained", sa.Integer(), nullable=False, server_default="0", index=True),
            sa.Column("abstain_reason", sa.String(), nullable=True),
            sa.Column("modalities_present_json", sa.Text(), nullable=True),
            sa.Column("modalities_missing_json", sa.Text(), nullable=True),
            sa.Column("confidence_modifier", sa.Float(), nullable=True),
            sa.Column("model_version", sa.String(), nullable=False),
            sa.Column("feature_set_version", sa.String(), nullable=True),
            sa.Column("threshold_config_json", sa.Text(), nullable=True),
            sa.Column("calibration_config_json", sa.Text(), nullable=True),
            sa.Column("safety_triggers_json", sa.Text(), nullable=True),
            sa.Column("validator_decision", sa.String(), nullable=True, index=True),
            sa.Column("rag_source_ids_json", sa.Text(), nullable=True),
            sa.Column("timeline_snapshot_hash", sa.String(), nullable=True, index=True),
            sa.Column("notes", sa.Text(), nullable=True),
        )

    if RAG_TABLE in _table_names():
        existing = _existing_columns(RAG_TABLE)
        new_columns = (
            ("rag_mode", sa.Column("rag_mode", sa.String(), nullable=True)),
            ("rewritten_query", sa.Column("rewritten_query", sa.Text(), nullable=True)),
            ("evidence_grade_json", sa.Column("evidence_grade_json", sa.Text(), nullable=True)),
            ("claim_validation_json", sa.Column("claim_validation_json", sa.Text(), nullable=True)),
            ("tier_filter_json", sa.Column("tier_filter_json", sa.Text(), nullable=True)),
            ("post_gen_validator_json", sa.Column("post_gen_validator_json", sa.Text(), nullable=True)),
        )
        with op.batch_alter_table(RAG_TABLE) as batch_op:
            for name, column in new_columns:
                if name not in existing:
                    batch_op.add_column(column)
        if (
            "ix_rag_evaluation_logs_rag_mode"
            not in _existing_indexes(RAG_TABLE)
        ):
            op.create_index("ix_rag_evaluation_logs_rag_mode", RAG_TABLE, ["rag_mode"])


def downgrade() -> None:
    if RAG_TABLE in _table_names():
        with op.batch_alter_table(RAG_TABLE) as batch_op:
            for name in (
                "post_gen_validator_json",
                "tier_filter_json",
                "claim_validation_json",
                "evidence_grade_json",
                "rewritten_query",
                "rag_mode",
            ):
                try:
                    batch_op.drop_column(name)
                except Exception:
                    pass
    if PREDICTION_TRACE_TABLE in _table_names():
        op.drop_table(PREDICTION_TRACE_TABLE)
