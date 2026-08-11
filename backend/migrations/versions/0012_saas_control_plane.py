"""Add tenant-scoped SaaS control-plane tables.

Revision ID: 0012_saas_control_plane
Revises: 0011_llm_usage_telemetry
Create Date: 2026-08-10

The control plane is synthetic-only and intentionally separate from the
legacy patient-demo tables. It does not make those tables multi-tenant or
healthcare-production ready.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0012_saas_control_plane"
down_revision = "0011_llm_usage_telemetry"
branch_labels = None
depends_on = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.func.now(),
        )
    ]


def upgrade() -> None:
    required_tables = {
        "saas_organizations",
        "saas_memberships",
        "saas_projects",
        "saas_environments",
        "saas_entitlements",
        "saas_usage_events",
        "saas_platform_jobs",
        "saas_outbox_events",
        "saas_audit_events",
    }
    existing_tables = set(inspect(op.get_bind()).get_table_names())
    if required_tables <= existing_tables:
        # The repository baseline intentionally creates current ORM metadata on
        # a fresh database. In that path these tables already exist.
        return
    partial = required_tables & existing_tables
    if partial:
        raise RuntimeError(
            "Partial SaaS control-plane schema detected; refusing an ambiguous migration: "
            f"{sorted(partial)}"
        )
    op.create_table(
        "saas_organizations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("plan_code", sa.String(), nullable=False, server_default="engineering_preview"),
        sa.Column("data_class", sa.String(), nullable=False, server_default="synthetic_only"),
        sa.Column("created_by_subject", sa.String(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("slug", name="uq_saas_organizations_slug"),
    )
    op.create_index("ix_saas_organizations_slug", "saas_organizations", ["slug"], unique=True)
    op.create_index("ix_saas_organizations_status", "saas_organizations", ["status"])
    op.create_index("ix_saas_organizations_created_by_subject", "saas_organizations", ["created_by_subject"])

    op.create_table(
        "saas_memberships",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        *_timestamps(),
        sa.UniqueConstraint("organization_id", "subject", name="uq_saas_membership_org_subject"),
    )
    op.create_index("ix_saas_memberships_organization_id", "saas_memberships", ["organization_id"])
    op.create_index("ix_saas_memberships_subject", "saas_memberships", ["subject"])
    op.create_index("ix_saas_memberships_role", "saas_memberships", ["role"])
    op.create_index("ix_saas_memberships_status", "saas_memberships", ["status"])

    op.create_table(
        "saas_projects",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("data_class", sa.String(), nullable=False, server_default="synthetic_only"),
        sa.Column("created_by_subject", sa.String(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("organization_id", "slug", name="uq_saas_project_org_slug"),
    )
    op.create_index("ix_saas_projects_organization_id", "saas_projects", ["organization_id"])
    op.create_index("ix_saas_projects_status", "saas_projects", ["status"])
    op.create_index("ix_saas_projects_created_by_subject", "saas_projects", ["created_by_subject"])

    op.create_table(
        "saas_environments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("project_id", sa.String(), sa.ForeignKey("saas_projects.id"), nullable=False),
        sa.Column("environment_key", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("retrieval_profile", sa.String(), nullable=False, server_default="sparse_governed"),
        sa.Column("data_class", sa.String(), nullable=False, server_default="synthetic_only"),
        *_timestamps(),
        sa.UniqueConstraint("project_id", "environment_key", name="uq_saas_environment_project_key"),
    )
    op.create_index("ix_saas_environments_organization_id", "saas_environments", ["organization_id"])
    op.create_index("ix_saas_environments_project_id", "saas_environments", ["project_id"])
    op.create_index("ix_saas_environments_status", "saas_environments", ["status"])

    op.create_table(
        "saas_entitlements",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("metric_key", sa.String(), nullable=False),
        sa.Column("unit", sa.String(), nullable=False),
        sa.Column("hard_limit", sa.Float(), nullable=False),
        sa.Column("soft_limit", sa.Float(), nullable=True),
        sa.Column("period", sa.String(), nullable=False, server_default="monthly"),
        sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source", sa.String(), nullable=False, server_default="engineering_preview"),
        *_timestamps(),
        sa.UniqueConstraint("organization_id", "metric_key", name="uq_saas_entitlement_org_metric"),
    )
    op.create_index("ix_saas_entitlements_organization_id", "saas_entitlements", ["organization_id"])
    op.create_index("ix_saas_entitlements_metric_key", "saas_entitlements", ["metric_key"])

    op.create_table(
        "saas_usage_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("project_id", sa.String(), sa.ForeignKey("saas_projects.id"), nullable=True),
        sa.Column("environment_id", sa.String(), sa.ForeignKey("saas_environments.id"), nullable=True),
        sa.Column("metric_key", sa.String(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("billable", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("provider_request_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint("organization_id", "idempotency_key", name="uq_saas_usage_org_idempotency"),
    )
    for column in ("organization_id", "project_id", "environment_id", "metric_key", "source", "billable", "provider_request_id", "occurred_at"):
        op.create_index(f"ix_saas_usage_events_{column}", "saas_usage_events", [column])

    op.create_table(
        "saas_platform_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("project_id", sa.String(), sa.ForeignKey("saas_projects.id"), nullable=False),
        sa.Column("environment_id", sa.String(), sa.ForeignKey("saas_environments.id"), nullable=True),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("progress_percent", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("created_by_subject", sa.String(), nullable=False),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(), nullable=True),
        sa.Column("lease_token", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recovery_count", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint("organization_id", "idempotency_key", name="uq_saas_job_org_idempotency"),
        sa.UniqueConstraint("lease_token", name="uq_saas_platform_jobs_lease_token"),
    )
    for column in ("organization_id", "project_id", "environment_id", "job_type", "status", "created_by_subject", "queued_at", "available_at", "lease_owner", "lease_token", "lease_expires_at"):
        op.create_index(f"ix_saas_platform_jobs_{column}", "saas_platform_jobs", [column])

    op.create_table(
        "saas_outbox_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("project_id", sa.String(), sa.ForeignKey("saas_projects.id"), nullable=True),
        sa.Column("aggregate_type", sa.String(), nullable=False),
        sa.Column("aggregate_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("lease_owner", sa.String(), nullable=True),
        sa.Column("lease_token", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recovery_count", sa.Integer(), nullable=False, server_default="0"),
        *_timestamps(),
        sa.UniqueConstraint("organization_id", "idempotency_key", name="uq_saas_outbox_org_idempotency"),
        sa.UniqueConstraint("lease_token", name="uq_saas_outbox_events_lease_token"),
    )
    for column in ("organization_id", "project_id", "aggregate_type", "aggregate_id", "event_type", "status", "available_at", "lease_owner", "lease_token", "lease_expires_at"):
        op.create_index(f"ix_saas_outbox_events_{column}", "saas_outbox_events", [column])

    op.create_table(
        "saas_audit_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("saas_organizations.id"), nullable=False),
        sa.Column("project_id", sa.String(), sa.ForeignKey("saas_projects.id"), nullable=True),
        sa.Column("actor_subject", sa.String(), nullable=False),
        sa.Column("actor_role", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("target_type", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=True),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=True),
        *_timestamps(),
    )
    for column in ("organization_id", "project_id", "actor_subject", "actor_role", "action", "request_id", "created_at"):
        op.create_index(f"ix_saas_audit_events_{column}", "saas_audit_events", [column])


def downgrade() -> None:
    existing_tables = set(inspect(op.get_bind()).get_table_names())
    for table in (
        "saas_audit_events",
        "saas_outbox_events",
        "saas_platform_jobs",
        "saas_usage_events",
        "saas_entitlements",
        "saas_environments",
        "saas_projects",
        "saas_memberships",
        "saas_organizations",
    ):
        if table in existing_tables:
            op.drop_table(table)
