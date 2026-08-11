from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import SaaSAuditEvent, SaaSEntitlement, SaaSOutboxEvent
from backend.services.saas_control_plane import (
    SaaSAccessError,
    SaaSActor,
    SaaSQuotaExceeded,
    SaaSValidationError,
    bootstrap_demo_workspace,
    cancel_platform_job,
    create_organization,
    create_project,
    enqueue_platform_job,
    list_organizations_for_actor,
    list_platform_jobs,
    record_usage_event,
    require_membership,
    sanitize_job_payload,
    usage_summary,
    workspace_overview,
)


def _db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _actor(subject: str = "oidc|owner", role: str = "admin", source: str = "oidc") -> SaaSActor:
    return SaaSActor(subject=subject, application_role=role, auth_source=source)


def test_organization_project_environment_and_entitlements_are_created_atomically():
    db = _db()
    actor = _actor()
    organization = create_organization(db, actor=actor, name="Assurance Team")
    project = create_project(
        db,
        organization_id=organization.id,
        actor=actor,
        name="RAG Evaluation",
        description="Synthetic evaluation only.",
    )
    db.commit()

    overview = workspace_overview(db, organization_id=organization.id, actor=actor)
    assert overview["organization"]["data_class"] == "synthetic_only"
    assert overview["membership_role"] == "owner"
    assert overview["projects"][0]["id"] == project.id
    assert overview["projects"][0]["environments"][0]["key"] == "synthetic-staging"
    assert {item["metric_key"] for item in overview["usage"]} >= {
        "evaluation_runs", "provider_tokens", "storage_bytes", "vector_count",
    }
    assert overview["billing_enabled"] is False
    assert overview["clinical_validation"] is False


def test_cross_tenant_membership_and_queries_fail_closed():
    db = _db()
    first = _actor("oidc|first")
    second = _actor("oidc|second")
    first_org = create_organization(db, actor=first, name="First Team")
    second_org = create_organization(db, actor=second, name="Second Team")
    db.commit()

    try:
        require_membership(db, organization_id=second_org.id, actor=first)
    except SaaSAccessError as exc:
        assert "not found or access" in str(exc)
    else:
        raise AssertionError("Cross-tenant membership must fail closed")

    assert [item["id"] for item in list_organizations_for_actor(db, first)] == [first_org.id]


def test_jobs_are_scoped_idempotent_audited_and_cancelable():
    db = _db()
    actor = _actor()
    organization = create_organization(db, actor=actor, name="Job Team")
    project = create_project(db, organization_id=organization.id, actor=actor, name="Agent Eval")
    db.commit()

    job, reused = enqueue_platform_job(
        db,
        organization_id=organization.id,
        project_id=project.id,
        actor=actor,
        job_type="agent_workflow_eval",
        idempotency_key="agent-eval-run-0001",
        payload={"suite_ref": "repository-default", "dry_run": True},
    )
    duplicate, duplicate_reused = enqueue_platform_job(
        db,
        organization_id=organization.id,
        project_id=project.id,
        actor=actor,
        job_type="agent_workflow_eval",
        idempotency_key="agent-eval-run-0001",
        payload={"suite_ref": "repository-default", "dry_run": True},
    )
    db.commit()

    assert reused is False
    assert duplicate_reused is True
    assert duplicate.id == job.id
    assert len(list_platform_jobs(db, organization_id=organization.id, actor=actor)) == 1
    assert db.query(SaaSAuditEvent).filter(SaaSAuditEvent.organization_id == organization.id).count() >= 3
    assert db.query(SaaSOutboxEvent).filter(SaaSOutboxEvent.organization_id == organization.id).count() >= 3

    cancelled = cancel_platform_job(db, organization_id=organization.id, job_id=job.id, actor=actor)
    db.commit()
    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_at is not None


def test_usage_ledger_is_idempotent_and_never_billing_authoritative():
    db = _db()
    actor = _actor()
    organization = create_organization(db, actor=actor, name="Usage Team")
    db.commit()

    event, reused = record_usage_event(
        db,
        organization_id=organization.id,
        metric_key="provider_tokens",
        quantity=250,
        unit="tokens",
        source="provider_usage_reconciliation",
        idempotency_key="provider-request-0001",
        provider_request_id="provider-123",
        metadata={"model": "synthetic-test-model"},
    )
    duplicate, duplicate_reused = record_usage_event(
        db,
        organization_id=organization.id,
        metric_key="provider_tokens",
        quantity=250,
        unit="tokens",
        source="provider_usage_reconciliation",
        idempotency_key="provider-request-0001",
        provider_request_id="provider-123",
    )
    db.commit()

    assert reused is False
    assert duplicate_reused is True
    assert duplicate.id == event.id
    assert event.billable == 0
    tokens = next(item for item in usage_summary(db, organization_id=organization.id, actor=actor) if item["metric_key"] == "provider_tokens")
    assert tokens["used"] == 250
    assert tokens["billing_authoritative"] is False


def test_usage_ledger_rejects_cross_tenant_project_relation():
    db = _db()
    first = _actor("oidc|first")
    second = _actor("oidc|second")
    first_org = create_organization(db, actor=first, name="First Usage Team")
    second_org = create_organization(db, actor=second, name="Second Usage Team")
    second_project = create_project(
        db,
        organization_id=second_org.id,
        actor=second,
        name="Second Project",
    )
    db.commit()

    try:
        record_usage_event(
            db,
            organization_id=first_org.id,
            project_id=second_project.id,
            metric_key="provider_tokens",
            quantity=10,
            unit="tokens",
            source="test",
            idempotency_key="cross-tenant-relation",
        )
    except SaaSAccessError:
        pass
    else:
        raise AssertionError("Cross-tenant project relations must fail closed")


def test_entitlement_hard_limit_blocks_new_usage_and_projects():
    db = _db()
    actor = _actor()
    organization = create_organization(db, actor=actor, name="Quota Team")
    db.commit()
    entitlements = {item.metric_key: item for item in db.query(SaaSEntitlement).all()}
    entitlements["project_count"].hard_limit = 1
    entitlements["evaluation_runs"].hard_limit = 1
    project = create_project(db, organization_id=organization.id, actor=actor, name="Only Project")
    enqueue_platform_job(
        db,
        organization_id=organization.id,
        project_id=project.id,
        actor=actor,
        job_type="release_gate",
        idempotency_key="release-gate-0001",
        payload={"dry_run": True},
    )
    db.commit()

    try:
        create_project(db, organization_id=organization.id, actor=actor, name="Blocked Project")
    except SaaSQuotaExceeded:
        pass
    else:
        raise AssertionError("Project quota should be enforced")

    try:
        enqueue_platform_job(
            db,
            organization_id=organization.id,
            project_id=project.id,
            actor=actor,
            job_type="release_gate",
            idempotency_key="release-gate-0002",
            payload={"dry_run": True},
        )
    except SaaSQuotaExceeded:
        pass
    else:
        raise AssertionError("Evaluation-run quota should be enforced")


def test_job_payload_rejects_phi_prompt_and_patient_fields_recursively():
    for payload in (
        {"patient_id": "P001"},
        {"raw_prompt": "hidden"},
        {"config": {"email_address": "demo@example.invalid"}},
        {"message": "I feel unwell"},
    ):
        try:
            sanitize_job_payload(payload)
        except SaaSValidationError:
            pass
        else:
            raise AssertionError(f"Payload should have been rejected: {payload}")

    assert sanitize_job_payload({"suite_ref": "frozen-v1", "dry_run": True}) == {
        "suite_ref": "frozen-v1",
        "dry_run": True,
    }


def test_demo_bootstrap_is_repeatable_and_does_not_create_duplicate_resources():
    db = _db()
    actor = _actor("demo:admin:global", source="demo_session")
    first = bootstrap_demo_workspace(db, actor)
    second = bootstrap_demo_workspace(db, actor)
    db.commit()

    assert first is not None and second is not None
    assert first.id == second.id
    organizations = list_organizations_for_actor(db, actor)
    assert len(organizations) == 1
    overview = workspace_overview(db, organization_id=first.id, actor=actor)
    assert len(overview["projects"]) == 1
    assert overview["projects"][0]["slug"] == "breast-monitoring-demo"
