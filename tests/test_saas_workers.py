from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import SaaSOutboxEvent, SaaSPlatformJob
from backend.services.saas_control_plane import (
    SaaSActor,
    append_outbox_event,
    create_organization,
    create_project,
    enqueue_platform_job,
)
from backend.services.saas_job_worker import (
    claim_next_platform_job,
    execute_platform_job,
    recover_expired_platform_jobs,
)
from backend.services.saas_outbox_dispatcher import (
    claim_next_outbox_event,
    dispatch_outbox_event,
    recover_expired_outbox_events,
    run_outbox_once,
)


def _db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _workspace(db):
    actor = SaaSActor(subject="oidc|worker-owner", application_role="admin", auth_source="oidc")
    organization = create_organization(db, actor=actor, name="Worker Team")
    project = create_project(db, organization_id=organization.id, actor=actor, name="Assurance Suite")
    db.commit()
    return actor, organization, project


def test_platform_worker_claims_with_lease_and_completes_dry_run():
    db = _db()
    actor, organization, project = _workspace(db)
    job, _ = enqueue_platform_job(
        db,
        organization_id=organization.id,
        project_id=project.id,
        actor=actor,
        job_type="rag_baseline_comparison",
        idempotency_key="worker-dry-run",
        payload={"dry_run": True, "suite_ref": "repository-default"},
    )
    db.commit()

    claimed = claim_next_platform_job(db, worker_id="worker-1", lease_seconds=600)
    assert claimed is not None
    assert claimed["id"] == job.id
    assert claimed["status"] == "running"
    assert claimed["lease_token"].startswith("lease_")
    with pytest.raises(PermissionError):
        execute_platform_job(db, job_id=job.id, lease_token="wrong")

    completed = execute_platform_job(db, job_id=job.id, lease_token=claimed["lease_token"])
    assert completed["status"] == "completed"
    assert completed["result"]["commands_executed"] is False
    assert completed["clinical_validation"] is False


def test_platform_worker_fails_closed_and_dead_letters_disabled_execution():
    db = _db()
    actor, organization, project = _workspace(db)
    job, _ = enqueue_platform_job(
        db,
        organization_id=organization.id,
        project_id=project.id,
        actor=actor,
        job_type="release_gate",
        idempotency_key="worker-disabled-execution",
        payload={"dry_run": False},
    )
    job.max_attempts = 1
    db.commit()
    claimed = claim_next_platform_job(db, worker_id="worker-2")

    result = execute_platform_job(
        db,
        job_id=job.id,
        lease_token=claimed["lease_token"],
        environment={"NLCARE_SAAS_JOB_EXECUTION_ENABLED": "false"},
    )
    assert result["status"] == "dead_lettered"
    assert "disabled" in result["error_message"].lower()


def test_expired_platform_job_lease_is_recovered():
    db = _db()
    actor, organization, project = _workspace(db)
    job, _ = enqueue_platform_job(
        db,
        organization_id=organization.id,
        project_id=project.id,
        actor=actor,
        job_type="agent_workflow_eval",
        idempotency_key="worker-recovery",
        payload={"dry_run": True},
    )
    job.status = "running"
    job.lease_owner = "lost-worker"
    job.lease_token = "lost-lease"
    job.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    assert recover_expired_platform_jobs(db) == 1
    db.refresh(job)
    assert job.status == "queued"
    assert job.recovery_count == 1
    assert job.lease_token is None


def test_outbox_dispatch_is_leased_signed_path_and_redacted():
    db = _db()
    _, organization, project = _workspace(db)
    db.query(SaaSOutboxEvent).update({"status": "dispatched"})
    event = append_outbox_event(
        db,
        organization_id=organization.id,
        project_id=project.id,
        aggregate_type="platform_job",
        aggregate_id="job_test",
        event_type="evaluation.job.completed",
        payload={"job_id": "job_test", "status": "completed"},
        idempotency_key="outbox-dispatch-test",
    )
    db.commit()
    captured = {}

    def fake_dispatcher(**kwargs):
        captured.update(kwargs)
        return {"sent": True, "status": "sent", "event_id": kwargs["event_id"]}

    claimed = claim_next_outbox_event(db, worker_id="outbox-1")
    result = dispatch_outbox_event(
        db,
        event_id=event.id,
        lease_token=claimed["lease_token"],
        environment={"N8N_WEBHOOK_DISPATCH_ENABLED": "true"},
        dispatcher=fake_dispatcher,
    )
    assert result["status"] == "dispatched"
    assert captured["workflow_id"] == "saas_workspace_event"
    assert captured["payload"]["organization_id"] == organization.id
    assert "patient_id" not in str(captured["payload"])
    assert captured["payload"]["clinical_validation"] is False


def test_disabled_outbox_delivery_does_not_claim_or_burn_attempts():
    db = _db()
    _workspace(db)
    assert run_outbox_once(
        db,
        worker_id="outbox-disabled",
        environment={"N8N_WEBHOOK_DISPATCH_ENABLED": "false"},
    ) is None
    assert db.query(SaaSOutboxEvent).filter(SaaSOutboxEvent.status == "pending").count() > 0
    assert db.query(SaaSOutboxEvent).filter(SaaSOutboxEvent.attempts > 0).count() == 0


def test_expired_outbox_lease_is_recovered():
    db = _db()
    _workspace(db)
    event = db.query(SaaSOutboxEvent).first()
    event.status = "dispatching"
    event.lease_owner = "lost-outbox"
    event.lease_token = "lost-outbox-lease"
    event.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    assert recover_expired_outbox_events(db) == 1
    db.refresh(event)
    assert event.status == "pending"
    assert event.recovery_count == 1
    assert event.lease_token is None
