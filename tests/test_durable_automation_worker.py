from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import AsyncTask
from backend.services.automation_job_queue import enqueue_automation_task
from backend.services.automation_worker import (
    claim_next_automation_task,
    execute_claimed_automation_task,
    heartbeat_automation_task,
    record_automation_delivery_receipt,
    recover_expired_automation_leases,
    run_automation_worker_once,
)


def _sessions(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'automation-worker.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_database_lease_prevents_second_worker_claim(tmp_path):
    sessions = _sessions(tmp_path)
    db = sessions()
    enqueue_automation_task(
        db,
        job_type="refresh_trace_envelope_v2_eval",
        requested_by="admin",
        dry_run=True,
    )
    first = claim_next_automation_task(db, worker_id="worker-a")
    second = claim_next_automation_task(db, worker_id="worker-b")
    assert first is not None
    assert second is None
    assert first["lease_owner"] == "worker-a"
    db.close()


def test_heartbeat_requires_matching_owner_and_token(tmp_path):
    sessions = _sessions(tmp_path)
    db = sessions()
    task = enqueue_automation_task(db, job_type="refresh_trace_envelope_v2_eval", requested_by="admin")
    claimed = claim_next_automation_task(db, worker_id="worker-a", lease_seconds=30)
    assert claimed["id"] == task["id"]
    assert heartbeat_automation_task(
        db,
        task_id=task["id"],
        worker_id="worker-b",
        lease_token=claimed["lease_token"],
    ) is False
    assert heartbeat_automation_task(
        db,
        task_id=task["id"],
        worker_id="worker-a",
        lease_token=claimed["lease_token"],
    ) is True
    db.close()


def test_expired_lease_is_recovered_without_losing_job(tmp_path):
    sessions = _sessions(tmp_path)
    db = sessions()
    task = enqueue_automation_task(db, job_type="refresh_trace_envelope_v2_eval", requested_by="admin")
    row = db.query(AsyncTask).filter(AsyncTask.id == task["id"]).one()
    row.status = "running"
    row.attempts = 1
    row.lease_owner = "crashed-worker"
    row.lease_token = "expired-token"
    row.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()
    recovered = recover_expired_automation_leases(db)
    db.refresh(row)
    assert recovered == {"recovered": 1, "dead_lettered": 0}
    assert row.status == "queued"
    assert row.recovery_count == 1
    assert row.lease_token is None
    db.close()


def test_worker_executes_dry_run_and_clears_lease(tmp_path):
    sessions = _sessions(tmp_path)
    db = sessions()
    task = enqueue_automation_task(
        db,
        job_type="refresh_trace_envelope_v2_eval",
        requested_by="admin",
        dry_run=True,
    )
    db.close()
    result = run_automation_worker_once(session_factory=sessions, worker_id="worker-a", lease_seconds=30)
    assert result["id"] == task["id"]
    assert result["status"] == "completed"
    assert result["result"]["commands_executed"] is False
    assert result["lease_owner"] is None
    assert result["clinical_validation"] is False


def test_receipt_is_persisted_but_not_human_acknowledgement(tmp_path):
    sessions = _sessions(tmp_path)
    db = sessions()
    task = enqueue_automation_task(db, job_type="publish_trace_quality_digest", requested_by="admin")
    row = db.query(AsyncTask).filter(AsyncTask.id == task["id"]).one()
    row.delivery_event_id = "event-1"
    row.delivery_receipt_status = "awaiting_receipt"
    db.commit()
    receipt = record_automation_delivery_receipt(
        db,
        event_id="event-1",
        receipt_id="receipt-1",
        delivery_status="delivered",
        occurred_at=datetime.now(timezone.utc),
    )
    db.commit()
    assert receipt.delivery_receipt_status == "delivered"
    assert receipt.delivery_receipt_id == "receipt-1"
    same = record_automation_delivery_receipt(
        db,
        event_id="event-1",
        receipt_id="receipt-1",
        delivery_status="delivered",
        occurred_at=datetime.now(timezone.utc),
    )
    assert same.id == receipt.id
    with pytest.raises(ValueError):
        record_automation_delivery_receipt(
            db,
            event_id="event-1",
            receipt_id="different",
            delivery_status="failed",
            occurred_at=datetime.now(timezone.utc),
        )
    db.close()
