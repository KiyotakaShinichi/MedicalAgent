from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import AsyncTask
from backend.services.automation_job_queue import enqueue_automation_task, requeue_automation_task


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_explicit_idempotency_key_reuses_task():
    db = _db()
    try:
        first = enqueue_automation_task(db, job_type="refresh_trace_envelope_v2_eval", requested_by="admin", dry_run=True, idempotency_key="same-request-123")
        second = enqueue_automation_task(db, job_type="refresh_trace_envelope_v2_eval", requested_by="admin", dry_run=True, idempotency_key="same-request-123")
        assert first["id"] == second["id"]
        assert second["idempotent_reuse"] is True
        assert db.query(AsyncTask).count() == 1
    finally:
        db.close()


def test_dead_lettered_job_requires_audited_requeue():
    db = _db()
    try:
        task = enqueue_automation_task(db, job_type="refresh_trace_envelope_v2_eval", requested_by="admin", dry_run=True)
        row = db.query(AsyncTask).filter(AsyncTask.id == task["id"]).one()
        row.status = "dead_lettered"
        row.attempts = 3
        db.commit()
        requeued = requeue_automation_task(db, task["id"], requested_by="admin")
        assert requeued["status"] == "queued"
        assert requeued["requeue_history"][0]["prior_attempts"] == 3
        assert requeued["clinical_validation"] is False
    finally:
        db.close()
