from __future__ import annotations

import threading
import time

from backend.services import patient_report_enrichment_jobs as jobs


class _FakeSession:
    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _wait_for(patient_id: str, status: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = jobs.get_patient_enrichment_job(patient_id)
        if snapshot and snapshot["status"] == status:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job {patient_id} did not reach {status}")


def setup_function():
    jobs.reset_patient_enrichment_jobs_for_tests()


def test_single_flight_queues_only_one_build(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(jobs, "SessionLocal", lambda: session)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def build(patient_id, db):
        calls.append(patient_id)
        started.set()
        assert release.wait(1.0)
        return {"patient_id": patient_id}

    first = jobs.schedule_patient_enrichment("PX", build=build)
    assert first["status"] in {"queued", "running"}
    assert started.wait(1.0)
    second = jobs.schedule_patient_enrichment("PX", build=build)
    assert second["status"] == "running"
    release.set()
    completed = _wait_for("PX", "complete")

    assert calls == ["PX"]
    assert completed["clinical_validation"] is False
    assert session.committed is True
    assert session.closed is True


def test_invalidation_discards_stale_result(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(jobs, "SessionLocal", lambda: session)
    started = threading.Event()
    release = threading.Event()
    discarded = []

    def build(patient_id, db):
        started.set()
        assert release.wait(1.0)
        return {"patient_id": patient_id, "stale": True}

    jobs.schedule_patient_enrichment("PX", build=build, discard_stale_result=lambda pid, row: discarded.append((pid, row)))
    assert started.wait(1.0)
    jobs.invalidate_patient_enrichment("PX")
    release.set()

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not discarded:
        time.sleep(0.01)
    assert discarded == [("PX", {"patient_id": "PX", "stale": True})]
    assert jobs.get_patient_enrichment_job("PX") is None


def test_failure_is_bounded_and_does_not_expose_message(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(jobs, "SessionLocal", lambda: session)

    def build(patient_id, db):
        raise RuntimeError("secret internal detail")

    jobs.schedule_patient_enrichment("PX", build=build)
    failed = _wait_for("PX", "failed")

    assert failed["error_code"] == "RuntimeError"
    assert "secret internal detail" not in str(failed)
    assert session.rolled_back is True
    assert session.closed is True
