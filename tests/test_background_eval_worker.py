from __future__ import annotations

from backend.services.background_eval_worker import (
    ALLOWED_JOB_TYPES,
    BLOCKED_JOB_TYPES,
    build_background_eval_worker_dry_run,
    execute_job,
    enqueue_job,
)


def test_allowed_job_is_accepted_and_redacted():
    job = enqueue_job(
        job_type="run_release_gate",
        requested_by="admin",
        payload={"run_id": "r1", "requested_by": "admin", "reason": "manual", "patient_name": "Blocked"},
    )

    assert job["accepted"] is False
    assert job["rejected_reason"] == "blocked_payload_fields_present"
    assert "patient_name" in job["blocked_payload_fields"]
    assert "patient_name" not in job["sanitized_payload"]
    assert job["clinical_validation"] is False


def test_allowed_job_without_phi_is_accepted():
    job = enqueue_job(
        job_type="refresh_trace_envelope_v2_eval",
        requested_by="admin",
        payload={"run_id": "r2", "requested_by": "admin", "reason": "refresh"},
    )

    assert job["accepted"] is True
    assert job["command"] == ["python", "scripts/run_trace_envelope_v2_eval.py"]
    assert job["payload_redacted"] is True


def test_blocked_clinical_job_is_rejected():
    for job_type in BLOCKED_JOB_TYPES:
        job = enqueue_job(job_type=job_type, requested_by="test", payload={})
        assert job["accepted"] is False
        assert job["rejected_reason"] == "blocked_clinical_or_phi_action"
        assert job["command"] is None


def test_unknown_job_is_rejected():
    job = enqueue_job(job_type="invented_job", requested_by="test", payload={})

    assert job["accepted"] is False
    assert job["rejected_reason"] == "unknown_job_type"


def test_nested_blocked_payload_is_rejected():
    job = enqueue_job(
        job_type="run_release_gate",
        requested_by="test",
        payload={"metadata": {"patient_id": "P001"}},
    )
    assert job["accepted"] is False
    assert job["blocked_payload_fields"] == ["metadata.patient_id"]


def test_local_command_dry_run_does_not_execute():
    job = enqueue_job(
        job_type="run_release_gate",
        requested_by="test",
        payload={"run_id": "dry"},
        dry_run=True,
    )
    result = execute_job(job, env={})
    assert result["status"] == "dry_run_completed"
    assert result["commands_executed"] is False


def test_live_command_requires_explicit_enable_flag():
    job = enqueue_job(
        job_type="run_release_gate",
        requested_by="test",
        payload={"run_id": "disabled"},
        dry_run=False,
    )
    try:
        execute_job(job, env={})
    except PermissionError as exc:
        assert "disabled" in str(exc).lower()
    else:
        raise AssertionError("Expected live command execution to remain disabled")


def test_background_eval_worker_dry_run_artifact(tmp_path):
    report = build_background_eval_worker_dry_run(
        output_path=tmp_path / "worker.json",
        doc_path=tmp_path / "worker.md",
    )

    assert report["status"] == "strong"
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["phi_allowed"] is False
    assert report["live_patient_route_enabled"] is False
    assert report["commands_executed"] is False
    assert report["accepted_job_count"] >= 2
    assert report["rejected_job_count"] >= 1
    assert set(ALLOWED_JOB_TYPES) <= set(report["allowed_job_types"])
    assert "not clinical validation" in report["claim_boundary"]
