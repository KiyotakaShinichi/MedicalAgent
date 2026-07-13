from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.services.automation_control_plane import (
    SCHEDULES,
    build_automation_control_plane,
    build_automation_schedule_plan,
    due_schedule_jobs,
)


def test_schedule_plan_is_redacted_and_not_installed(tmp_path):
    report = build_automation_schedule_plan(output_path=tmp_path / "schedule.json")
    assert report["status"] == "ready_for_scheduler_or_n8n"
    assert report["scheduler_installed"] is False
    assert report["clinical_validation"] is False
    assert report["phi_allowed"] is False
    assert len(report["schedules"]) >= 4


def test_due_schedule_jobs_respects_intervals():
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    last_runs = {schedule["id"]: now for schedule in SCHEDULES}
    assert due_schedule_jobs(now=now, last_run_by_schedule=last_runs) == []

    first = SCHEDULES[0]
    last_runs[first["id"]] = now - timedelta(hours=int(first["interval_hours"]) + 1)
    due = due_schedule_jobs(now=now, last_run_by_schedule=last_runs)
    assert {row["job_type"] for row in due} == set(first["jobs"])
    assert all(row["dry_run"] is True for row in due)


def test_control_plane_build_is_preview_only(tmp_path):
    report = build_automation_control_plane(
        output_path=tmp_path / "control.json",
        schedule_path=tmp_path / "schedule.json",
        doc_path=tmp_path / "control.md",
    )
    assert report["status"] == "strong"
    assert report["commands_executed"] is False
    assert report["webhooks_sent"] is False
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["phi_allowed"] is False
    assert report["accepted_event_count"] == report["event_candidate_count"]
    assert report["rejected_event_count"] == 0
    assert all(job["payload_redacted"] for job in report["event_jobs"])
