from __future__ import annotations

import json
import subprocess

from backend.services.fail_closed_rag_assurance import (
    MINIMUM_FAULT_CASES,
    build_fail_closed_assurance_report,
    run_fail_closed_assurance,
)


def test_report_passes_only_with_complete_fault_suite(tmp_path):
    report = build_fail_closed_assurance_report(
        return_code=0,
        output=f"{MINIMUM_FAULT_CASES + 38} passed in 8.0s",
        duration_seconds=8.0,
        root=tmp_path,
    )
    assert report["status"] == "passed"
    assert report["summary"]["passed_tests"] == 68
    assert report["coverage"]["all_known_response_paths_enforced"] is True
    assert report["contract"]["only_allow_releases_evidence"] is True
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False


def test_report_fails_closed_on_nonzero_exit_even_with_many_passes(tmp_path):
    report = build_fail_closed_assurance_report(
        return_code=1,
        output="67 passed, 1 failed in 8.0s",
        duration_seconds=8.0,
        root=tmp_path,
    )
    assert report["status"] == "failed"
    assert report["summary"]["failed_tests"] == 1
    assert report["coverage"]["all_known_response_paths_enforced"] is False


def test_report_fails_when_case_floor_is_not_met(tmp_path):
    report = build_fail_closed_assurance_report(
        return_code=0,
        output="29 passed in 2.0s",
        duration_seconds=2.0,
        root=tmp_path,
    )
    assert report["status"] == "failed"


def test_runner_persists_failure_artifact(tmp_path):
    output = tmp_path / "assurance.json"

    def fake_executor(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, "30 passed, 1 failed", "")

    report = run_fail_closed_assurance(
        output_path=output,
        root=tmp_path,
        executor=fake_executor,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == persisted["status"] == "failed"
    assert persisted["provenance"]["raw_patient_content_retained"] is False


def test_runner_timeout_is_a_failed_release_signal(tmp_path):
    output = tmp_path / "assurance.json"

    def timeout_executor(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)

    report = run_fail_closed_assurance(
        output_path=output,
        root=tmp_path,
        executor=timeout_executor,
    )
    assert report["status"] == "failed"
    assert report["summary"]["timed_out"] is True
    assert report["summary"]["pytest_exit_code"] == 124
