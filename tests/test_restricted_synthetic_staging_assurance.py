from __future__ import annotations

import json
import subprocess

from backend.services.restricted_synthetic_staging_assurance import (
    MINIMUM_TESTS,
    build_report,
    run_assurance,
)


def _dependency_artifact(root, *, status="acceptable", high=0, unaccepted=0):
    path = root / "Data/evals/ops/latest_dependency_security_scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": status,
        "summary": {
            "high_or_critical_count": high,
            "unaccepted_known_vulnerability_count": unaccepted,
        },
    }), encoding="utf-8")


def test_report_requires_tests_and_clean_dependency_evidence(tmp_path):
    _dependency_artifact(tmp_path)
    report = build_report(
        return_code=0,
        output=f"{MINIMUM_TESTS + 2} passed in 1.0s",
        duration_seconds=1.0,
        root=tmp_path,
    )
    assert report["status"] == "passed"
    assert report["controls"]["quarantine_before_promotion"] is True
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False


def test_report_fails_when_dependency_evidence_is_not_clean(tmp_path):
    _dependency_artifact(tmp_path, status="needs_attention", high=1, unaccepted=1)
    report = build_report(
        return_code=0,
        output=f"{MINIMUM_TESTS + 2} passed in 1.0s",
        duration_seconds=1.0,
        root=tmp_path,
    )
    assert report["status"] == "failed"


def test_report_fails_when_test_floor_is_not_met(tmp_path):
    _dependency_artifact(tmp_path)
    report = build_report(
        return_code=0,
        output=f"{MINIMUM_TESTS - 1} passed in 1.0s",
        duration_seconds=1.0,
        root=tmp_path,
    )
    assert report["status"] == "failed"


def test_runner_persists_fail_closed_result(tmp_path):
    _dependency_artifact(tmp_path)
    output = tmp_path / "assurance.json"

    def fake_executor(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, "20 passed, 1 failed", "")

    report = run_assurance(output_path=output, root=tmp_path, executor=fake_executor)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == persisted["status"] == "failed"
    assert persisted["provenance"]["patient_content_retained"] is False


def test_runner_timeout_is_a_failure(tmp_path):
    _dependency_artifact(tmp_path)

    def timeout_executor(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)

    report = run_assurance(root=tmp_path, output_path=tmp_path / "out.json", executor=timeout_executor)
    assert report["status"] == "failed"
    assert report["summary"]["timed_out"] is True
