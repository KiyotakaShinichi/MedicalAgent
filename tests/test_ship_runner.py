from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import ship


def test_ship_step_uses_bounded_timeout(monkeypatch):
    observed = {}

    def fake_run(command, *, cwd, env, check, timeout):
        observed.update(
            {
                "command": command,
                "cwd": cwd,
                "check": check,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    result = ship._run(
        ship.Step(
            name="fixture",
            command=[sys.executable, "-c", "print('ok')"],
            timeout_seconds=45,
        )
    )

    assert observed["check"] is True
    assert observed["timeout"] == 45
    assert result["status"] == "passed"
    assert result["timeout_seconds"] == 45


def test_ship_manifest_keeps_nonclinical_boundary(tmp_path, monkeypatch):
    target = tmp_path / "ship.json"
    monkeypatch.setattr(ship, "SHIP_MANIFEST", target)

    ship._write_manifest(
        status="failed",
        step_results=[{"name": "fixture", "status": "timed_out"}],
        failed_step="fixture",
        failure_kind="timeout",
    )

    text = target.read_text(encoding="utf-8")
    assert '"clinical_validation": false' in text
    assert '"healthcare_production_ready": false' in text
    assert '"failure_kind": "timeout"' in text


def test_ship_tiers_keep_fast_and_evidence_surfaces_distinct():
    steps = ship._build_steps()
    fast = ship._select_steps(steps, "fast")
    evidence = ship._select_steps(steps, "evidence")
    release = ship._select_steps(steps, "release")
    assert fast
    assert evidence
    assert len(release) > len(fast)
    assert all(step.name in ship.FAST_STEP_NAMES for step in fast)
    assert all(ship._is_evidence_step(step) for step in evidence)
    assert "Frontend Playwright smoke" not in {step.name for step in fast}


def test_dependency_fingerprint_changes_with_relevant_source(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    step = ship.Step(name="fixture", command=[sys.executable, "source.py"])
    monkeypatch.setattr(
        ship, "_candidate_dependency_paths", lambda _step: [source]
    )
    first = ship._dependency_fingerprint(step)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = ship._dependency_fingerprint(step)
    assert first != second


def test_resume_reuses_only_matching_passed_fingerprint():
    step = ship.Step(name="fixture", command=[sys.executable, "-V"])
    previous = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "steps": [
            {
                "name": "fixture",
                "status": "passed",
                "dependency_fingerprint": "same",
            }
        ],
    }
    cached = ship._cached_result(previous, step, "same")
    assert cached is not None
    assert cached["status"] == "cached_pass"
    assert ship._cached_result(previous, step, "changed") is None


def test_nonpassing_step_is_never_resumed():
    step = ship.Step(name="fixture", command=[sys.executable, "-V"])
    previous = {
        "steps": [
            {
                "name": "fixture",
                "status": "failed",
                "dependency_fingerprint": "same",
            }
        ]
    }
    assert ship._cached_result(previous, step, "same") is None
