from __future__ import annotations

import subprocess
import sys

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
