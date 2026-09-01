from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import bootstrap


def test_run_launches_the_resolved_windows_command(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    monkeypatch.setattr(bootstrap.shutil, "which", lambda command: f"C:/tools/{command}.cmd")

    def fake_run(command, **_kwargs):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="installed\n", stderr="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    result = bootstrap._run("frontend_npm_ci", ["npm", "ci"], cwd=tmp_path)

    assert result.ok is True
    assert captured == [["C:/tools/npm.cmd", "ci"]]


def test_run_preserves_an_explicit_python_executable(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    result = bootstrap._run("python_check", [sys.executable, "--version"], cwd=tmp_path)

    assert result.ok is True
    assert captured == [[sys.executable, "--version"]]


def test_run_fails_closed_when_a_required_command_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _command: None)

    result = bootstrap._run("missing", ["not-installed"])

    assert result.ok is False
    assert result.detail == "not-installed not found on PATH"
