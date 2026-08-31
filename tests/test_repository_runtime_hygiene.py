"""Repository tests must not rely on tracked interpreter cache files."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_bytecode_and_cache_files_are_not_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "*.pyc", "*/__pycache__/*", ".coverage", ".coverage.*"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], f"interpreter cache files must be ignored, not tracked: {tracked}"


def test_generated_runtime_directories_are_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=ROOT,
        input=(
            "backend/__pycache__/probe.pyc\0"
            ".coverage\0"
            ".coverage.worker\0"
            "Data/test_tmp/runtime_probe.json\0"
            "Data/runtime/runtime_probe.json\0"
        ),
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0
    assert len([path for path in result.stdout.split("\0") if path]) == 5
