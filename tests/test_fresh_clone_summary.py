"""Contract tests for the ephemeral fresh-clone summary artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_fresh_clone_summary import build_summary


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reports(tmp_path: Path, *, network_skips: list[str] | None = None) -> tuple[Path, Path, Path]:
    backend = _write(
        tmp_path / "backend.json",
        {
            "passed": True,
            "full_suite_detail": {
                "tests_collected": 2500,
                "passed": 2498,
                "failed": 0,
                "coverage_percent": 69.9,
                "coverage_floor": 60,
                "network_or_credential_skips": network_skips or [],
            },
        },
    )
    frontend = _write(
        tmp_path / "frontend.json",
        {"numTotalTests": 267, "numPassedTests": 267, "numFailedTests": 0},
    )
    coverage = _write(
        tmp_path / "coverage.json",
        {
            "total": {
                "statements": {"pct": 39.1},
                "branches": {"pct": 65.42},
                "functions": {"pct": 34.24},
                "lines": {"pct": 39.1},
            }
        },
    )
    return backend, frontend, coverage


def test_summary_preserves_measured_counts_and_floors(tmp_path: Path) -> None:
    summary = build_summary(*_reports(tmp_path))

    assert summary["schema_version"] == "fresh_clone_summary_v1"
    assert summary["backend"]["tests_collected"] == 2500
    assert summary["backend"]["coverage_percent"] == 69.9
    assert summary["backend"]["coverage_floor"] == 60
    assert summary["frontend"]["tests_total"] == 267
    assert summary["frontend"]["coverage"]["branches"] == 65.42
    assert summary["offline_mode"] == "NLCARE_TEST_OFFLINE=true"
    assert summary["commit_sha"]


def test_summary_rejects_network_or_credential_skips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="network/credential skips"):
        build_summary(*_reports(tmp_path, network_skips=["missing provider credential"]))


def test_summary_rejects_zero_test_counts(tmp_path: Path) -> None:
    backend, frontend, coverage = _reports(tmp_path)
    payload = json.loads(frontend.read_text(encoding="utf-8"))
    payload["numTotalTests"] = 0
    frontend.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="test counts must be positive"):
        build_summary(backend, frontend, coverage)
