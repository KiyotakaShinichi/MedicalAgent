"""Contract tests for the fresh-clone offline reproducibility checker.

The checker exists to catch a specific failure mode: a repository that runs
only on the machine where its untracked files were generated. These tests
verify the checker actually fails when that condition is simulated, because a
verifier that always passes is worse than none — it manufactures confidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_fresh_clone_offline import (
    OFFLINE_ENV,
    OFFLINE_TEST_SUBSET,
    REQUIRED_TRACKED_DIRS,
    REQUIRED_TRACKED_FILES,
    CheckResult,
    _check_offline_subset_present,
    _check_required_paths,
    _check_ruff_config_self_contained,
    run_checks,
)

ROOT = Path(__file__).resolve().parents[1]


def test_repository_satisfies_its_own_structural_contract() -> None:
    """The live repository must pass every structural check."""
    results = _check_required_paths(ROOT)
    failures = [r for r in results if not r.passed]
    assert not failures, [r.detail for r in failures]


def test_required_paths_are_actually_tracked_files() -> None:
    for relative in REQUIRED_TRACKED_FILES:
        assert (ROOT / relative).is_file(), f"{relative} is declared required but absent"
    for relative in REQUIRED_TRACKED_DIRS:
        assert (ROOT / relative).is_dir(), f"{relative} is declared required but absent"


def test_missing_required_file_is_detected(tmp_path: Path) -> None:
    """A clone missing a required file must fail, not pass quietly."""
    (tmp_path / "backend").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "config").mkdir()
    # No pyproject.toml, uv.lock, .env.example, ...

    results = _check_required_paths(tmp_path)
    files_check = next(r for r in results if r.name == "required_tracked_files")
    assert files_check.passed is False
    assert "pyproject.toml" in files_check.detail


def test_missing_required_directory_is_detected(tmp_path: Path) -> None:
    for relative in REQUIRED_TRACKED_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    # `config/` deliberately absent.
    results = _check_required_paths(tmp_path)
    dirs_check = next(r for r in results if r.name == "required_tracked_dirs")
    assert dirs_check.passed is False
    assert "config" in dirs_check.detail


def test_offline_subset_is_declared_and_present() -> None:
    result = _check_offline_subset_present(ROOT)
    assert result.passed, result.detail
    assert len(OFFLINE_TEST_SUBSET) >= 1


def test_offline_subset_missing_is_detected(tmp_path: Path) -> None:
    result = _check_offline_subset_present(tmp_path)
    assert result.passed is False
    assert "missing" in result.detail


def test_ruff_config_is_self_contained_and_non_trivial() -> None:
    """Lint rules must live in the repo, and must be more than a syntax check."""
    result = _check_ruff_config_self_contained(ROOT)
    assert result.passed, result.detail
    # Guards against silently narrowing the rule set back to syntax-only.
    assert "'F'" in result.detail or '"F"' in result.detail


def test_ruff_config_absence_is_detected(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    result = _check_ruff_config_self_contained(tmp_path)
    assert result.passed is False


def test_offline_env_pins_providers_and_network_off() -> None:
    """The declared offline environment must actually disable live services."""
    assert OFFLINE_ENV["NLCARE_TEST_OFFLINE"] == "true"
    assert OFFLINE_ENV["LLM_ADJUDICATION_ENABLED"] == "false"
    assert OFFLINE_ENV["HF_HUB_OFFLINE"] == "1"
    assert OFFLINE_ENV["TRANSFORMERS_OFFLINE"] == "1"
    assert OFFLINE_ENV["RAG_FORCE_SPARSE"] == "true"


def test_run_checks_reports_structured_results() -> None:
    results, _full_suite = run_checks(ROOT, run_tests=False)
    assert results, "run_checks returned nothing"
    assert all(isinstance(r, CheckResult) for r in results)
    names = {r.name for r in results}
    assert {"required_tracked_files", "ruff_config_self_contained"} <= names
    # The expensive subset must not run unless explicitly requested.
    assert "offline_test_subset_passes" not in names


def test_json_payload_shape_is_serialisable() -> None:
    """The artifact this writes must be machine-readable for a reviewer."""
    results, _full_suite = run_checks(ROOT, run_tests=False)
    payload = {
        "schema_version": "fresh_clone_offline_check_v1",
        "checks": [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results],
        "passed": all(r.passed for r in results),
    }
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["schema_version"] == "fresh_clone_offline_check_v1"
    assert isinstance(round_tripped["checks"], list)
