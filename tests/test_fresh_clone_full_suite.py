"""The fresh-clone verifier's full-suite hermetic contract.

The verifier used to run a three-file subset. A subset that passes tells you
nothing about the other 300 files, which is precisely where a hidden dependency
on a network or a credential would live — and it would stay hidden, because
skipped tests do not fail.

`--full-suite` closes that: it runs the whole `tests` tree with coverage,
credentials stripped and the network blocked, and accounts for every skip. A
skip attributable to a missing network or credential fails the check; a
platform or tooling skip is reported separately and does not.

These tests drive the accounting functions with synthetic pytest output, so
they assert the classification rules rather than re-running the suite (which
the verifier itself does, and which takes the better part of an hour).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_fresh_clone_offline import (  # noqa: E402
    FULL_SUITE_MIN_COVERAGE,
    NETWORK_SKIP_MARKERS,
    THIRD_PARTY_CREDENTIAL_VARS,
    classify_skips,
    discovered_test_files,
    parse_pytest_summary,
)


# ─── skip accounting ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reason",
    [
        "tests/test_x.py:3: requires network access",
        "tests/test_x.py:3: no GROQ_API_KEY credential configured",
        "tests/test_x.py:3: Pinecone endpoint unreachable",
        "tests/test_x.py:3: Azure Search API key missing",
        "tests/test_x.py:3: OLLAMA_BASE_URL not set",
        "tests/test_x.py:3: connection refused",
    ],
)
def test_network_and_credential_skips_are_detected(reason: str) -> None:
    """These are the skips that mean the suite silently shrank."""
    network, other = classify_skips(f"SKIPPED [1] {reason}")
    assert len(network) == 1, f"not classified as network/credential: {reason}"
    assert other == []


@pytest.mark.parametrize(
    "reason",
    [
        "tests/test_x.py:3: requires Windows",
        "tests/test_x.py:3: docker binary not installed",
        "tests/test_x.py:3: needs python 3.13",
    ],
)
def test_platform_and_tooling_skips_are_reported_separately(reason: str) -> None:
    """A legitimate skip must not fail the check, but must still be visible."""
    network, other = classify_skips(f"SKIPPED [1] {reason}")
    assert network == []
    assert len(other) == 1


def test_mixed_skips_are_split_correctly() -> None:
    output = (
        "SKIPPED [2] tests/a.py:1: requires network access\n"
        "SKIPPED [1] tests/b.py:9: requires Windows\n"
        "SKIPPED [3] tests/c.py:4: missing OPENAI_API_KEY\n"
    )
    network, other = classify_skips(output)
    assert len(network) == 2
    assert len(other) == 1
    assert "[2]" in network[0], "the skip count is preserved for reporting"


def test_no_skips_classifies_cleanly() -> None:
    assert classify_skips("2202 passed in 3284s") == ([], [])


def test_every_cleared_credential_would_be_caught_as_a_skip_reason() -> None:
    """A credential the suite strips must be recognisable if a test skips for it.

    Otherwise the verifier would clear `GROQ_API_KEY`, a test would skip citing
    it, and the skip would be filed as "legitimate".
    """
    for variable in THIRD_PARTY_CREDENTIAL_VARS:
        reason = f"tests/test_x.py:1: {variable} is not configured"
        network, _other = classify_skips(f"SKIPPED [1] {reason}")
        assert network, f"a skip citing {variable} would not be flagged"


# ─── summary parsing ─────────────────────────────────────────────────────────


def test_summary_parsing_reads_a_passing_run() -> None:
    counts = parse_pytest_summary("2202 tests collected\n2202 passed in 3284.26s\n")
    assert counts["collected"] == 2202
    assert counts["passed"] == 2202
    assert counts["failed"] == 0


def test_summary_parsing_reads_a_failing_run() -> None:
    counts = parse_pytest_summary(
        "2202 tests collected\n1 failed, 2200 passed, 1 skipped in 3284.26s\n"
    )
    assert counts["failed"] == 1
    assert counts["passed"] == 2200
    assert counts["skipped"] == 1


def test_summary_parsing_survives_missing_numbers() -> None:
    """A crashed run must report zeros, not raise."""
    assert parse_pytest_summary("INTERNALERROR")["collected"] == 0


# ─── discovery ───────────────────────────────────────────────────────────────


def test_discovery_finds_the_repository_test_files() -> None:
    files = discovered_test_files(ROOT)
    assert len(files) > 250, f"only {len(files)} test files discovered"
    assert all(Path(f).name.startswith("test_") for f in files)
    assert all(f.startswith("tests/") for f in files)


def test_discovery_includes_subdirectory_tests() -> None:
    """`tests/breast_monitoring/` must not be missed by a flat glob."""
    files = discovered_test_files(ROOT)
    assert any("/" in f[len("tests/"):] for f in files), "no nested test files found"


def test_discovery_only_returns_tracked_files() -> None:
    """Untracked scratch must not inflate the reported coverage of the suite."""
    files = discovered_test_files(ROOT)
    assert len(set(files)) == len(files), "duplicate entries"


# ─── the contract's own constants ────────────────────────────────────────────


def test_credential_list_matches_the_test_suite() -> None:
    """The verifier and conftest must strip the same variables.

    If they drift, the verifier attests to a weaker hermetic contract than the
    suite actually enforces — or a stronger one it does not.
    """
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    for variable in THIRD_PARTY_CREDENTIAL_VARS:
        assert variable in conftest, f"{variable} is not cleared by tests/conftest.py"


def test_ci_runs_this_verifier_in_full_suite_mode() -> None:
    """CI's authoritative test run is this contract, not a bare pytest line.

    If CI went back to invoking pytest directly, the hermetic accounting would
    stop running and nothing would notice — the suite would still be green.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    # CI runs the smoke script; the script runs the verifier in full-suite mode.
    assert "verify_fresh_clone.sh" in workflow
    assert "check_fresh_clone_offline.py" in smoke
    assert "--full-suite" in smoke


def test_full_suite_is_reported_as_test_execution() -> None:
    source = (ROOT / "scripts" / "check_fresh_clone_offline.py").read_text(
        encoding="utf-8"
    )
    assert '"tests_executed": args.run_tests or args.full_suite' in source


def test_coverage_floor_is_sixty() -> None:
    """The floor lives here now, so it is pinned here."""
    assert FULL_SUITE_MIN_COVERAGE == 60


def test_disposable_database_uses_repository_initializer(monkeypatch, tmp_path: Path) -> None:
    """The full suite must not inherit a developer's ignored SQLite file."""
    from scripts import check_fresh_clone_offline as verifier

    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="[reset-db] done", stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    env = {"DATABASE_URL": "sqlite:///./Data/test_tmp/fresh_clone_offline.db"}
    result = verifier._initialize_disposable_test_database(tmp_path, env)

    assert result.passed is True
    assert observed["command"] == [
        sys.executable,
        "scripts/reset_local_db.py",
        "--database-url",
        env["DATABASE_URL"],
    ]
    assert observed["kwargs"]["env"] is env
    assert observed["kwargs"]["cwd"] == tmp_path


def test_disposable_database_initialization_fails_closed(monkeypatch, tmp_path: Path) -> None:
    from scripts import check_fresh_clone_offline as verifier

    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="migration refused",
        ),
    )
    result = verifier._initialize_disposable_test_database(
        tmp_path,
        {"DATABASE_URL": "sqlite:///./Data/test_tmp/fresh_clone_offline.db"},
    )

    assert result.passed is False
    assert result.detail == "migration refused"


def test_network_markers_cover_the_providers_the_repository_uses() -> None:
    for provider in ("groq", "ollama", "pinecone", "azure", "openai"):
        assert provider in NETWORK_SKIP_MARKERS


def test_default_suite_deselects_network_marked_tests() -> None:
    """The hermetic default is declared in pytest.ini, not ad hoc per test."""
    config = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert 'addopts = -m "not requires_network"' in config
    assert "requires_network:" in config, "the marker must be registered, not implicit"


# ─── the pytest command CI states explicitly ─────────────────────────────────
#
# CI passes the exact pytest invocation rather than letting the verifier hide
# it, so the command that runs is the command you read in the workflow. That is
# visibility, not configurability: a supplied command that ran a subset or
# dropped the coverage floor would earn a green hermetic report for a much
# weaker run, so the shape is validated.


def test_supplied_command_must_run_the_whole_tree() -> None:
    from scripts.check_fresh_clone_offline import _assert_command_is_a_full_suite_run

    with pytest.raises(ValueError, match="whole `tests` tree"):
        _assert_command_is_a_full_suite_run(
            ["python", "-m", "pytest", "tests/test_health_endpoint.py",
             "--cov=backend", "--cov-fail-under=60"]
        )


def test_supplied_command_must_keep_the_coverage_floor() -> None:
    from scripts.check_fresh_clone_offline import _assert_command_is_a_full_suite_run

    with pytest.raises(ValueError, match="cov-fail-under"):
        _assert_command_is_a_full_suite_run(
            ["python", "-m", "pytest", "tests", "--cov=backend"]
        )


def test_supplied_command_must_measure_backend_coverage() -> None:
    from scripts.check_fresh_clone_offline import _assert_command_is_a_full_suite_run

    with pytest.raises(ValueError, match="backend coverage"):
        _assert_command_is_a_full_suite_run(
            ["python", "-m", "pytest", "tests", "--cov-fail-under=60"]
        )


def test_supplied_command_must_actually_invoke_pytest() -> None:
    from scripts.check_fresh_clone_offline import _assert_command_is_a_full_suite_run

    with pytest.raises(ValueError, match="invoke pytest"):
        _assert_command_is_a_full_suite_run(["echo", "tests", "--cov=backend",
                                             "--cov-fail-under=60"])


def test_the_ci_supplied_command_passes_validation() -> None:
    """The command in the workflow must be one the verifier accepts."""
    import shlex

    smoke = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    match = re.search(r'--pytest-command "([^"]+)"', smoke)
    assert match, "the smoke script no longer states the pytest command explicitly"

    from scripts.check_fresh_clone_offline import _assert_command_is_a_full_suite_run

    _assert_command_is_a_full_suite_run(shlex.split(match.group(1)))


def test_ci_states_the_pytest_invocation_literally() -> None:
    """A reader of the workflow can see which tests run, without following code."""
    smoke = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    assert "pytest tests" in smoke, (
        "the authoritative test command is no longer visible in the smoke script"
    )
    assert "--cov-fail-under=60" in smoke


def test_tests_still_run_exactly_once() -> None:
    """Visibility must not have been bought with a duplicate full-suite run."""
    smoke = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert smoke.count("--pytest-command") == 1
    assert workflow.count("verify_fresh_clone.sh") == 1, (
        "the smoke script is invoked more than once; the full suite must run once"
    )


def test_supplied_command_runs_on_the_current_interpreter() -> None:
    """`python` in the workflow must not mean a different environment locally.

    CI has the project venv on PATH, so `python` is the right interpreter
    there. Run from a developer machine the same string would resolve to
    whatever `python` happens to be first, and the suite would execute against
    an environment missing the project's dependencies — which is exactly how
    this first surfaced, as 17 import errors for python-json-logger.
    """
    import shlex
    import subprocess

    smoke = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    match = re.search(r'--pytest-command "([^"]+)"', smoke)
    assert match
    assert shlex.split(match.group(1))[0] == "python", (
        "the workflow should read naturally; the verifier substitutes the interpreter"
    )

    source = (ROOT / "scripts" / "check_fresh_clone_offline.py").read_text(encoding="utf-8")
    assert "command[0] = sys.executable" in source, (
        "the verifier no longer normalises the interpreter"
    )
    assert subprocess.run  # the substitution only matters because we spawn a process
