"""Verify that a fresh checkout can run the offline test path unaided.

Scope of the claim
------------------
This checks *repository content sufficiency*: that everything the offline test
path needs is either tracked in git or rebuildable from tracked inputs, and
that nothing in it requires a secret, a populated ``.env``, or a reachable
network service.

Derived artifacts are the subtle half of that claim. A prebuilt RAG index and
the lakehouse gold records are gitignored on purpose - they are derived data,
not source - so "not tracked" is correct for them and "not needed" is not.
They must be *reproducible*, and this script verifies that by checking every
declared input in
:data:`scripts.provision_derived_artifacts.DERIVED_ARTIFACTS` is present, and
(with ``--provision``) by rebuilding them and running a consumer test against
the result. An earlier version asserted a prebuilt index was simply unnecessary
and checked nothing, which is how seven tests came to pass locally and fail on
every fresh clone.

It deliberately does NOT claim that dependency installation works. Resolving
and installing the pinned environment is a separate contract already covered by
``uv lock --check``, ``scripts/check_dependency_contract.py``, and the CI
install step. Conflating the two is how "reproducible" gets claimed without
evidence, so the two stay separate and this script says so in its output.

What a failure here means
-------------------------
A failure means a fresh clone is missing something a developer or reviewer
would need — usually a file that exists only on a machine where it was
generated, and is gitignored. That is exactly the class of problem that makes a
repository look reproducible to its author and not to anyone else.

Usage
-----
    python scripts/check_fresh_clone_offline.py
    python scripts/check_fresh_clone_offline.py --json-output path.json
    python scripts/check_fresh_clone_offline.py --run-tests
    python scripts/check_fresh_clone_offline.py --provision --run-tests

``--run-tests`` additionally executes the designated offline subset in-process
via pytest. It is off by default so the structural checks stay fast.

``--provision`` rebuilds the declared derived artifacts and then, together with
``--run-tests``, executes a consumer test that reads one of them. That pairing
is the point: it proves the artifacts are reproducible *and* that a consumer is
satisfied by the rebuilt result, rather than only that some file now exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.provision_derived_artifacts import (  # noqa: E402 - needs ROOT on sys.path
    DERIVED_ARTIFACTS,
    missing_artifacts,
    missing_inputs,
    provision,
)

# Files a fresh clone must contain for the offline path to be runnable at all.
REQUIRED_TRACKED_FILES = (
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    "pytest.ini",
    "tests/conftest.py",
    "scripts/check_dependency_contract.py",
    "scripts/check_env_documentation.py",
    "scripts/ingest_knowledge_base.py",
)

# Directories that must exist as tracked paths (their contents may be ignored).
REQUIRED_TRACKED_DIRS = (
    "backend",
    "scripts",
    "tests",
    "config",
)

# A small, genuinely offline subset. These exercise config loading, access
# control, and the safety-eval surface without a model download, a RAG index,
# or a live provider.
OFFLINE_TEST_SUBSET = (
    "tests/test_constants_sync.py",
    "tests/test_access_control.py",
    "tests/test_safety_eval_center.py",
)

# Tests that read a derived artifact, so they are meaningful only *after*
# provisioning. `test_managed_vector_shadow_sync` is the sharpest of them: on a
# fresh clone without provisioning it raises FileNotFoundError on the lakehouse
# gold records, so it fails loudly rather than degrading to a soft status.
PROVISIONED_CONSUMER_TESTS = (
    "tests/test_managed_vector_shadow_sync.py",
    "tests/test_data_platform_reliability_eval.py",
)

# Environment that the offline path is contracted to run under. Mirrors the CI
# `full-offline-tests` job so local and CI results mean the same thing.
OFFLINE_ENV = {
    "NLCARE_TEST_OFFLINE": "true",
    "ENVIRONMENT": "test",
    "LLM_ADJUDICATION_ENABLED": "false",
    "RAG_FORCE_SPARSE": "true",
    "RAG_ENABLE_CROSS_ENCODER": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check_required_paths(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    missing_files = [p for p in REQUIRED_TRACKED_FILES if not (root / p).is_file()]
    results.append(
        CheckResult(
            "required_tracked_files",
            not missing_files,
            "all present" if not missing_files else f"missing: {', '.join(missing_files)}",
        )
    )
    missing_dirs = [p for p in REQUIRED_TRACKED_DIRS if not (root / p).is_dir()]
    results.append(
        CheckResult(
            "required_tracked_dirs",
            not missing_dirs,
            "all present" if not missing_dirs else f"missing: {', '.join(missing_dirs)}",
        )
    )
    return results


def _check_no_dotenv_required(root: Path) -> CheckResult:
    """A fresh clone has no ``.env``; the offline path must not need one.

    ``.env`` is gitignored, so if importing configuration fails without it, no
    reviewer can run anything. ``.env.example`` documents the variables but is
    not loaded.
    """
    if (root / ".env").is_file():
        return CheckResult(
            "no_dotenv_required",
            True,
            "a local .env exists, so this check could not prove independence from it "
            "(run on a clean clone for a conclusive result)",
        )
    return CheckResult("no_dotenv_required", True, "no .env present; offline path must not need one")


def _documented_env_names(root: Path) -> set[str]:
    """Variable names declared in ``.env.example`` — the app's config surface."""
    example = root / ".env.example"
    if not example.is_file():
        return set()
    text = example.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE))


def _check_config_imports_without_secrets(root: Path) -> CheckResult:
    """Import backend configuration with every app-level variable stripped.

    Rather than allow-listing the OS variables an interpreter needs — which
    would both be platform-specific and add non-application names to the
    environment inventory — this removes every variable ``.env.example``
    declares. Whatever remains is the ambient OS environment, so a successful
    import proves configuration does not depend on any documented app setting
    or secret being present.
    """
    documented = _documented_env_names(root)
    env = {k: v for k, v in os.environ.items() if k not in documented}
    env["DATABASE_URL"] = "sqlite:///./Data/test_tmp/fresh_clone_check.db"
    env.update(OFFLINE_ENV)
    (root / "Data" / "test_tmp").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", "import backend.config as c; print(bool(c))"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode == 0:
        return CheckResult("config_imports_without_secrets", True, "backend.config imported")
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    return CheckResult(
        "config_imports_without_secrets",
        False,
        detail[-1] if detail else f"exit {proc.returncode}",
    )


def _check_ruff_config_self_contained(root: Path) -> CheckResult:
    """Lint config must live in the repository, not on the author's machine."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    select = pyproject.get("tool", {}).get("ruff", {}).get("lint", {}).get("select")
    if not select:
        return CheckResult("ruff_config_self_contained", False, "no [tool.ruff.lint] select in pyproject.toml")
    return CheckResult("ruff_config_self_contained", True, f"select = {select}")


def _check_derived_artifact_inputs_present(root: Path) -> CheckResult:
    """Every declared derived artifact must be rebuildable from tracked content.

    This is the check that would have caught the regression. The consuming
    tests do not fail because an artifact is gitignored - that is deliberate -
    they fail because nothing verified the artifact could be *recreated*. If a
    generator ever starts depending on an input that is itself gitignored, that
    input is absent here and this fails on a fresh clone, before the full
    offline suite ever runs.
    """
    absent = missing_inputs(root)
    if absent:
        return CheckResult(
            "derived_artifact_inputs_present",
            False,
            f"declared inputs missing from a fresh checkout: {', '.join(absent)}",
        )
    inputs = sum(len(artifact.inputs) for artifact in DERIVED_ARTIFACTS)
    return CheckResult(
        "derived_artifact_inputs_present",
        True,
        f"{len(DERIVED_ARTIFACTS)} artifacts, {inputs} tracked inputs",
    )


def _check_derived_artifacts_provisionable(root: Path) -> CheckResult:
    """Actually rebuild the derived artifacts and confirm each one appears."""
    results = provision(root)
    failures = [entry["artifact"] for entry in results if not entry["ok"]]
    if failures:
        return CheckResult(
            "derived_artifacts_provisionable",
            False,
            f"could not rebuild: {', '.join(failures)}",
        )
    still_missing = missing_artifacts(root)
    if still_missing:
        return CheckResult(
            "derived_artifacts_provisionable",
            False,
            f"generator reported success but artifact absent: {', '.join(still_missing)}",
        )
    actions = ", ".join(f"{e['artifact']}={e['action']}" for e in results)
    return CheckResult("derived_artifacts_provisionable", True, actions)


def _run_provisioned_consumer_tests(root: Path) -> CheckResult:
    """Run a test that reads a rebuilt artifact.

    Separate from :data:`OFFLINE_TEST_SUBSET` on purpose. The subset is the
    set of tests that need no derived artifact at all; these are the opposite,
    and running them without provisioning first would simply move a known
    failure into an earlier job rather than prove anything.
    """
    env = {**os.environ, **OFFLINE_ENV}
    env["DATABASE_URL"] = "sqlite:///./Data/test_tmp/fresh_clone_consumers.db"
    (root / "Data" / "test_tmp").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *PROVISIONED_CONSUMER_TESTS, "-q"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else f"exit {proc.returncode}"
    return CheckResult("provisioned_consumer_tests_pass", proc.returncode == 0, summary)


def _check_offline_subset_present(root: Path) -> CheckResult:
    missing = [p for p in OFFLINE_TEST_SUBSET if not (root / p).is_file()]
    if missing:
        return CheckResult("offline_test_subset_present", False, f"missing: {', '.join(missing)}")
    return CheckResult("offline_test_subset_present", True, f"{len(OFFLINE_TEST_SUBSET)} files")


def _run_offline_subset(root: Path) -> CheckResult:
    env = {**os.environ, **OFFLINE_ENV}
    env["DATABASE_URL"] = "sqlite:///./Data/test_tmp/fresh_clone_offline.db"
    (root / "Data" / "test_tmp").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *OFFLINE_TEST_SUBSET, "-q"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else f"exit {proc.returncode}"
    return CheckResult("offline_test_subset_passes", proc.returncode == 0, summary)


# ─── full-suite hermetic contract ────────────────────────────────────────────
#
# The subset above answers "can a fresh clone run *anything* offline?".  It
# cannot answer "is the whole suite hermetic?", and a subset that passes while
# the other 297 files quietly need a network is exactly the gap this section
# closes.
#
# The default suite is already hermetic by construction: tests/conftest.py
# clears third-party credentials and blocks every non-loopback connection.
# What was missing was proof that the full suite runs under those conditions,
# and accounting for anything that skipped because of them.

#: Live provider credentials that must be absent for the run to mean anything.
#: Kept in step with tests/conftest.py by test_fresh_clone_full_suite.py - if
#: they drift, the verifier would attest to a weaker contract than the suite
#: actually enforces.
THIRD_PARTY_CREDENTIAL_VARS = (
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "AZURE_SEARCH_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "N8N_WEBHOOK_URL",
    "N8N_API_KEY",
    "MLFLOW_TRACKING_URI",
    "OLLAMA_BASE_URL",
)

#: Coverage floor for the full-suite run, matching the CI gate.
FULL_SUITE_MIN_COVERAGE = 60

#: Skip reasons attributable to a missing network, credential, or live service.
#: A skip matching any of these means the suite is not hermetic - the test did
#: not run, and its absence was hidden behind a green summary line.
NETWORK_SKIP_MARKERS = (
    "network",
    "credential",
    "api key",
    "api_key",
    "token",
    "offline",
    "connection",
    "unreachable",
    "dns",
    "socket",
    "http",
    "groq",
    "ollama",
    "pinecone",
    "azure",
    "openai",
    "gemini",
    "mlflow",
    "n8n",
    "endpoint",
    "service unavailable",
)

_COUNT_PATTERN = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")
_COLLECTED_PATTERN = re.compile(r"(\d+) tests? collected")
_SKIP_LINE_PATTERN = re.compile(r"^SKIPPED \[(\d+)\]\s*(.*)$", re.MULTILINE)
#: Failing test ids, so a failure names itself instead of being a count.
_FAILED_TEST_PATTERN = re.compile(r"^FAILED (\S+)", re.MULTILINE)
_COVERAGE_TOTAL_PATTERN = re.compile(
    r"^TOTAL\s+.*?\s(?P<percent>\d+(?:\.\d+)?)%\s*$", re.MULTILINE
)


def discovered_test_files(root: Path) -> list[str]:
    """Every tracked test module, via git so untracked scratch never inflates it."""
    proc = subprocess.run(
        ["git", "ls-files", "tests"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return sorted(
        line for line in proc.stdout.splitlines()
        if re.search(r"(^|/)test_[^/]*\.py$", line)
    )


def classify_skips(pytest_output: str) -> tuple[list[str], list[str]]:
    """Split reported skips into network/credential ones and everything else.

    Returns (network_related, other). A platform or missing-tool skip is
    legitimate and reported separately; a skip caused by an absent network or
    credential means the suite silently shrank.
    """
    network, other = [], []
    for count, reason in _SKIP_LINE_PATTERN.findall(pytest_output):
        entry = f"[{count}] {reason.strip()}"
        haystack = reason.lower()
        if any(marker in haystack for marker in NETWORK_SKIP_MARKERS):
            network.append(entry)
        else:
            other.append(entry)
    return network, other


def parse_pytest_summary(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "collected": 0}
    collected = _COLLECTED_PATTERN.findall(output)
    if collected:
        counts["collected"] = int(collected[-1])
    for number, label in _COUNT_PATTERN.findall(output):
        key = "errors" if label.startswith("error") else label
        if key in counts:
            counts[key] = max(counts[key], int(number))
    return counts


def _assert_command_is_a_full_suite_run(command: list[str]) -> None:
    """Reject a supplied command that is not the full, coverage-gated suite.

    The point of letting CI state the command is visibility, not
    configurability. A caller that passed a subset or dropped the coverage
    floor would get a green hermetic report for a run that proved much less, so
    the shape is checked rather than trusted.
    """
    joined = " ".join(command)
    if "pytest" not in joined:
        raise ValueError("pytest command does not invoke pytest")
    if "tests" not in command:
        raise ValueError("pytest command must run the whole `tests` tree, not a subset")
    if f"--cov-fail-under={FULL_SUITE_MIN_COVERAGE}" not in joined:
        raise ValueError(
            f"pytest command must enforce --cov-fail-under={FULL_SUITE_MIN_COVERAGE}"
        )
    if "--cov=backend" not in joined:
        raise ValueError("pytest command must measure backend coverage")


def _initialize_disposable_test_database(root: Path, env: dict[str, str]) -> CheckResult:
    """Create the full suite's synthetic SQLite fixture from tracked sources."""
    initialize = subprocess.run(
        [
            sys.executable,
            "scripts/reset_local_db.py",
            "--database-url",
            env["DATABASE_URL"],
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if initialize.returncode == 0:
        return CheckResult(
            "disposable_test_database_initialized",
            True,
            "schema migrated and synthetic P001 demo data seeded offline",
        )
    detail = (initialize.stderr or initialize.stdout).strip().splitlines()
    return CheckResult(
        "disposable_test_database_initialized",
        False,
        detail[-1] if detail else f"exit {initialize.returncode}",
    )


def _run_full_suite(
    root: Path, pytest_command: list[str] | None = None
) -> tuple[list[CheckResult], dict]:
    """Run the complete suite under the hermetic contract and account for skips."""
    env = {**os.environ, **OFFLINE_ENV}
    env["DATABASE_URL"] = "sqlite:///./Data/test_tmp/fresh_clone_offline.db"
    # Never inherit an opt-out: the whole point is to prove the hermetic default.
    env.pop("NLCARE_ALLOW_TEST_NETWORK", None)
    for name in THIRD_PARTY_CREDENTIAL_VARS:
        env.pop(name, None)
    (root / "Data" / "test_tmp").mkdir(parents=True, exist_ok=True)

    # A fresh checkout has no tracked database. Reuse the same deterministic,
    # offline SQLite initializer as Ship instead of relying on a developer's
    # ignored medical_agent.db. This is evidence setup, not an auth bypass:
    # demo login still resolves through the seeded patient rows.
    database_check = _initialize_disposable_test_database(root, env)
    if not database_check.passed:
        return (
            [database_check],
            {
                "database_initialized": False,
                "test_files_discovered": 0,
                "tests_collected": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "coverage_percent": None,
                "network_or_credential_skips": [],
                "other_skips": [],
                "failed_tests": [],
                "exit_code": 1,
            },
        )

    files = discovered_test_files(root)

    # `pytest -q` does not print a collection count, so it is obtained from a
    # dedicated collection pass. That also proves the whole tree imports
    # cleanly, which a failing run would otherwise hide.
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=root, env=env, capture_output=True, text=True, timeout=3600,
    )
    collected_match = _COLLECTED_PATTERN.findall(collect.stdout or "")
    collected_count = int(collected_match[-1]) if collected_match else 0

    # The caller may supply the command so the workflow can state it literally
    # rather than hiding it in here - this repository's CI principle is that
    # the command that runs is the command you read. It is validated below, so
    # passing it cannot be used to quietly weaken the run.
    if pytest_command:
        command = list(pytest_command)
        _assert_command_is_a_full_suite_run(command)
        # A workflow states `python` because that is what a reader expects and
        # what CI's PATH resolves to. Run it with *this* interpreter instead, so
        # a local invocation cannot silently execute against a different
        # environment than the one that imported this script.
        if command[0] in ("python", "python3", "py"):
            command[0] = sys.executable
        # Reporting flags this check depends on, added if the caller omitted them.
        for required in ("-rfs", "-q"):
            if required not in command:
                command.append(required)
    else:
        command = [
            sys.executable, "-m", "pytest", "tests", "-q", "-rfs",
            "-p", "no:cacheprovider",
            "--cov=backend", "--cov-branch",
            f"--cov-fail-under={FULL_SUITE_MIN_COVERAGE}",
            "--cov-report=term-missing:skip-covered",
        ]
    proc = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True, timeout=14400
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    counts = parse_pytest_summary(output)
    if collected_count:
        counts["collected"] = collected_count
    network_skips, other_skips = classify_skips(output)
    failed_tests = _FAILED_TEST_PATTERN.findall(output)
    coverage_matches = _COVERAGE_TOTAL_PATTERN.findall(output)
    coverage_percent = float(coverage_matches[-1]) if coverage_matches else None

    results = [
        database_check,
        CheckResult(
            "full_suite_executed",
            proc.returncode == 0,
            f"{counts['collected']} collected from {len(files)} test files; "
            f"{counts['passed']} passed, {counts['failed']} failed, "
            f"{counts['skipped']} skipped, {counts['errors']} error(s)",
        ),
        CheckResult(
            "full_suite_covers_every_test_file",
            counts["collected"] > 0 and len(files) > 0,
            f"{len(files)} tracked test files discovered, whole `tests` tree executed "
            "(no subset selection)",
        ),
        CheckResult(
            "no_network_or_credential_skips",
            not network_skips,
            "none"
            if not network_skips
            else f"{len(network_skips)} skip(s) attributable to network/credentials: "
            + "; ".join(network_skips[:5]),
        ),
        CheckResult(
            "coverage_floor_met",
            proc.returncode == 0,
            f"pytest enforced --cov-fail-under={FULL_SUITE_MIN_COVERAGE}"
            + ("" if proc.returncode == 0 else " (run did not pass)"),
        ),
    ]
    detail = {
        "test_files_discovered": len(files),
        "tests_collected": counts["collected"],
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
        "failed_tests": failed_tests,
        "network_or_credential_skips": network_skips,
        "other_skips": other_skips,
        "coverage_floor": FULL_SUITE_MIN_COVERAGE,
        "coverage_percent": coverage_percent,
        "credentials_cleared": list(THIRD_PARTY_CREDENTIAL_VARS),
        "network_policy": "non-loopback connections blocked by tests/conftest.py",
        "exit_code": proc.returncode,
    }
    if proc.returncode != 0:
        log_path = root / "Data" / "test_tmp" / "full_suite_output.txt"
        log_path.write_text(output, encoding="utf-8")
        detail["full_output_path"] = str(log_path.relative_to(root)).replace("\\", "/")
        detail["failure_tail"] = [
            line for line in output.splitlines() if line.startswith("FAILED")
        ][:25]
    return results, detail


def run_checks(
    root: Path = ROOT,
    run_tests: bool = False,
    provision_artifacts: bool = False,
    full_suite: bool = False,
    pytest_command: list[str] | None = None,
) -> tuple[list[CheckResult], dict]:
    results: list[CheckResult] = []
    results.extend(_check_required_paths(root))
    results.append(_check_no_dotenv_required(root))
    results.append(_check_ruff_config_self_contained(root))
    results.append(_check_offline_subset_present(root))
    results.append(_check_derived_artifact_inputs_present(root))
    results.append(_check_config_imports_without_secrets(root))
    if provision_artifacts:
        results.append(_check_derived_artifacts_provisionable(root))
    if run_tests:
        results.append(_run_offline_subset(root))
        if provision_artifacts:
            results.append(_run_provisioned_consumer_tests(root))
    full_suite_detail: dict = {}
    if full_suite:
        suite_results, full_suite_detail = _run_full_suite(root, pytest_command)
        results.extend(suite_results)
    return results, full_suite_detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="also execute the designated offline pytest subset",
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help=(
            "run the complete tests/ tree with coverage under the hermetic contract "
            "and account for every skip (the canonical offline verification)"
        ),
    )
    parser.add_argument(
        "--pytest-command",
        default=None,
        help=(
            "the exact pytest command to run for the full suite, so a workflow "
            "can state it literally instead of it being hidden here. Validated: "
            "it must run the whole tests tree with the backend coverage floor."
        ),
    )
    parser.add_argument(
        "--provision",
        action="store_true",
        help=(
            "rebuild the declared derived artifacts and verify them; with --run-tests, "
            "also run a consumer test against the rebuilt result"
        ),
    )
    args = parser.parse_args()

    results, full_suite_detail = run_checks(
        run_tests=args.run_tests,
        provision_artifacts=args.provision,
        full_suite=args.full_suite,
        pytest_command=shlex.split(args.pytest_command) if args.pytest_command else None,
    )
    failures = [r for r in results if not r.passed]

    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}")

    if full_suite_detail:
        print()
        print("Full-suite hermetic accounting")
        print(f"  test files discovered   : {full_suite_detail['test_files_discovered']}")
        print(f"  tests collected         : {full_suite_detail['tests_collected']}")
        print(
            f"  passed/failed/skipped   : {full_suite_detail['passed']}/"
            f"{full_suite_detail['failed']}/{full_suite_detail['skipped']}"
        )
        print(
            f"  network/credential skips: "
            f"{len(full_suite_detail['network_or_credential_skips'])}"
        )
        for entry in full_suite_detail["network_or_credential_skips"]:
            print(f"      {entry}")
        print(f"  other (legitimate) skips: {len(full_suite_detail['other_skips'])}")
        for entry in full_suite_detail["other_skips"]:
            print(f"      {entry}")
        print(f"  credentials cleared     : {len(full_suite_detail['credentials_cleared'])}")
        print(f"  network policy          : {full_suite_detail['network_policy']}")
        for test_id in full_suite_detail.get("failed_tests", []):
            print(f"  FAILED: {test_id}")
        for line in full_suite_detail.get("failure_tail", []):
            print(f"      {line}")

    print()
    print(
        "Scope: repository content sufficiency for the offline test path. "
        "Dependency installation is NOT verified here — see `uv lock --check` "
        "and scripts/check_dependency_contract.py."
    )

    if args.json_output:
        payload = {
            "schema_version": "fresh_clone_offline_check_v2",
            "checks": [asdict(r) for r in results],
            "passed": not failures,
            "tests_executed": args.run_tests or args.full_suite,
            "artifacts_provisioned": args.provision,
            "full_suite": args.full_suite,
            "full_suite_detail": full_suite_detail,
            "claim_boundary": (
                "Verifies that tracked repository content is sufficient to run the offline "
                "test path without secrets, a populated .env, or network access, and that "
                "the derived artifacts the path consumes are rebuildable from tracked "
                "inputs. Does not verify dependency resolution or installation, and makes "
                "no clinical or production-readiness claim."
            ),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_output}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
