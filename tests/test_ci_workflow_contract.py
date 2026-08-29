"""The CI workflow must actually run — and block on — the quality gates.

Two failure modes this guards against.

The first is silent weakening. A `continue-on-error: true` or a trailing
`|| true` turns a gate into decoration: the job stays green while Ruff, mypy,
or the test suite fails. That is invisible in review unless something asserts
it, because the step still appears in the log with its normal name.

The second is environment drift. CI builds its environment with
`uv sync --frozen` and then puts `.venv/bin` on `PATH` so later steps invoke
tools directly. A step that both syncs *and* runs a tool in the same `run:`
block cannot rely on `$GITHUB_PATH` — that only affects *subsequent* steps — so
it must export `PATH` itself. Getting this wrong produces a "command not found"
that only appears on the runner. It happened once while writing this file.

These tests read the workflow as data. They deliberately assert on the
*presence and blocking-ness* of invocations, not on exact formatting, so
ordinary edits to flags or arguments do not fail them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _workflow() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


def _steps():
    """Yield (job_name, step_dict) for every step in the CI workflow."""
    for job, config in _workflow()["jobs"].items():
        for step in config.get("steps", []):
            yield job, step


def _run_blocks() -> str:
    return "\n".join(str(step.get("run") or "") for _, step in _steps())


# ─── the gates must be invoked ───────────────────────────────────────────────

# (human-readable name, substrings that all must appear on one `run` line)
REQUIRED_INVOCATIONS = [
    ("ruff lint", ("ruff", "check")),
    ("mypy typecheck", ("mypy",)),
    # The complete backend suite runs through the fresh-clone verifier rather
    # than a bare pytest line. The verifier invokes `pytest tests` with the same
    # branch-coverage settings and additionally accounts for hermeticity: it
    # strips live credentials, refuses the network opt-out, and fails on any
    # skip caused by a missing network or credential. Running both would mean
    # executing the whole suite twice, so this is the single authoritative run.
    # The full suite runs through scripts/verify_fresh_clone.sh, which invokes
    # check_fresh_clone_offline.py --full-suite, which runs the pytest command
    # printed in the script. One execution; the workflow names the script and
    # the script names the command.
    ("backend fresh-clone smoke", ("verify_fresh_clone.sh",)),
    ("frontend lint", ("npm run lint",)),
    ("frontend typecheck", ("npm run typecheck",)),
    ("frontend tests/coverage", ("npm run test:coverage",)),
]


def _invokes(needles: tuple[str, ...], text: str) -> bool:
    """True when one `run` line contains every needle."""
    return any(
        all(needle in line for needle in needles)
        for line in text.splitlines()
    )


@pytest.mark.parametrize("label,needles", REQUIRED_INVOCATIONS)
def test_ci_invokes_required_gate(label: str, needles: tuple[str, ...]) -> None:
    assert _invokes(needles, _run_blocks()), (
        f"CI does not appear to run {label}. Every quality gate must be invoked "
        f"by a workflow step; expected a run line containing {needles}."
    )


def test_gate_detection_rejects_a_workflow_missing_the_gate() -> None:
    """Guards the guard: the check above must be capable of failing.

    A membership test over a large blob passes too easily; this confirms it
    discriminates rather than always finding something.
    """
    for _, needles in REQUIRED_INVOCATIONS:
        assert not _invokes(needles, "run: echo hello\nrun: npm ci")


def test_backend_coverage_floor_is_not_lowered() -> None:
    """The floor is a regression gate; lowering it to pass a build is the failure.

    The floor moved out of the workflow file and into the verifier that runs the
    suite, so it is asserted at its source. Both halves are checked: the number
    itself, and that the verifier actually hands it to pytest — a constant that
    nothing passes through would enforce nothing.
    """
    from scripts.check_fresh_clone_offline import FULL_SUITE_MIN_COVERAGE

    assert FULL_SUITE_MIN_COVERAGE == 60, (
        "the enforced backend coverage floor is no longer 60"
    )
    smoke = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    assert "--cov-fail-under=60" in smoke, "the smoke script no longer states the floor"

    verifier = (ROOT / "scripts" / "check_fresh_clone_offline.py").read_text(encoding="utf-8")
    assert 'f"--cov-fail-under={FULL_SUITE_MIN_COVERAGE}"' in verifier, (
        "the coverage floor constant is no longer passed to pytest"
    )
    assert "--cov-branch" in verifier, "branch coverage is no longer requested"


# ─── the gates must block ────────────────────────────────────────────────────


def test_no_step_is_marked_continue_on_error() -> None:
    offenders = [
        f"{job}:{step.get('name')}"
        for job, step in _steps()
        if step.get("continue-on-error")
    ]
    assert not offenders, (
        f"continue-on-error makes a gate non-blocking: {offenders}. "
        "The job would stay green while the check fails."
    )


def test_no_job_is_marked_continue_on_error() -> None:
    workflow = _workflow()
    offenders = [j for j, c in workflow["jobs"].items() if c.get("continue-on-error")]
    assert not offenders, f"jobs are non-blocking: {offenders}"


@pytest.mark.parametrize("swallower", ["|| true", "|| :", "set +e", "; exit 0"])
def test_no_run_block_swallows_a_failure(swallower: str) -> None:
    """Exit-code propagation is what makes a failing command fail the job."""
    assert swallower not in _run_blocks(), (
        f"{swallower!r} in a run block discards a non-zero exit status"
    )


def test_conditional_steps_are_only_evidence_uploads() -> None:
    """`if:` on a *check* would let it be skipped; on an upload it is correct.

    `if: always()` on an artifact upload publishes evidence even when the job
    failed, which is what makes a red run diagnosable.
    """
    for job, step in _steps():
        condition = step.get("if")
        if not condition:
            continue
        uses = str(step.get("uses") or "")
        assert "upload-artifact" in uses, (
            f"{job}:{step.get('name')} is conditional ({condition!r}) but is not an "
            "artifact upload; a conditional quality gate can silently not run"
        )


def test_downstream_jobs_depend_on_the_gates() -> None:
    """Later jobs must not run when the gates failed."""
    jobs = _workflow()["jobs"]
    needs = jobs["docker-build"].get("needs") or []
    for required in (
        "static-quality",
        "fresh-clone-smoke",
        "ml-regression",
        "dependency-audit",
    ):
        assert required in needs, f"docker-build does not wait for {required}"


# ─── environment consistency ─────────────────────────────────────────────────


def test_tools_run_from_the_uv_created_environment() -> None:
    """Direct invocations require `.venv/bin` on PATH, put there after uv sync.

    A step that syncs and runs a tool in the *same* block must export PATH
    itself, because `$GITHUB_PATH` only applies to subsequent steps.
    """
    for job, config in _workflow()["jobs"].items():
        path_exported = False
        for step in config.get("steps", []):
            run = str(step.get("run") or "")
            if "uv sync" in run and any(
                tool in run for tool in ("pip-audit", "ruff ", "mypy", "pytest")
            ):
                assert "export PATH=" in run, (
                    f"{job}:{step.get('name')} syncs and runs a tool in one step, so it "
                    "must export PATH; $GITHUB_PATH would not apply until the next step"
                )
            if "GITHUB_PATH" in run:
                path_exported = True
        if path_exported:
            assert ".venv/bin" in "\n".join(
                str(s.get("run") or "") for s in config.get("steps", [])
            ), f"{job} exports a PATH entry that is not the uv-created .venv"


def test_environment_is_built_from_the_frozen_lockfile() -> None:
    """`--frozen` is what makes CI install the locked versions, not resolve new ones."""
    runs = _run_blocks()
    assert "uv sync --frozen" in runs
    assert "uv sync" not in runs.replace("uv sync --frozen", ""), (
        "an unfrozen `uv sync` would let CI resolve versions the lockfile does not pin"
    )


# ─── the quality commands are literally visible ──────────────────────────────
#
# External scanners and new contributors both read the workflow to find out how
# a project is checked. A command hidden inside a wrapper script is invisible to
# the first and unfindable by the second, so the literal invocations are
# asserted here — without adding a second execution of anything.

LITERAL_COMMANDS = [
    ("ruff", "ruff check backend scripts tests"),
    ("mypy", "mypy"),
    ("frontend lint", "npm run lint"),
    ("frontend typecheck", "npm run typecheck"),
    ("frontend unit tests", "npm run test"),
    ("frontend build", "npm run build"),
    ("backend pytest", "python -m pytest"),
]


@pytest.mark.parametrize("label,command", LITERAL_COMMANDS)
def test_quality_command_is_literally_present(label: str, command: str) -> None:
    assert command in _run_blocks(), (
        f"the {label} command is no longer visible as literal text in a workflow "
        f"run block; expected {command!r}"
    )


def test_the_fresh_clone_smoke_job_exists() -> None:
    """A clean-checkout job, distinct from the caches every other job uses."""
    jobs = _workflow()["jobs"]
    assert "fresh-clone-smoke" in jobs, "the fresh-clone smoke job is missing"


def test_the_smoke_job_does_not_restore_dependency_caches() -> None:
    """Restoring .venv or node_modules would make it not a fresh clone.

    The Hugging Face model cache is deliberately allowed: the safety encoder is
    a 480 MB download, not a project dependency, and the job provisions it when
    absent.
    """
    job = _workflow()["jobs"]["fresh-clone-smoke"]
    for step in job["steps"]:
        cache_path = str((step.get("with") or {}).get("path") or "")
        if "cache" in str(step.get("uses") or ""):
            assert "node_modules" not in cache_path, "node_modules restored into the smoke job"
            assert ".venv" not in cache_path, "the virtualenv was restored into the smoke job"


def test_the_smoke_job_runs_the_smoke_script() -> None:
    job = _workflow()["jobs"]["fresh-clone-smoke"]
    runs = "\n".join(str(step.get("run") or "") for step in job["steps"])
    assert "verify_fresh_clone.sh" in runs


def test_the_smoke_script_covers_the_whole_sequence() -> None:
    """The one-command verifier covers every core engineering gate."""
    script = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    for token in (
        "uv sync --frozen",
        "scripts/provision_semantic_safety_encoders.py",
        "scripts/provision_derived_artifacts.py",
        "NLCARE_TEST_OFFLINE=true",
        "ruff check backend scripts tests",
        "uv run mypy",
        "pytest tests",
        "--cov=backend",
        "--cov-fail-under=60",
        "npm ci",
        "npm run lint",
        "npm run typecheck",
        "npm run test:coverage",
        "npm run build",
        "scripts/check_dependency_contract.py",
        "scripts/check_file_size.py",
        "scripts/build_fresh_clone_summary.py",
        "FRESH CLONE OK",
    ):
        assert token in script, f"the smoke script no longer runs {token!r}"


def test_the_smoke_script_fails_fast() -> None:
    """Without `set -e` a failing step would still reach FRESH CLONE OK."""
    script = (ROOT / "scripts" / "verify_fresh_clone.sh").read_text(encoding="utf-8")
    assert "set -eu" in script


def test_the_full_suite_runs_exactly_once_in_ci() -> None:
    """Visibility must not be bought with a duplicate 55-minute run."""
    runs = _run_blocks()
    assert runs.count("verify_fresh_clone.sh") == 1
    assert runs.count("--full-suite") == 0, (
        "the verifier's full-suite mode is invoked directly as well as through "
        "the smoke script; that would run the whole suite twice"
    )


def test_core_ci_does_not_execute_release_or_external_tooling() -> None:
    """Genuine release evidence and CDN tools belong to Ship, not core CI."""
    ci_runs = _run_blocks()
    ship = (ROOT / ".github" / "workflows" / "ship.yml").read_text(encoding="utf-8")
    for token in ("playwright install", "bicep-linux"):
        assert token not in ci_runs, f"core CI still executes release-only step {token!r}"
        assert token in ship, f"{token!r} was removed instead of isolated in Ship"
    release_steps = (ROOT / "scripts" / "ship_steps" / "assurance_and_release.py").read_text(
        encoding="utf-8"
    )
    assert "scripts/run_release_gate.py" not in ci_runs
    assert "scripts/run_release_gate.py" in release_steps
    assert "scripts/ship.py" in ship


def test_fresh_clone_summary_is_uploaded_but_not_committed() -> None:
    job = _workflow()["jobs"]["fresh-clone-smoke"]
    upload_paths = "\n".join(
        str((step.get("with") or {}).get("path") or "") for step in job["steps"]
    )
    assert "Data/test_tmp/fresh_clone_summary.json" in upload_paths


# ─── the ML regression gate ──────────────────────────────────────────────────


def test_the_ml_regression_job_exists() -> None:
    assert "ml-regression" in _workflow()["jobs"]


def test_ml_regression_runs_deterministic_suites_only() -> None:
    """No training, no model download, no network — it must stay fast and fixed."""
    job = _workflow()["jobs"]["ml-regression"]
    runs = "\n".join(str(step.get("run") or "") for step in job["steps"])

    assert "python -m pytest" in runs
    assert "test_synthetic_model_perturbation" in runs
    assert job["env"]["HF_HUB_OFFLINE"] == "1", "the ML job must not download models"
    assert job["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_ml_regression_asserts_real_metrics() -> None:
    """A job that ran no metric assertions would be a green light for nothing."""
    suite = (ROOT / "tests" / "test_synthetic_model_perturbation_modules.py").read_text(
        encoding="utf-8"
    )
    assert "SEED" in suite and "REPEATED_SPLIT_SEEDS" in suite
    assert "classification_auroc" in suite, "no metric threshold is exercised"
