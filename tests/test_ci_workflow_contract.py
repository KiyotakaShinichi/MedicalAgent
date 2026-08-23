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
    ("backend pytest", ("pytest", "tests")),
    ("backend coverage floor", ("--cov-fail-under",)),
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
    """The floor is a regression gate; lowering it to pass a build is the failure."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "--cov-fail-under=60" in text, (
        "the enforced backend coverage floor is no longer 60 in CI"
    )


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
    needs = jobs["quality-gates"].get("needs") or []
    for required in ("static-quality", "full-offline-tests", "dependency-audit"):
        assert required in needs, f"quality-gates does not wait for {required}"
    assert "quality-gates" in (jobs["docker-build"].get("needs") or [])


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
