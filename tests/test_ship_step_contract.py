"""The ship gate's step list survived decomposition unchanged.

`scripts/ship.py` used to define all 80 steps in a single 477-line
`_build_steps()` literal. They now live in `scripts/ship_steps/`, grouped by
responsibility. That move is only safe if it changed *nothing* about what runs:
the ship gate is the release contract, so a dropped step, a reordered pair, a
lost `--fail-under`, or a widened timeout would quietly weaken it.

`tests/contracts/ship_steps_baseline.json` is a frozen snapshot of the step list
taken from the pre-decomposition `_build_steps()`. These tests replay it against
the current package and require an exact match on name, command, working
directory, environment, timeout, and position.

The baseline is a *historical record*, not an expectation to be refreshed. A
deliberate future change to the ship gate should update it in the same commit as
the change, so the diff shows the step contract moving on purpose.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ship import (  # noqa: E402
    _build_post_success_reconciliation_steps,
    _build_steps,
)
from scripts.ship_steps import STEP_GROUPS  # noqa: E402

BASELINE_PATH = ROOT / "tests" / "contracts" / "ship_steps_baseline.json"
FRONTEND = ROOT / "frontend-react"


def _tokenize(value: str) -> str:
    """Map a runtime value onto the fixture's machine-independent tokens."""
    if value == sys.executable:
        return "{python}"
    if value == str(ROOT):
        return "{root}"
    if value == str(FRONTEND):
        return "{root}/frontend-react"
    if value in ("npm.cmd", "npm"):
        return "{npm}"
    return value


def _snapshot(step) -> dict:
    data = dataclasses.asdict(step)
    return {
        "name": data["name"],
        "command": [_tokenize(str(part)) for part in data["command"]],
        "cwd": _tokenize(str(data["cwd"])),
        "env": data["env"],
        "timeout_seconds": data["timeout_seconds"],
    }


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    return {
        "main": [_snapshot(s) for s in _build_steps()],
        "post_success": [
            _snapshot(s) for s in _build_post_success_reconciliation_steps()
        ],
    }


# ─── the equivalence proof ───────────────────────────────────────────────────


def test_step_count_is_unchanged(baseline: dict, current: dict) -> None:
    """A step lost in the move is a check that silently stops running."""
    assert len(current["main"]) == len(baseline["main"])
    assert len(current["post_success"]) == len(baseline["post_success"])


def test_step_order_is_unchanged(baseline: dict, current: dict) -> None:
    """Order is a real dependency: later steps consume earlier steps' evidence."""
    assert [s["name"] for s in current["main"]] == [s["name"] for s in baseline["main"]]


def test_every_step_is_byte_identical(baseline: dict, current: dict) -> None:
    """Command, cwd, env, and timeout, position by position.

    Reported per step rather than as one opaque list diff, so a failure names
    the step and field that moved.
    """
    for index, (before, after) in enumerate(zip(baseline["main"], current["main"])):
        assert after == before, (
            f"ship step {index} ({before['name']!r}) changed during decomposition:\n"
            f"  before: {before}\n  after:  {after}"
        )


def test_post_success_reconciliation_is_unchanged(baseline: dict, current: dict) -> None:
    assert current["post_success"] == baseline["post_success"]


def test_full_contract_matches(baseline: dict, current: dict) -> None:
    """Whole-structure equality, catching anything the field checks miss."""
    assert current == {"main": baseline["main"], "post_success": baseline["post_success"]}


# ─── properties the grouped layout must keep ─────────────────────────────────


def test_groups_concatenate_to_the_step_list(current: dict) -> None:
    """`all_steps()` must be exactly the groups in `STEP_GROUPS` order.

    Guards the seam the decomposition introduced: a group could be defined,
    linted, and imported correctly yet left out of the tuple, and nothing else
    would notice.
    """
    from_groups = [_snapshot(s) for group in STEP_GROUPS for s in group()]
    assert from_groups == current["main"]


def test_no_group_is_empty() -> None:
    for group in STEP_GROUPS:
        assert group(), f"{group.__module__} contributes no steps"


def test_step_names_are_unique(current: dict) -> None:
    """Names key the resume cache and the run manifest; duplicates corrupt both."""
    names = [s["name"] for s in current["main"]]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate ship step names: {sorted(duplicates)}"


def test_thresholds_are_preserved(baseline: dict, current: dict) -> None:
    """Numeric gate arguments, extracted and compared as a set.

    A decomposition that dropped a `--fail-under` or relaxed a `--min-*` would
    still produce a runnable step list, and only this comparison would catch it.
    """

    def thresholds(steps: list[dict]) -> set[str]:
        found = set()
        for step in steps:
            for i, part in enumerate(step["command"]):
                if part.startswith("--") and any(
                    k in part for k in ("min", "max", "fail-under", "threshold", "limit")
                ):
                    value = step["command"][i + 1] if i + 1 < len(step["command"]) else ""
                    found.add(f"{step['name']}::{part}={value}")
        return found

    assert thresholds(current["main"]) == thresholds(baseline["main"])


def test_timeouts_are_preserved(baseline: dict, current: dict) -> None:
    """A widened timeout weakens a hang-detection gate without failing anything."""
    assert {(s["name"], s["timeout_seconds"]) for s in current["main"]} == {
        (s["name"], s["timeout_seconds"]) for s in baseline["main"]
    }


def test_environment_overrides_are_preserved(baseline: dict, current: dict) -> None:
    """Step env selects fast/sparse modes; losing it changes what is exercised."""
    assert [(s["name"], s["env"]) for s in current["main"]] == [
        (s["name"], s["env"]) for s in baseline["main"]
    ]


# ─── failure behaviour of the shared execution loop ──────────────────────────
#
# The main pass and the post-success reconciliation pass used to be two
# near-identical 70-line blocks. They now share `_execute_steps`. These tests
# pin the behaviour that duplication used to guarantee: exit codes, manifest
# failure labels, and the step count recorded for each phase.


@pytest.fixture
def harness(monkeypatch):
    """Drive `ship.main()` over fake steps, capturing every manifest write."""
    import subprocess

    from scripts import ship

    main_steps = [ship.Step(name=f"step-{i}", command=["x"]) for i in range(3)]
    recon_steps = [ship.Step(name="recon-0", command=["y"])]
    manifests: list[dict] = []

    def run(*, fail_step=None, mode=None, tier="release"):
        def fake_run(step, *, dependency_fingerprint):
            if step.name == fail_step:
                if mode == "timeout":
                    raise subprocess.TimeoutExpired(cmd=step.command, timeout=99)
                raise subprocess.CalledProcessError(returncode=7, cmd=step.command)
            return {"name": step.name, "status": "passed"}

        monkeypatch.setattr(ship, "_build_steps", lambda: list(main_steps))
        monkeypatch.setattr(
            ship, "_build_post_success_reconciliation_steps", lambda: list(recon_steps)
        )
        monkeypatch.setattr(ship, "_select_steps", lambda steps, _tier: list(steps))
        monkeypatch.setattr(ship, "_run", fake_run)
        monkeypatch.setattr(ship, "_dependency_fingerprint", lambda step: "fp")
        monkeypatch.setattr(ship, "_effective_timeout", lambda step: 99)
        monkeypatch.setattr(
            ship, "_write_manifest", lambda **kw: manifests.append(kw)
        )
        return ship.main(["--tier", tier]), manifests

    return run


def test_a_timeout_exits_124_and_is_labelled(harness) -> None:
    code, manifests = harness(fail_step="step-1", mode="timeout")
    assert code == 124
    assert manifests[-1]["status"] == "failed"
    assert manifests[-1]["failure_kind"] == "timeout"
    assert manifests[-1]["failed_step"] == "step-1"


def test_a_nonzero_exit_is_propagated_verbatim(harness) -> None:
    """The step's own exit code must reach the caller; CI branches on it."""
    code, manifests = harness(fail_step="step-2", mode="nonzero")
    assert code == 7
    assert manifests[-1]["failure_kind"] == "nonzero_exit"


def test_reconciliation_failures_keep_their_own_label(harness) -> None:
    """A post-success failure is distinguishable from a gate failure."""
    code, manifests = harness(fail_step="recon-0", mode="timeout")
    assert code == 124
    assert manifests[-1]["failure_kind"] == "post_success_reconciliation_timeout"


def test_reconciliation_failure_counts_both_phases(harness) -> None:
    _, manifests = harness(fail_step="recon-0", mode="nonzero")
    assert manifests[-1]["failure_kind"] == "post_success_reconciliation_nonzero_exit"
    assert manifests[-1]["selected_step_count"] == 4  # 3 main + 1 reconciliation


def test_a_green_release_writes_passed_twice(harness) -> None:
    """Once after the gate, once after reconciliation - the pre-existing shape."""
    code, manifests = harness()
    assert code == 0
    assert [m["status"] for m in manifests] == ["passed", "passed"]
    assert [m["selected_step_count"] for m in manifests] == [3, 4]


def test_non_release_tiers_skip_reconciliation(harness) -> None:
    code, manifests = harness(tier="fast")
    assert code == 0
    assert len(manifests) == 1
    assert manifests[0]["selected_step_count"] == 3
