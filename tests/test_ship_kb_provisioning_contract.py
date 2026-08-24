"""Retrieval tests need a retrieval index, and the runner has to build one.

The RAG chunk artifact and vector index are gitignored and rebuilt from tracked
inputs under `KnowledgeBase/raw`. A job that runs the retrieval suites without
rebuilding them does not fail loudly: retrieval still scores candidates against
the corpus, the tier filter then drops every one of them for want of ingested
chunks, and the suite fails on empty results rather than on anything it was
testing. On a fresh runner that read as `ingested_chunks: 0`, `retrieved_count:
19`, `tier_kept: 0`.

`quality-gates` provisions with `scripts/provision_derived_artifacts.py`, and
the fresh-clone job gets the same script through `scripts/bootstrap.py`. The
Ship Gate ran neither, which is what these tests pin.

The provisioning is deliberately not something this contract reinvents: it
asserts the Ship Gate uses the same script the already-green paths use, so
there is one ingestion implementation rather than a workflow-shaped copy of one.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.provision_derived_artifacts import DERIVED_ARTIFACTS  # noqa: E402

SHIP_WORKFLOW = ROOT / ".github/workflows/ship.yml"
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
BOOTSTRAP = ROOT / "scripts/bootstrap.py"

PROVISIONER = "scripts/provision_derived_artifacts.py"
GATE = "scripts/ship.py"

# The suite whose retrieval starved, and the artifacts it consumes.
RETRIEVAL_SUITE = "tests/test_breast_monitoring.py"


def _steps(workflow: Path, job: str) -> list[dict]:
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    return data["jobs"][job].get("steps") or []


def _lines(steps: list[dict]) -> list[str]:
    lines: list[str] = []
    for step in steps:
        lines.extend(str(step.get("run") or "").splitlines())
    return lines


def _ship_lines() -> list[str]:
    return _lines(_steps(SHIP_WORKFLOW, "ship"))


def _first(lines: list[str], needle: str) -> int | None:
    return next((i for i, line in enumerate(lines) if needle in line), None)


# --- the Ship Gate provisions, and does it first -----------------------------


def test_the_ship_gate_provisions_derived_artifacts() -> None:
    assert _first(_ship_lines(), PROVISIONER) is not None, (
        "the Ship Gate never rebuilds the RAG chunk artifact or vector index; "
        "its retrieval suites will run against an empty index"
    )


def test_provisioning_happens_before_the_gate_runs() -> None:
    """Ordering, not mere presence: after ship.py has run it is too late."""
    lines = _ship_lines()
    provision_at = _first(lines, PROVISIONER)
    gate_at = _first(lines, GATE)

    assert provision_at is not None, "no provisioning step"
    assert gate_at is not None, "no ship.py step"
    assert provision_at < gate_at, (
        f"provisioning runs at line {provision_at} but the gate starts at "
        f"{gate_at}; the suites would still see an empty index"
    )


def test_provisioning_failure_is_not_swallowed() -> None:
    """If ingestion fails, the gate must not run on a half-built index."""
    for step in _steps(SHIP_WORKFLOW, "ship"):
        if PROVISIONER in str(step.get("run") or ""):
            assert step.get("continue-on-error") is not True, (
                "provisioning is marked continue-on-error; the gate would run "
                "against whatever partial state it left behind"
            )


def test_the_gate_verifies_provisioning_actually_produced_the_artifacts() -> None:
    """`--check-only` exits non-zero when an artifact is missing.

    Without it a generator that silently produced nothing would surface much
    later as an unexplained retrieval failure, which is exactly the failure
    mode this whole contract exists to prevent.
    """
    lines = _ship_lines()
    assert any(PROVISIONER in line and "--check-only" in line for line in lines), (
        "the Ship Gate generates artifacts but never verifies they exist"
    )


# --- it is the same path the green workflows use -----------------------------


def test_quality_gates_uses_the_same_provisioner() -> None:
    lines = _lines(_steps(CI_WORKFLOW, "quality-gates"))
    assert _first(lines, PROVISIONER) is not None, (
        "quality-gates no longer uses this script; the Ship Gate would be "
        "provisioning by a path nothing else exercises"
    )


def test_bootstrap_uses_the_same_provisioner() -> None:
    """The fresh-clone job reaches the same script through bootstrap.py."""
    assert PROVISIONER in BOOTSTRAP.read_text(encoding="utf-8"), (
        "bootstrap.py no longer provisions derived artifacts"
    )


def test_no_second_ingestion_implementation_was_invented() -> None:
    """The Ship Gate must not hand-roll ingestion inline."""
    lines = _ship_lines()
    for line in lines:
        if "ingest_knowledge_base.py" in line:
            pytest.fail(
                "the Ship Gate calls the ingestion script directly; it should "
                f"go through {PROVISIONER}, which owns ordering between the "
                "chunk artifact and the pipeline that reads it"
            )


# --- the built artifacts stay out of the repository --------------------------


def test_the_retrieval_suite_is_a_declared_consumer() -> None:
    """The link between the failing suite and these artifacts, made explicit."""
    consumers = {c for artifact in DERIVED_ARTIFACTS for c in artifact.consumers}
    assert RETRIEVAL_SUITE in consumers, (
        f"{RETRIEVAL_SUITE} is no longer a declared consumer of the derived "
        "artifacts; this contract's premise needs rechecking"
    )


def test_generated_artifacts_are_not_committed() -> None:
    """They are rebuilt from tracked inputs, so they must not be in the tree."""
    tracked = []
    for artifact in DERIVED_ARTIFACTS:
        result = subprocess.run(
            ["git", "ls-files", "--", artifact.path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            tracked.append(artifact.path)

    assert not tracked, f"generated artifacts are committed: {tracked}"


def test_the_inputs_that_rebuild_them_are_tracked() -> None:
    """Otherwise a fresh runner has nothing to rebuild from."""
    for artifact in DERIVED_ARTIFACTS:
        for source in artifact.inputs:
            result = subprocess.run(
                ["git", "ls-files", "--", source],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            assert result.stdout.strip(), (
                f"{source!r} rebuilds {artifact.name} but is not tracked"
            )


# --- provisioning stays offline ----------------------------------------------


def test_the_ship_gate_does_not_enable_test_network_access() -> None:
    """Provisioning is offline; nothing here may quietly open the network."""
    text = SHIP_WORKFLOW.read_text(encoding="utf-8")
    assert "NLCARE_ALLOW_TEST_NETWORK" not in text, (
        "the Ship Gate enables test network access; derived artifacts are "
        "meant to be rebuilt offline from tracked inputs"
    )


@pytest.mark.parametrize(
    "script",
    ["scripts/provision_derived_artifacts.py", "scripts/ingest_knowledge_base.py"],
)
def test_the_provisioning_scripts_make_no_network_calls(script: str) -> None:
    """A generator that downloaded its corpus would not be reproducible."""
    source = (ROOT / script).read_text(encoding="utf-8")
    for marker in ("import requests", "import httpx", "urllib.request", "urlopen"):
        assert marker not in source, f"{script} appears to reach the network ({marker})"
    assert not re.search(r"https?://\S+\"", source), f"{script} embeds a URL"
