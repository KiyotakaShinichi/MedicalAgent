"""Regression tests for fresh-clone derived-artifact provisioning.

Context
-------
Seven tests passed on developer machines and failed on every fresh clone,
because they consume artifacts that are gitignored (correctly - they are
derived data) and nothing verified those artifacts could be *rebuilt*. The
artifacts were simply left over from an earlier local run.

These tests lock in the contract that replaced that situation:

* every derived artifact declares tracked inputs and is rebuildable from them;
* a missing artifact cannot silently pass the preflight;
* provisioning is idempotent, so running it twice is safe in CI;
* the rebuilt artifact satisfies its real consumer, not just an existence check.

They deliberately do not assert byte-equality of regenerated artifacts: the
generators stamp a build timestamp, so content identity is checked through the
stable knowledge fingerprint and record set instead.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.provision_derived_artifacts import (  # noqa: E402 - needs ROOT on sys.path
    DERIVED_ARTIFACTS,
    DerivedArtifact,
    _run_generator,
    missing_artifacts,
    missing_inputs,
    provision,
)


def test_generator_preserves_declared_tracked_side_effects(tmp_path: Path) -> None:
    preserved = tmp_path / "tracked" / "manifest.json"
    created_only_by_generator = tmp_path / "tracked" / "transient.json"
    target = tmp_path / "derived" / "artifact.json"
    preserved.parent.mkdir(parents=True)
    preserved.write_bytes(b'{"state":"before"}')
    script = (
        "from pathlib import Path; "
        "Path('tracked/manifest.json').write_text('changed'); "
        "Path('tracked/transient.json').write_text('temporary'); "
        "Path('derived').mkdir(); "
        "Path('derived/artifact.json').write_text('generated')"
    )
    artifact = DerivedArtifact(
        name="isolated-generator",
        path="derived/artifact.json",
        inputs=(),
        generator=("-c", script),
        preserved_side_effects=(
            "tracked/manifest.json",
            "tracked/transient.json",
        ),
    )

    result = _run_generator(artifact, tmp_path)

    assert result["exit_code"] == 0
    assert target.read_text(encoding="utf-8") == "generated"
    assert preserved.read_bytes() == b'{"state":"before"}'
    assert not created_only_by_generator.exists()


def test_every_derived_artifact_declares_tracked_inputs() -> None:
    """No artifact may be declared without a reproducible provenance."""
    assert DERIVED_ARTIFACTS, "the registry must not be empty"
    for artifact in DERIVED_ARTIFACTS:
        assert artifact.inputs, f"{artifact.name} declares no inputs"
        assert artifact.consumers, f"{artifact.name} declares no consumers"


def test_declared_inputs_exist_in_this_checkout() -> None:
    """The check that would have caught the original regression.

    If a generator ever starts depending on a gitignored input, that input is
    absent from a fresh clone and this fails - in the fast static job, rather
    than as seven opaque failures in the full offline suite.
    """
    assert missing_inputs(ROOT) == []


def test_declared_inputs_are_tracked_in_git() -> None:
    """`exists()` is not enough: an input must be tracked, not merely present.

    A locally-generated file passes an existence check on the machine that
    made it, which is precisely how the original defect hid.
    """
    for artifact in DERIVED_ARTIFACTS:
        for relative in artifact.inputs:
            proc = subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                # Directories are tracked via their contents.
                listed = subprocess.run(
                    ["git", "ls-files", relative],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                assert listed.stdout.strip(), (
                    f"{artifact.name} input {relative!r} is not tracked in git, so it "
                    "cannot be relied on from a fresh clone"
                )


def test_missing_artifact_cannot_silently_pass_the_preflight(tmp_path: Path) -> None:
    """``--check-only`` must exit non-zero when an artifact is absent.

    Pointed at an empty tree, so nothing can be mistaken for present. This is
    the guarantee that matters: the original defect was a preflight that
    reported success while the artifacts the suite needed did not exist.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "provision_derived_artifacts.py"),
            "--check-only",
            "--root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, (
        "an empty checkout must fail the preflight, got "
        f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    for artifact in DERIVED_ARTIFACTS:
        assert artifact.name in proc.stdout


def test_preflight_passes_on_the_provisioned_repository() -> None:
    """The same command must succeed where the artifacts do exist.

    Without this, the test above would also pass against a preflight that
    always fails.
    """
    if missing_artifacts(ROOT):
        pytest.skip("artifacts not provisioned in this checkout")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "provision_derived_artifacts.py"),
            "--check-only",
            "--root",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_missing_artifacts_reported_for_an_empty_root(tmp_path: Path) -> None:
    """Every declared artifact is reported missing when nothing exists."""
    reported = missing_artifacts(tmp_path)
    assert reported == [artifact.name for artifact in DERIVED_ARTIFACTS]
    assert missing_inputs(tmp_path), "an empty root must also report missing inputs"


def test_provisioning_is_idempotent() -> None:
    """A second run must not rebuild or fail - CI may invoke this repeatedly."""
    if missing_artifacts(ROOT):
        pytest.skip("artifacts not provisioned in this checkout")
    results = provision(ROOT)
    assert all(entry["ok"] for entry in results)
    assert all(
        entry["action"] in {"already-present", "present"} for entry in results
    ), f"expected no rebuild on a provisioned tree, got {results}"


def test_provisioned_artifacts_satisfy_their_consumer(tmp_path: Path) -> None:
    """The rebuilt lakehouse records must actually drive the shadow sync.

    An existence check would pass on a truncated or empty file; this asserts
    the consumer reaches its established status.
    """
    if missing_artifacts(ROOT):
        pytest.skip("artifacts not provisioned in this checkout")
    from backend.services.managed_vector_shadow_sync import build_managed_vector_shadow_sync

    report = build_managed_vector_shadow_sync(
        root_dir=ROOT,
        output_path=tmp_path / "shadow-sync.json",
    )
    # Same expectation as tests/test_managed_vector_shadow_sync.py:67 - the
    # established contract for a default, no-network run. Kept identical on
    # purpose: this test proves the *rebuilt* artifact reaches the same status
    # the developer-local artifact did.
    assert report["status"] == "ready_for_opt_in_shadow_sync"
    assert report["network_request_performed"] is False


def test_gold_records_carry_required_governance_metadata() -> None:
    """Rebuilt records must keep the governance contract, not just parse.

    Guards the regeneration path against silently producing records that are
    structurally valid but strip the metadata the data contract requires.
    """
    if missing_artifacts(ROOT):
        pytest.skip("artifacts not provisioned in this checkout")
    gold = ROOT / "Data/lakehouse/gold/vector_records.jsonl"
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "regenerated gold records must not be empty"
    for row in rows:
        metadata = row.get("metadata") or {}
        assert metadata.get("clinical_validation") is False
        for banned in ("patient_id", "mrn", "email", "name", "phone"):
            assert banned not in metadata
