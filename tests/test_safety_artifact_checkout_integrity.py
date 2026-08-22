"""Every DEP-001 runtime artifact must hash correctly on *any* platform.

The defect this guards against
------------------------------
Each DEP-001 runtime verifies its threshold and dataset-manifest files against
a SHA-256 recorded in `semantic_safety_model_manifest.json` before it will
serve a prediction. Five of those hashes were recorded over CRLF bytes on a
Windows machine, while git stores the files normalised to LF.

The result was invisible locally and fatal in CI: on Windows `core.autocrlf`
restored CRLF and the hashes matched, so every developer saw green. On Linux
the runtime received LF content, the hash check raised, and the runtime FAILED
CLOSED — classifying every prompt, including plainly benign ones, as high risk.
`Verify safety runtimes load offline` failed on GitHub for exactly this reason,
taking `full-offline-tests`, Ship Gate, and every downstream job with it.

The fix pins those paths in `.gitattributes` so a checkout reproduces the
signed bytes everywhere, without altering one byte of DEP-001 evidence.

This test asserts the invariant directly rather than the fix: for every
artifact a runtime hashes, the bytes a fresh checkout produces on *this*
platform must match the recorded hash. It fails on Linux if a pin is missing,
and on Windows if a pin is wrong — which is what makes it worth having.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Manifests whose `artifacts` block a runtime verifies before loading.
RUNTIME_MANIFESTS = (
    "Data/evals/safety/dep001a/semantic_safety_model_manifest.json",
    "Data/evals/safety/dep001b/semantic_safety_model_manifest.json",
    "Data/evals/safety/dep001d/runtime/semantic_safety_model_manifest.json",
)


def _artifact_records() -> list[tuple[str, str, str]]:
    """(manifest, relative_path, expected_sha256) for every verified artifact."""
    records: list[tuple[str, str, str]] = []
    for manifest_rel in RUNTIME_MANIFESTS:
        manifest_path = ROOT / manifest_rel
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in (payload.get("artifacts") or {}).values():
            path = record.get("path")
            sha = record.get("sha256")
            if path and sha:
                # Manifests store Windows-style separators for some entries.
                records.append((manifest_rel, str(path).replace("\\", "/"), str(sha)))
    return records


def test_manifests_are_discoverable() -> None:
    """Guards the guard: an empty record set would make everything below vacuous."""
    records = _artifact_records()
    assert len(records) >= 9, f"expected the DEP-001 runtime artifacts, found {len(records)}"


@pytest.mark.parametrize("manifest_rel,rel,expected", _artifact_records())
def test_checked_out_artifact_matches_recorded_hash(
    manifest_rel: str, rel: str, expected: str
) -> None:
    """The working-tree bytes must hash to what the manifest recorded.

    This is what the runtime itself does at load time, so a failure here is a
    fail-closed safety runtime on this platform.
    """
    path = ROOT / rel
    assert path.is_file(), f"{rel} is referenced by {manifest_rel} but absent"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"{rel} does not match the SHA-256 recorded in {manifest_rel}.\n"
        "If this fails on Linux and passes on Windows the cause is line endings: "
        "the hash was recorded over CRLF bytes and git stores the file as LF. "
        "Pin the path in .gitattributes with `text eol=crlf`; do not re-record "
        "the hash, which would modify DEP-001 evidence."
    )


@pytest.mark.parametrize("rel", sorted({rel for _, rel, _ in _artifact_records()}))
def test_text_artifacts_hashed_as_crlf_are_pinned(rel: str) -> None:
    """A CRLF-hashed artifact must be pinned, or it breaks on a Linux checkout.

    Compares the recorded hash against the bytes git *stores* (always LF). When
    those differ, the checkout depends on an end-of-line conversion, and only an
    explicit `.gitattributes` rule makes that conversion happen on every
    platform rather than only where `core.autocrlf` is on.
    """
    expected = {r: sha for _, r, sha in _artifact_records()}[rel]
    blob = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{rel}"],
        cwd=ROOT, capture_output=True,
    )
    if blob.returncode != 0:
        pytest.skip(f"{rel} not present at HEAD (uncommitted working tree)")

    if hashlib.sha256(blob.stdout).hexdigest() == expected:
        return  # stored bytes already hash correctly; no pin needed

    attr = subprocess.run(
        ["git", "check-attr", "text", "eol", "--", rel],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    assert "eol: crlf" in attr, (
        f"{rel} is hashed as CRLF but git stores it as LF, and no .gitattributes "
        f"rule pins it. A Linux checkout will hand the runtime LF bytes, the "
        f"SHA-256 check will raise, and the DEP-001 runtime will FAIL CLOSED — "
        f"classifying benign prompts as high risk.\nGot: {attr.strip()}"
    )
