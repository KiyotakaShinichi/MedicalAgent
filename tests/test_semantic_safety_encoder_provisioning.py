"""Regression tests for semantic-safety encoder provisioning.

Background
----------
The DEP-001 safety runtimes load their base encoder with
``SentenceTransformer(..., local_files_only=True)``. When that model is absent
from the local Hugging Face cache the call raises and the runtime **fails
closed**: every prompt, benign or not, becomes ``UNKNOWN_HIGH_RISK`` with
``policy_action=FAIL_CLOSED``.

That behaviour is correct and is deliberately asserted here so it cannot be
"fixed" into failing open. What went wrong was that CI never provisioned the
encoder, so a fresh runner silently ran the whole offline suite against
fail-closed defaults instead of the real classifier — surfacing as ~89 tests
failing on prompts like "1+1".

These tests protect two things:

1. The provisioning script's notion of "required encoders" stays in sync with
   what the committed runtimes actually load. Retraining onto a new base
   encoder must not silently desynchronise CI.
2. The fail-closed contract itself.

None of these tests require the encoder to be present, so they pass in any
environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.provision_semantic_safety_encoders import (
    CONFIG_FILES,
    MANIFEST_GLOBS,
    required_encoders,
    verify_safety_runtimes,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/ci.yml",
    ROOT / ".github/workflows/ship.yml",
)


def test_required_encoders_is_not_empty() -> None:
    """An empty list would make provisioning a silent no-op."""
    assert required_encoders(ROOT), "no base encoders discovered from committed config"


def test_every_config_base_encoder_is_provisioned() -> None:
    """Each committed DEP-001 config's encoder must be in the provisioning set."""
    discovered = set(required_encoders(ROOT))
    for relative in CONFIG_FILES:
        path = ROOT / relative
        if not path.is_file():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        encoder = payload.get("base_encoder")
        assert encoder, f"{relative} declares no base_encoder"
        assert encoder in discovered, f"{relative} encoder {encoder!r} is not provisioned"


def test_every_runtime_manifest_base_encoder_is_provisioned() -> None:
    """The manifest is what the runtime loads, so it is the binding contract."""
    discovered = set(required_encoders(ROOT))
    seen_any = False
    for pattern in MANIFEST_GLOBS:
        for manifest in ROOT.glob(pattern):
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            encoder = payload.get("base_encoder")
            if not encoder:
                continue
            seen_any = True
            relative = manifest.relative_to(ROOT)
            assert encoder in discovered, f"{relative} encoder {encoder!r} is not provisioned"
    assert seen_any, "no runtime manifests discovered; the glob patterns may have drifted"


def test_dep001b_runtime_encoder_matches_provisioned_set() -> None:
    """Close the loop on the module the CI failures actually came from."""
    manifest = ROOT / "Data/evals/safety/dep001b/semantic_safety_model_manifest.json"
    if not manifest.is_file():
        return
    encoder = json.loads(manifest.read_text(encoding="utf-8")).get("base_encoder")
    assert encoder in set(required_encoders(ROOT))


def test_missing_runtime_artifacts_fail_closed_not_open(tmp_path: Path) -> None:
    """A safety runtime that cannot load must classify as unsafe, never safe.

    Pointing the runtime at an empty artifact directory reproduces the same
    exception path an absent encoder takes, without needing to manipulate the
    Hugging Face cache. This is the behaviour that made CI look broken; it is
    the correct behaviour and must stay.
    """
    from backend.services.dep001b_semantic_safety import classify_dep001b_safety

    prediction = classify_dep001b_safety("What is chemotherapy in general?", artifact_dir=tmp_path)

    assert prediction.policy_action == "FAIL_CLOSED"
    assert prediction.unsafe_probability == 1.0
    assert prediction.intent_family == "UNKNOWN_HIGH_RISK"
    assert prediction.failure_reason, "a fail-closed prediction must record why"


def test_runtime_probe_detects_a_degraded_runtime(tmp_path: Path, monkeypatch) -> None:
    """The verifier must catch a degraded runtime, not just a missing encoder.

    Regression for the gap that made CI unreadable: an encoder-only check exits
    0 while the DEP-001 runtime is failing closed on its joblib/hash/dimension
    checks, so the suite failed ~88 tests with no diagnostic. Pointing the
    runtime at an empty artifact directory reproduces that state deterministically
    and without touching the Hugging Face cache.
    """
    monkeypatch.setenv("NLCARE_DEP001B_ARTIFACT_DIR", str(tmp_path))

    results = verify_safety_runtimes()
    dep001b = next(r for r in results if r["runtime"] == "dep001b_semantic_safety")

    assert dep001b["ok"] is False, "a degraded runtime must not report ok"
    assert "FAIL_CLOSED" in dep001b["detail"]
    # The reason must name the underlying cause so CI is actionable.
    assert "failure_reason=" in dep001b["detail"]


def test_runtime_probe_returns_structured_results() -> None:
    """Every probe entry must be machine-readable for the CI evidence artifact."""
    for entry in verify_safety_runtimes():
        assert set(entry) == {"runtime", "ok", "detail"}
        assert isinstance(entry["ok"], bool)
        assert entry["detail"], "a probe result must explain itself"


def _workflow_steps(path: Path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in doc["jobs"].items():
        for step in job.get("steps") or []:
            yield job_name, step


def test_every_provisioning_job_verifies_runtimes_not_just_the_encoder() -> None:
    """Guards against silently reverting to the encoder-only check.

    A job that provisions encoders but verifies only that they load will let a
    degraded safety layer through to the test run.
    """
    offenders = []
    for path in WORKFLOWS:
        for job_name, step in _workflow_steps(path):
            run = step.get("run") or ""
            if "provision_semantic_safety_encoders.py" not in run:
                continue
            if "--check-only" in run and "--verify-runtimes" not in run:
                offenders.append(f"{path.name}:{job_name}:{step.get('name')}")
    assert not offenders, "verification steps that skip the runtime probe:\n  " + "\n  ".join(offenders)


def test_all_workflows_share_one_canonical_encoder_cache_path() -> None:
    """Provisioning and the test run must read the same cache root.

    If jobs cached different paths, a runner could provision into one location
    and read from another - passing verification while the tests fail closed.
    """
    cache_paths = set()
    for path in WORKFLOWS:
        for _job, step in _workflow_steps(path):
            if str(step.get("uses", "")).startswith("actions/cache"):
                with_block = step.get("with") or {}
                if "huggingface" in str(with_block.get("path", "")):
                    cache_paths.add(str(with_block["path"]).strip())
    assert cache_paths, "no Hugging Face cache step found in any workflow"
    assert len(cache_paths) == 1, f"workflows cache different HF paths: {cache_paths}"


def test_fail_closed_reason_is_distinguishable_from_a_real_unsafe_verdict(tmp_path: Path) -> None:
    """An operator must be able to tell 'model missing' from 'genuinely unsafe'.

    Without this, a provisioning outage looks identical to a spike in unsafe
    traffic.
    """
    from backend.services.dep001b_semantic_safety import classify_dep001b_safety

    prediction = classify_dep001b_safety("Who are you?", artifact_dir=tmp_path)

    assert prediction.policy_reason == "safety_signal_failure"
    assert prediction.failure_reason.startswith("dep001b_runtime_error:")
