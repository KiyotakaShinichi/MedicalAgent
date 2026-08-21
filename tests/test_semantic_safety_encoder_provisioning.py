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
)

ROOT = Path(__file__).resolve().parents[1]


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


def test_fail_closed_reason_is_distinguishable_from_a_real_unsafe_verdict(tmp_path: Path) -> None:
    """An operator must be able to tell 'model missing' from 'genuinely unsafe'.

    Without this, a provisioning outage looks identical to a spike in unsafe
    traffic.
    """
    from backend.services.dep001b_semantic_safety import classify_dep001b_safety

    prediction = classify_dep001b_safety("Who are you?", artifact_dir=tmp_path)

    assert prediction.policy_reason == "safety_signal_failure"
    assert prediction.failure_reason.startswith("dep001b_runtime_error:")
