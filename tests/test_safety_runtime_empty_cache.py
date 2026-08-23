"""What the safety layer does on a first-time clone with an empty model cache.

The DEP-001 runtimes load their base encoder with
`SentenceTransformer(..., local_files_only=True)`. On a fresh clone that call
has nothing to load, and the interesting question is not whether it fails — it
must — but *how*.

The answer has to be **fail closed**: an unavailable classifier must make every
prompt look high-risk, never low-risk. A safety layer that degrades to
permissive when its model is missing is far worse than one that degrades to
refusing, because the failure is invisible in exactly the case that matters.
This file proves that property rather than assuming it, in a genuinely
isolated cache with the network denied.

It also proves the preflight catches the condition: an offline suite run
against fail-closed defaults would report ninety failures whose actual cause is
a missing model, so `--check-only` exits non-zero before the suite starts.

What this file deliberately does NOT do is substitute a fixture encoder. See
`test_a_bundled_fixture_encoder_cannot_substitute` for the arithmetic — the
required model's embedding matrix alone is larger than any reasonable fixture,
and a randomly-initialised stand-in would satisfy the loading contract while
silently destroying the classification contract the tracked joblib heads
implement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: Cache variables that must all be redirected. Missing one leaves a real
#: cached model reachable and the "empty cache" proof proves nothing.
CACHE_VARS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
)


def _isolated_env(cache_root: Path) -> dict[str, str]:
    """An environment with an empty model cache, offline, and no credentials."""
    env = {**os.environ}
    for name in CACHE_VARS:
        env[name] = str(cache_root / "hub")
    env["HF_HOME"] = str(cache_root)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["ENVIRONMENT"] = "test"
    # Never inherit a network opt-out or a live credential.
    env.pop("NLCARE_ALLOW_TEST_NETWORK", None)
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "PINECONE_API_KEY"):
        env.pop(name, None)
    return env


def _run_isolated(code: str, cache_root: Path) -> subprocess.CompletedProcess:
    """Run Python in a subprocess so the cache redirection actually applies.

    The huggingface_hub constants are read at import time, so an in-process
    monkeypatch would be ignored by an already-imported module.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, env=_isolated_env(cache_root), capture_output=True, text=True, timeout=900,
    )


@pytest.fixture
def empty_cache(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("empty-hf-cache")
    assert not list(root.rglob("*.safetensors")), "the fixture cache is not empty"
    return root


# ─── the empty cache is genuinely empty ──────────────────────────────────────


def test_the_encoder_is_absent_from_the_isolated_cache(empty_cache: Path) -> None:
    """Guards the guard: if the real cache leaked in, everything below is vacuous."""
    result = _run_isolated(
        "from sentence_transformers import SentenceTransformer\n"
        f"SentenceTransformer({REQUIRED_ENCODER!r}, local_files_only=True)\n"
        "print('LOADED')",
        empty_cache,
    )
    assert result.returncode != 0, (
        "the encoder loaded from an supposedly empty cache; the isolation is broken "
        f"and every other test here proves nothing.\n{result.stdout}"
    )
    assert "LOADED" not in result.stdout


# ─── the safety-critical property: fail closed, never open ───────────────────


def test_the_runtime_fails_closed_on_an_empty_cache(empty_cache: Path) -> None:
    """A missing classifier must make a benign prompt look risky, not safe.

    This is the whole reason an unprovisioned run is allowed to be useless
    rather than dangerous.
    """
    result = _run_isolated(
        "import json\n"
        "from backend.services.multilingual_semantic_safety import classify_multilingual_safety\n"
        "prediction = classify_multilingual_safety(["
        "'What does a complete blood count measure in general?'])\n"
        "print(json.dumps({"
        "'status': prediction.status,"
        "'risk_category': prediction.risk_category,"
        "'unsafe_probability': prediction.unsafe_probability,"
        "'uncertainty': prediction.uncertainty}))",
        empty_cache,
    )
    assert result.returncode == 0, f"the runtime raised instead of failing closed:\n{result.stderr[-2000:]}"

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "fail_closed"
    assert payload["risk_category"] == "classifier_unavailable"
    assert payload["unsafe_probability"] == 1.0, "fail-open: a benign prompt was treated as safe"
    assert payload["uncertainty"] == 1.0


def test_fail_closed_is_not_silently_permissive(empty_cache: Path) -> None:
    """The degraded state must be labelled, so a caller can tell it apart.

    A fail-closed decision that looked like a confident low-risk verdict would
    be indistinguishable from a real one.
    """
    result = _run_isolated(
        "from backend.services.multilingual_semantic_safety import classify_multilingual_safety\n"
        "p = classify_multilingual_safety(['Anong ibig sabihin ng CBC?'])\n"
        "print(p.status, p.risk_category)",
        empty_cache,
    )
    assert result.returncode == 0
    assert "fail_closed" in result.stdout
    assert "classifier_unavailable" in result.stdout


# ─── the preflight catches it before the suite runs ──────────────────────────


def test_preflight_check_only_exits_non_zero_on_an_empty_cache(empty_cache: Path) -> None:
    """Otherwise ninety tests fail for a reason none of them names."""
    result = subprocess.run(
        [sys.executable, "scripts/provision_semantic_safety_encoders.py", "--check-only"],
        cwd=ROOT, env=_isolated_env(empty_cache), capture_output=True, text=True, timeout=900,
    )
    assert result.returncode != 0, "the preflight passed with no encoder cached"
    assert REQUIRED_ENCODER in result.stdout


def test_preflight_explains_the_consequence(empty_cache: Path) -> None:
    """A failure that does not say what it means gets ignored or worked around."""
    result = subprocess.run(
        [sys.executable, "scripts/provision_semantic_safety_encoders.py", "--check-only"],
        cwd=ROOT, env=_isolated_env(empty_cache), capture_output=True, text=True, timeout=900,
    )
    combined = (result.stdout + result.stderr).lower()
    assert "fail closed" in combined
    assert "do not relax the tests" in combined


def test_preflight_with_runtime_verification_also_fails(empty_cache: Path) -> None:
    """`--verify-runtimes` is the stronger check and must not be weaker here."""
    result = subprocess.run(
        [
            sys.executable, "scripts/provision_semantic_safety_encoders.py",
            "--check-only", "--verify-runtimes",
        ],
        cwd=ROOT, env=_isolated_env(empty_cache), capture_output=True, text=True, timeout=900,
    )
    assert result.returncode != 0


def test_preflight_names_the_encoder_it_looked_for(empty_cache: Path) -> None:
    """A preflight that checked a different target than the run would be worthless.

    Asserted through the encoder name it reports, which is derived from the
    committed config and manifests. An earlier version of this test asserted on
    the script's cache diagnostic block instead; that block is only printed on
    some paths, so it passed locally and failed in CI against the very
    missing-encoder case this file exists to cover. The name is the stable,
    meaningful signal.
    """
    result = subprocess.run(
        [
            sys.executable, "scripts/provision_semantic_safety_encoders.py",
            "--check-only", "--verify-runtimes",
        ],
        cwd=ROOT, env=_isolated_env(empty_cache), capture_output=True, text=True, timeout=900,
    )
    combined = result.stdout + result.stderr

    assert result.returncode != 0
    assert REQUIRED_ENCODER in combined, (
        "the preflight did not say which encoder it looked for"
    )
    # And it is the encoder the committed runtime config actually names.
    manifest = json.loads(
        (ROOT / "Data/evals/safety/dep001d/runtime/semantic_safety_model_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["base_encoder"] == REQUIRED_ENCODER


# ─── why there is no bundled fixture encoder ─────────────────────────────────


def test_a_bundled_fixture_encoder_cannot_substitute() -> None:
    """The arithmetic that rules out a "tiny" local fixture.

    The runtimes load the encoder *by name* and verify only that its embedding
    dimension is 384, so a randomly-initialised stand-in would load. It could
    not be small, and it could not be correct:

    * **Size.** The tokenizer's vocabulary is 250,037 tokens. The embedding
      matrix alone is 250037 x 384 x 4 bytes, about 384 MB, before a single
      transformer layer. A measured one-layer build came to 389.4 MB against
      the real model's 479.7 MB - the depth is not what makes it large, the
      multilingual vocabulary is, and that cannot be shrunk without changing
      the tokenizer.
    * **Correctness.** The tracked classifier heads
      (`semantic_safety_model.joblib`) were fitted on the real encoder's
      384-dimensional space. Random weights produce a different distribution,
      so the heads would emit arbitrary labels while every structural check
      still passed - a fixture that exercises the loading contract and destroys
      the classification contract.

    So the encoder stays a provisioning step, and the empty-cache state is
    covered by proving it fails closed rather than by faking a model.
    """
    manifest_path = ROOT / "Data/evals/safety/dep001d/runtime/semantic_safety_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["base_encoder"] == REQUIRED_ENCODER
    assert manifest["embedding_dimension"] == 384

    # The heads that would be invalidated by a different embedding space are
    # tracked in the repository; only the encoder is not.
    head = ROOT / manifest["artifacts"]["model"]["path"]
    assert head.is_file(), "the classifier head should be tracked"
    assert head.stat().st_size > 1_000_000

    vocab_size = 250_037
    embedding_bytes = vocab_size * manifest["embedding_dimension"] * 4
    assert embedding_bytes > 300_000_000, (
        "if this ever becomes small, revisit bundling a fixture encoder"
    )


def test_the_offline_suite_documents_the_provisioning_step() -> None:
    """The requirement must be discoverable, since it cannot be removed."""
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "provision_semantic_safety_encoders" in contributing
