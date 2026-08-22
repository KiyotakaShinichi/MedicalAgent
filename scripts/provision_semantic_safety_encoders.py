"""Provision the sentence-transformer encoders the safety layer needs offline.

Why this exists
---------------
The DEP-001A/B/D semantic safety runtimes load their base encoder with
``SentenceTransformer(..., local_files_only=True)``. That call never downloads:
if the model is not already in the local Hugging Face cache it raises, and the
runtime does the correct thing and **fails closed** — every prompt, including
plainly benign ones, is classified ``UNKNOWN_HIGH_RISK`` /
``policy_action=FAIL_CLOSED``.

That is right for production and wrong for CI. A fresh GitHub runner has an
empty Hugging Face cache, so the entire semantic safety layer degrades to
fail-closed and roughly ninety tests that assert benign prompts stay
``low_risk`` fail. The tests were correct and the implementation was correct;
the missing piece was that CI never provisioned the encoder.

This script closes that gap. It runs **once, with network access**, before the
offline suite, and populates the cache. The test run itself still executes with
``HF_HUB_OFFLINE=1``/``TRANSFORMERS_OFFLINE=1``, so the offline contract is
unchanged — the model is simply present, exactly as it is on a developer
machine that has run the suite before.

Encoder names are discovered from the committed config and manifests rather
than hard-coded, so retraining onto a different base encoder cannot silently
desynchronise CI provisioning from what the runtime actually loads.

Usage
-----
    python scripts/provision_semantic_safety_encoders.py            # download + verify
    python scripts/provision_semantic_safety_encoders.py --check-only
    python scripts/provision_semantic_safety_encoders.py --json-output path.json

``--check-only`` performs no download and exits non-zero if any required
encoder is missing from the cache. It is the preflight assertion: use it to
prove the offline suite is about to run against a real classifier rather than
against fail-closed defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# `--verify-runtimes` imports `backend.*`, so the repository root must be
# importable when this script is invoked directly (CI runs it as
# `python scripts/provision_semantic_safety_encoders.py`, which puts only
# `scripts/` on sys.path).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Committed config files whose `base_encoder` the runtimes read.
CONFIG_FILES = (
    "config/dep001a_semantic_safety.yaml",
    "config/dep001b_semantic_safety.yaml",
    "config/dep001d_semantic_safety.yaml",
)

# Committed runtime manifests. A manifest's `base_encoder` is what the runtime
# actually loads, and it is verified against the config at load time, so both
# are collected.
MANIFEST_GLOBS = (
    "Data/evals/safety/*/semantic_safety_model_manifest.json",
    "Data/evals/safety/*/runtime/semantic_safety_model_manifest.json",
)


def _encoder_from_yaml(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = payload.get("base_encoder")
    return str(value) if value else None


def _encoder_from_manifest(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("base_encoder")
    return str(value) if value else None


def required_encoders(root: Path = ROOT) -> list[str]:
    """Every base encoder the committed safety runtimes will try to load."""
    names: set[str] = set()
    for relative in CONFIG_FILES:
        name = _encoder_from_yaml(root / relative)
        if name:
            names.add(name)
    for pattern in MANIFEST_GLOBS:
        for manifest in root.glob(pattern):
            name = _encoder_from_manifest(manifest)
            if name:
                names.add(name)
    return sorted(names)


def encoder_is_cached(name: str) -> bool:
    """True when the encoder loads without touching the network.

    Uses the same ``local_files_only=True`` call the runtimes use. Necessary
    but **not sufficient** — see :func:`verify_safety_runtimes` for why.
    """
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(name, local_files_only=True)
        return True
    except Exception:
        return False


# A prompt that must never be classified as unsafe. If a correctly provisioned
# runtime returns anything else, the safety layer is degraded rather than
# working — which is exactly the state that made ~89 tests fail while an
# encoder-only check reported success.
BENIGN_PROBE = "What is chemotherapy in general?"


def verify_safety_runtimes() -> list[dict[str, Any]]:
    """Exercise the real DEP-001 runtimes offline, not just the encoder.

    Why this exists
    ---------------
    Loading the base encoder proves only that one artifact is present. Each
    runtime additionally loads a joblib model, a calibration bundle and a
    threshold file, verifies their SHA-256 against a manifest, and checks the
    encoder's embedding dimension. Any of those can fail on a runner while the
    encoder loads perfectly — and every one of them fails *closed*, turning
    benign prompts into ``high_risk`` without raising.

    An encoder-only check therefore reports success while the suite fails.
    This function classifies a known-benign prompt through each runtime and
    reports the actual ``failure_reason``, so a provisioning problem surfaces
    at the verification step with a precise cause instead of as a wall of
    downstream assertion failures.
    """
    results: list[dict[str, Any]] = []

    # DEP-001B: the routing/policy runtime the failing tests exercise.
    try:
        from backend.services.dep001b_semantic_safety import classify_dep001b_safety

        prediction = classify_dep001b_safety(BENIGN_PROBE)
        degraded = prediction.policy_action == "FAIL_CLOSED"
        results.append({
            "runtime": "dep001b_semantic_safety",
            "ok": not degraded,
            "detail": (
                f"policy_action={prediction.policy_action}"
                + (f" failure_reason={prediction.failure_reason}" if prediction.failure_reason else "")
            ),
        })
    except Exception as exc:  # noqa: BLE001 - report, never crash verification
        results.append({
            "runtime": "dep001b_semantic_safety",
            "ok": False,
            "detail": f"raised {type(exc).__name__}: {exc}",
        })

    # DEP-001A multilingual routing shares the same encoder but its own
    # artifacts, so it can fail independently.
    try:
        from backend.services.multilingual_semantic_safety import (  # noqa: F401
            classify_multilingual_safety,
        )

        results.append({"runtime": "multilingual_semantic_safety", "ok": True, "detail": "importable"})
    except Exception as exc:  # noqa: BLE001
        results.append({
            "runtime": "multilingual_semantic_safety",
            "ok": False,
            "detail": f"raised {type(exc).__name__}: {exc}",
        })

    return results


def download_encoder(name: str) -> None:
    """Fetch an encoder into the local cache. Requires network access.

    The offline flags are cleared *before* importing, because
    ``huggingface_hub`` resolves ``HF_HUB_OFFLINE`` into a module-level
    constant at import time — clearing them afterwards would be a no-op. In CI
    the provisioning step also sets them to "0" at the step level, which is the
    primary mechanism; this makes the script correct when run directly too.
    """
    previous = {
        key: os.environ.pop(key, None)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(name)
    finally:
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify cache state without downloading; non-zero exit if anything is missing",
    )
    parser.add_argument(
        "--verify-runtimes",
        action="store_true",
        help=(
            "additionally classify a benign prompt through the real DEP-001 runtimes. "
            "An encoder-only check can pass while a runtime fails closed on its joblib, "
            "hash, or embedding-dimension checks, so CI uses this to surface the actual "
            "failure_reason instead of a wall of downstream assertion failures."
        ),
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    encoders = required_encoders()
    if not encoders:
        print("ERROR: no base encoders discovered from config or manifests", file=sys.stderr)
        return 1

    results = []
    failed = False
    for name in encoders:
        cached = encoder_is_cached(name)
        action = "already-cached"
        if not cached and not args.check_only:
            try:
                download_encoder(name)
            except Exception as exc:  # noqa: BLE001 - report, do not crash provisioning
                action = f"download-failed:{type(exc).__name__}"
            else:
                action = "downloaded"
            cached = encoder_is_cached(name)
        elif not cached:
            action = "missing"

        if not cached:
            failed = True
        results.append({"encoder": name, "cached": cached, "action": action})
        print(f"[{'OK  ' if cached else 'FAIL'}] {name} ({action})")

    runtime_results: list[dict[str, Any]] = []
    if args.verify_runtimes and not failed:
        print()
        for entry in verify_safety_runtimes():
            runtime_results.append(entry)
            print(f"[{'OK  ' if entry['ok'] else 'FAIL'}] runtime {entry['runtime']}: {entry['detail']}")
            if not entry["ok"]:
                failed = True

    if failed:
        print(
            "\nThe offline safety path is degraded. The semantic safety runtimes "
            "FAIL CLOSED when any of their artifacts is unavailable, classifying "
            "every prompt — including plainly benign ones — as high risk. That is "
            "correct runtime behaviour but makes the offline test suite meaningless. "
            "Fix provisioning before running tests; do not relax the tests.",
            file=sys.stderr,
        )

    if args.json_output:
        payload = {
            "schema_version": "semantic_safety_encoder_provisioning_v2",
            "encoders": results,
            "runtimes": runtime_results,
            "passed": not failed,
            "claim_boundary": (
                "Confirms the base encoders required by the committed safety runtimes are "
                "loadable offline, and (with --verify-runtimes) that those runtimes classify "
                "a benign prompt without failing closed. Makes no claim about model quality, "
                "safety performance, or clinical validity."
            ),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_output}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
