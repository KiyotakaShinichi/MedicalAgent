"""Frozen-source worker used by a DEP-001C official internal evaluation.

The orchestrator launches the copy inside the immutable candidate snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Mapping


SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from backend.services.dep001b_safety_evaluation import evaluate_dep001b_rows
from backend.services.dep001b_semantic_safety import classify_dep001b_safety


_MUTABLE = re.compile(r"^(latest|current)(?:[._-]|$)", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--blind-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    for path in (args.candidate_manifest, args.blind_manifest):
        if any(_MUTABLE.match(part) for part in path.parts):
            raise RuntimeError("mutable_alias_rejected")

    candidate = _read_json(args.candidate_manifest)
    blind = _read_json(args.blind_manifest)
    checkpoints: list[dict[str, Any]] = []

    def checkpoint(case_count: int) -> None:
        candidate_verification = _verify(args.candidate_manifest)
        blind_verification = _verify(args.blind_manifest)
        record = {
            "case_count": int(case_count),
            "candidate_passed": candidate_verification["passed"],
            "blind_passed": blind_verification["passed"],
            "candidate_mismatches": candidate_verification["mismatches"],
            "blind_mismatches": blind_verification["mismatches"],
        }
        checkpoints.append(record)
        _atomic_json(args.progress, {"checkpoints": checkpoints})
        if not record["candidate_passed"] or not record["blind_passed"]:
            raise RuntimeError("checkpoint_integrity_failure")

    checkpoint(0)
    candidate_dir = args.candidate_manifest.parent
    blind_dir = args.blind_manifest.parent
    bank_relative = str(blind["canonical_payload"]["bank_path"])
    bank_path = blind_dir / bank_relative
    rows = _jsonl(bank_path)
    classifier = partial(
        classify_dep001b_safety,
        artifact_dir=candidate_dir / "runtime",
    )
    scored = evaluate_dep001b_rows(
        rows,
        classifier=classifier,
        include_case_results=True,
        checkpoint_callback=checkpoint,
        checkpoint_interval=args.checkpoint_interval,
    )
    checkpoint(len(rows))
    payload = {
        "schema_version": "dep001c_frozen_worker_result_v1",
        "candidate_id": candidate["candidate_id"],
        "blind_bank_id": blind["blind_bank_id"],
        "case_n": len(rows),
        "metrics": scored["metrics"],
        "confidence_intervals": scored["confidence_intervals"],
        "failed_case_ids": scored["failed_case_ids"],
        "case_results": scored["cases"],
        "checkpoint_count": len(checkpoints),
        "prompt_text_persisted": False,
    }
    _atomic_json(args.output, payload)
    return 0


def _verify(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    canonical = manifest["canonical_payload"]
    expected_manifest_hash = str(manifest["canonical_manifest_sha256"])
    mismatches = []
    if _canonical_hash(canonical) != expected_manifest_hash:
        mismatches.append("manifest:canonical_hash")
    root = manifest_path.parent
    for relative, record in dict(canonical["artifacts"]).items():
        path = root / relative
        if not path.is_file():
            mismatches.append(f"missing:{relative}")
        elif _sha256(path) != str(record["sha256"]):
            mismatches.append(f"hash:{relative}")
    return {"passed": not mismatches, "mismatches": mismatches}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected_json_object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
