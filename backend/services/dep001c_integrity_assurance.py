"""Disposable integrity fault injection for the DEP-001C harness."""
from __future__ import annotations

import stat
import tempfile
from pathlib import Path
from typing import Any

from backend.services.dep001c_integrity import (
    IntegrityViolation,
    atomic_write_json,
    canonical_hash,
    sha256_file,
    transition_transaction,
    utc_now,
    verify_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data/evals/safety/dep001c/integrity_fault_injection.json"


def run_integrity_fault_injection(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dep001c-integrity-fault-") as temporary:
        root = Path(temporary) / "candidate"
        root.mkdir(parents=True)
        artifact = root / "runtime/policy.bin"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"frozen-policy-v1")
        canonical = {
            "schema_version": "dep001c_disposable_snapshot_v1",
            "artifacts": {
                "runtime/policy.bin": {
                    "sha256": sha256_file(artifact),
                    "bytes": artifact.stat().st_size,
                }
            },
        }
        manifest_hash = canonical_hash(canonical)
        snapshot_id = f"dep001c-{manifest_hash[:20]}"
        manifest = {
            "snapshot_id": snapshot_id,
            "canonical_manifest_sha256": manifest_hash,
            "canonical_payload": canonical,
        }
        manifest_path = root / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        artifact.chmod(stat.S_IREAD)
        pre = verify_snapshot(manifest_path, expected_id=snapshot_id)

        transaction_path = Path(temporary) / "transaction.json"
        transaction = transition_transaction(transaction_path, state="PREPARED")
        transaction = transition_transaction(transaction_path, state="VERIFIED_PRE", transaction=transaction)
        artifact.chmod(stat.S_IREAD | stat.S_IWRITE)
        artifact.write_bytes(b"injected-mutation")
        checkpoint = verify_snapshot(manifest_path, expected_id=snapshot_id)
        transaction = transition_transaction(
            transaction_path,
            state="INVALIDATED",
            transaction=transaction,
            detail={"mismatches": checkpoint["mismatches"]},
        )
        promotion_rejected = False
        try:
            transition_transaction(transaction_path, state="COMMITTED", transaction=transaction)
        except IntegrityViolation:
            promotion_rejected = True
        passed = bool(pre["passed"] and not checkpoint["passed"] and promotion_rejected)
        payload = {
            "schema_version": "dep001c_integrity_fault_injection_v1",
            "generated_at": utc_now(),
            "status": "passed" if passed else "failed",
            "pre_mutation_integrity": pre["passed"],
            "mid_run_mutation_detected": not checkpoint["passed"],
            "transaction_invalidated": transaction["state"] == "INVALIDATED",
            "invalid_run_promotion_rejected": promotion_rejected,
            "frozen_artifact_mutated_for_test": True,
            "production_candidate_modified": False,
            "clinical_validation": False,
        }
        atomic_write_json(output_path, payload)
        return payload


__all__ = ["DEFAULT_OUTPUT", "run_integrity_fault_injection"]
