"""One-shot transactional evaluator for immutable DEP-001D evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from backend.services.dep001d_candidate_snapshot import PREDECLARED_GATES, gates_pass
from backend.services.dep001d_integrity import (
    BLIND_ROOT, CANDIDATE_ROOT, LOCK_ROOT, RUN_ROOT, EvaluationLock,
    IntegrityViolation, atomic_write_json, canonical_hash,
    detect_conflicting_writers, sha256_file, transition_transaction,
    utc_now, verify_snapshot,
)


def run_dep001d_official_internal_once(
    *,
    candidate_id: str,
    blind_bank_id: str,
    candidate_root: Path = CANDIDATE_ROOT,
    blind_root: Path = BLIND_ROOT,
    run_root: Path = RUN_ROOT,
    lock_root: Path = LOCK_ROOT,
    process_rows: list[Mapping[str, Any]] | None = None,
    checkpoint_interval: int = 100,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    candidate_dir = candidate_root / candidate_id
    blind_dir = blind_root / blind_bank_id
    candidate_manifest_path = candidate_dir / "manifest.json"
    blind_manifest_path = blind_dir / "manifest.json"
    if not candidate_manifest_path.is_file() or not blind_manifest_path.is_file():
        raise FileNotFoundError("explicit immutable candidate and blind manifests are required")
    candidate_manifest_sha = sha256_file(candidate_manifest_path)
    blind_manifest_sha = sha256_file(blind_manifest_path)
    run_identity = {
        "candidate_id": candidate_id,
        "blind_bank_id": blind_bank_id,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "blind_manifest_sha256": blind_manifest_sha,
        "predeclared_gates": PREDECLARED_GATES,
    }
    run_id = f"dep001d-run-{canonical_hash(run_identity)[:20]}"
    run_dir = run_root / run_id
    receipt_path = run_dir / "receipt.json"
    transaction_path = run_dir / "transaction.json"
    if receipt_path.exists() or transaction_path.exists():
        raise IntegrityViolation("dep001d_blind_is_one_shot_and_already_consumed")
    conflicts = detect_conflicting_writers(process_rows)
    if conflicts:
        raise IntegrityViolation(f"conflicting_writer_processes:{len(conflicts)}")
    run_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "dep001d_official_receipt_v1",
        "run_id": run_id, **run_identity,
        "status": "started_no_rerun", "started_at": utc_now(), "rerun_allowed": False,
    }
    atomic_write_json(receipt_path, receipt)
    transaction: dict[str, Any] = {
        "schema_version": "dep001d_evidence_transaction_v1",
        "run_id": run_id, "candidate_id": candidate_id,
        "blind_bank_id": blind_bank_id, "clinical_validation": False,
    }
    transaction = transition_transaction(transaction_path, state="PREPARED", transaction=transaction)
    try:
        with EvaluationLock(candidate_id=candidate_id, run_id=run_id, lock_root=lock_root) as lock:
            transaction = transition_transaction(
                transaction_path, state="LOCKED", transaction=transaction,
                detail={"stale_lock_recovered": lock.recovered_stale_lock},
            )
            pre_candidate = verify_snapshot(candidate_manifest_path, expected_id=candidate_id)
            pre_blind = verify_snapshot(blind_manifest_path, expected_id=blind_bank_id)
            pre = {
                "schema_version": "dep001d_pre_run_integrity_v1",
                "generated_at": utc_now(), "candidate": pre_candidate,
                "blind_bank": pre_blind,
                "passed": bool(pre_candidate["passed"] and pre_blind["passed"]),
            }
            atomic_write_json(run_dir / "integrity_pre.json", pre)
            if not pre["passed"]:
                raise IntegrityViolation("pre_run_integrity_failure")
            transaction = transition_transaction(transaction_path, state="VERIFIED_PRE", transaction=transaction)
            transaction = transition_transaction(transaction_path, state="RUNNING", transaction=transaction)

            worker_output = run_dir / "worker_result.json"
            progress_path = run_dir / "checkpoint_integrity.json"
            environment = dict(os.environ)
            environment.update({
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(candidate_dir / "source"),
                "NLCARE_DEP001B_ARTIFACT_DIR": str(candidate_dir / "runtime"),
                "NLCARE_DEP001D_ARTIFACT_DIR": str(candidate_dir / "runtime"),
            })
            completed = subprocess.run(
                [
                    sys.executable, str(candidate_dir / "source/dep001d_snapshot_worker.py"),
                    "--candidate-manifest", str(candidate_manifest_path),
                    "--blind-manifest", str(blind_manifest_path),
                    "--output", str(worker_output),
                    "--progress", str(progress_path),
                    "--checkpoint-interval", str(checkpoint_interval),
                ],
                cwd=candidate_dir, env=environment, capture_output=True, text=True,
                timeout=timeout_seconds, check=False,
            )
            if completed.returncode != 0:
                raise IntegrityViolation(
                    f"frozen_worker_failed:{completed.returncode}:{completed.stderr[-240:]}"
                )

            post_candidate = verify_snapshot(candidate_manifest_path, expected_id=candidate_id)
            post_blind = verify_snapshot(blind_manifest_path, expected_id=blind_bank_id)
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            checkpoints = list(progress.get("checkpoints") or [])
            checkpoints_passed = bool(
                checkpoints and all(
                    row.get("candidate_passed") and row.get("blind_passed")
                    for row in checkpoints
                )
            )
            post = {
                "schema_version": "dep001d_post_run_integrity_v1",
                "generated_at": utc_now(), "candidate": post_candidate,
                "blind_bank": post_blind, "checkpoint_count": len(checkpoints),
                "checkpoint_integrity_passed": checkpoints_passed,
                "candidate_manifest_sha256_unchanged": sha256_file(candidate_manifest_path) == candidate_manifest_sha,
                "blind_manifest_sha256_unchanged": sha256_file(blind_manifest_path) == blind_manifest_sha,
            }
            post["passed"] = bool(
                post_candidate["passed"] and post_blind["passed"] and checkpoints_passed
                and post["candidate_manifest_sha256_unchanged"]
                and post["blind_manifest_sha256_unchanged"]
            )
            atomic_write_json(run_dir / "integrity_post.json", post)
            if not post["passed"]:
                raise IntegrityViolation("post_run_integrity_failure")
            transaction = transition_transaction(transaction_path, state="VERIFIED_POST", transaction=transaction)

            worker = json.loads(worker_output.read_text(encoding="utf-8"))
            candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
            fault = json.loads((candidate_dir / "assurance/fault_injection.json").read_text(encoding="utf-8"))
            integrity_fault = json.loads((candidate_dir / "assurance/integrity_fault_injection.json").read_text(encoding="utf-8"))
            fault_rate = float(fault.get("pass_rate", 0.0))
            behavior_passed = gates_pass(worker["metrics"], fault_injection_pass_rate=fault_rate)
            if int(worker["metrics"]["unsafe_released_outputs"]) > 0:
                decision = "BLOCKED_UNSAFE_RELEASE"
            elif not behavior_passed:
                decision = "BLOCKED_BEHAVIORAL"
            else:
                decision = "INTERNAL_PASS_EXTERNAL_REQUIRED"
            result = {
                "schema_version": "dep001d_official_internal_evaluation_v1",
                "generated_at": utc_now(), "run_id": run_id, "status": decision,
                "integrity_valid": True, "behavioral_gates_passed": behavior_passed,
                "candidate_id": candidate_id,
                "candidate_manifest_path": str(candidate_manifest_path.resolve()),
                "candidate_manifest_sha256": candidate_manifest_sha,
                "frozen_artifact_count": candidate_manifest["frozen_artifact_count"],
                "blind_bank_id": blind_bank_id,
                "blind_bank_manifest_path": str(blind_manifest_path.resolve()),
                "blind_bank_manifest_sha256": blind_manifest_sha,
                "blind_bank_sha256": json.loads(blind_manifest_path.read_text(encoding="utf-8"))["blind_bank_sha256"],
                "cases_evaluated": worker["case_n"], "pre_run_integrity": pre,
                "checkpoint_integrity_passed": checkpoints_passed,
                "post_run_integrity": post, "metrics": worker["metrics"],
                "confidence_intervals": worker["confidence_intervals"],
                "failed_case_ids": worker["failed_case_ids"],
                "predeclared_gates": PREDECLARED_GATES,
                "fault_injection": fault, "fault_injection_pass_rate": fault_rate,
                "integrity_fault_injection": integrity_fault,
                "candidate_frozen": True,
                "ready_for_new_external_holdout": decision == "INTERNAL_PASS_EXTERNAL_REQUIRED",
                "dep001_status": "BLOCKED", "dep001_complete": False,
                "new_external_no_read_holdout_required": True,
                "clinical_validation": False, "healthcare_production_ready": False,
                "limitations": [
                    "The bank is internally generated synthetic engineering evidence.",
                    "The perfect development metrics may reflect grammar-family homogeneity.",
                    "A future independent no-read external holdout remains required.",
                ],
            }
            atomic_write_json(run_dir / "result.json", result)
            transaction = transition_transaction(
                transaction_path, state="COMMITTED", transaction=transaction,
                detail={"decision": decision, "result_sha256": sha256_file(run_dir / "result.json")},
            )
            receipt.update({
                "status": "completed_no_rerun", "completed_at": utc_now(),
                "transaction_state": transaction["state"], "decision": decision,
                "result_sha256": sha256_file(run_dir / "result.json"),
            })
            atomic_write_json(receipt_path, receipt)
            return result
    except Exception as exc:
        transaction = transition_transaction(
            transaction_path, state="INVALIDATED", transaction=transaction,
            detail={"failure_type": type(exc).__name__, "failure_code": str(exc)[:240]},
        )
        receipt.update({
            "status": "invalidated_no_rerun", "invalidated_at": utc_now(),
            "transaction_state": transaction["state"], "failure_type": type(exc).__name__,
        })
        atomic_write_json(receipt_path, receipt)
        atomic_write_json(run_dir / "invalidated.json", {
            "schema_version": "dep001d_invalidated_run_v1", "run_id": run_id,
            "status": "BLOCKED_EVIDENCE_INTEGRITY",
            "behavioral_metrics_admissible": False,
            "failure_type": type(exc).__name__, "failure_code": str(exc)[:240],
            "candidate_frozen": False, "ready_for_new_external_holdout": False,
            "dep001_status": "BLOCKED", "clinical_validation": False,
        })
        raise


__all__ = ["run_dep001d_official_internal_once"]
