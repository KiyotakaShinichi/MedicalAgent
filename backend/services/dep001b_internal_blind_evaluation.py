"""One-shot internal blind evaluation for a frozen DEP-001B candidate."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.dep001b_candidate_freeze import (
    BLIND_RECEIPT_PATH,
    MANIFEST_PATH,
    ROOT,
    verify_frozen_candidate,
)
from backend.services.dep001b_safety_evaluation import evaluate_dep001b_rows


DEP001B_DIR = ROOT / "Data/evals/safety/dep001b"
BLIND_PATH = DEP001B_DIR / "internal_blind_safety_bank.jsonl"
OUTPUT_PATH = DEP001B_DIR / "latest_internal_blind_evaluation.json"
RAW_RESULTS_PATH = DEP001B_DIR / "internal_blind_raw_results.json"


def run_dep001b_internal_blind_once(
    *,
    root: Path = ROOT,
    blind_path: Path = BLIND_PATH,
    manifest_path: Path = MANIFEST_PATH,
    receipt_path: Path = BLIND_RECEIPT_PATH,
    output_path: Path = OUTPUT_PATH,
    raw_results_path: Path = RAW_RESULTS_PATH,
) -> dict[str, Any]:
    if receipt_path.exists():
        raise RuntimeError("internal blind evaluation is one-shot and has already started")
    verification = verify_frozen_candidate(root=root, manifest_path=manifest_path)
    if not verification["passed"]:
        raise RuntimeError(f"frozen candidate hash mismatch: {verification['mismatches']}")
    manifest = verification["manifest"]
    expected = manifest["frozen_artifacts"]["Data/evals/safety/dep001b/internal_blind_safety_bank.jsonl"]["sha256"]
    if _sha256(blind_path) != expected:
        raise RuntimeError("internal blind bank hash does not match frozen candidate")

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    started = {
        "schema_version": "dep001b_internal_blind_receipt_v1",
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest_sha256": _sha256(manifest_path),
        "blind_bank_sha256": expected,
        "rerun_allowed": False,
    }
    receipt_path.write_text(json.dumps(started, indent=2), encoding="utf-8")
    try:
        rows = _jsonl(blind_path)
        scored = evaluate_dep001b_rows(rows, include_case_results=True)
        assurance = json.loads(
            (root / "Data/evals/safety/dep001b/latest_runtime_assurance.json").read_text(encoding="utf-8")
        )
        faults_passed = bool(assurance.get("fault_injection", {}).get("passed"))
        passed = bool(scored["targets_passed"] and faults_passed)
        raw_payload = {
            "schema_version": "dep001b_internal_blind_raw_results_v1",
            "case_results": scored["cases"],
            "prompt_text_persisted": False,
        }
        raw_results_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
        payload = {
            "schema_version": "dep001b_internal_blind_evaluation_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": (
                "passed_internal_blind_pending_new_external_holdout"
                if passed else "failed_internal_blind"
            ),
            "completed": True,
            "candidate_frozen": True,
            "ready_for_new_external_holdout": passed,
            "dep001_status": "blocked_pending_new_external_no_read_holdout",
            "evaluation_scope": "internally authored blind engineering bank; not independent external evidence",
            "metrics": scored["metrics"],
            "confidence_intervals": scored["confidence_intervals"],
            "targets_passed": scored["targets_passed"],
            "failed_case_ids": scored["failed_case_ids"],
            "fault_injection": assurance["fault_injection"],
            "candidate_manifest_sha256": _sha256(manifest_path),
            "blind_bank_sha256": expected,
            "raw_results_sha256": _sha256(raw_results_path),
            "burned_external_holdout_rerun": False,
            "new_external_human_holdout_required": True,
            "new_external_holdout_completed": False,
            "clinical_validation": False,
            "healthcare_production_ready": False,
            "limitations": [
                "The bank was withheld from training, threshold selection, and runtime-assurance development.",
                "The bank shares an internal compositional authoring process and is not external evidence.",
                "A new independently authored no-read holdout is required to unblock DEP-001.",
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        completed = {
            **started,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result_status": payload["status"],
            "result_sha256": _sha256(output_path),
        }
        receipt_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
        return payload
    except Exception as exc:
        failed = {
            **started,
            "status": "failed_no_rerun",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "failure_type": type(exc).__name__,
        }
        receipt_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        raise


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["run_dep001b_internal_blind_once"]
