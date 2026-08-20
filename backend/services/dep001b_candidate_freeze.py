"""Hash-bind the DEP-001B implementation before the internal blind run."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEP001B_DIR = ROOT / "Data/evals/safety/dep001b"
MANIFEST_PATH = DEP001B_DIR / "candidate_freeze_manifest.json"
BLIND_RECEIPT_PATH = DEP001B_DIR / "internal_blind_run_receipt.json"

FROZEN_PATHS = (
    "config/dep001b_semantic_safety.yaml",
    "config/release_gate_thresholds.yaml",
    "backend/services/dep001b_candidate_freeze.py",
    "backend/services/dep001b_overlap_audit.py",
    "backend/services/safety_policy_action.py",
    "backend/services/dep001b_semantic_safety.py",
    "backend/services/dep001b_semantic_safety_training.py",
    "backend/services/dep001b_safety_corpus.py",
    "backend/services/dep001b_safety_evaluation.py",
    "backend/services/dep001b_runtime_assurance.py",
    "backend/services/dep001b_internal_blind_evaluation.py",
    "backend/services/agent_safety.py",
    "backend/services/agent_intent_router.py",
    "backend/services/agent_rag.py",
    "backend/services/post_generation_validator.py",
    "Data/evals/safety/dep001b/train_safety_bank.jsonl",
    "Data/evals/safety/dep001b/validation_safety_bank.jsonl",
    "Data/evals/safety/dep001b/internal_test_safety_bank.jsonl",
    "Data/evals/safety/dep001b/internal_blind_safety_bank.jsonl",
    "Data/evals/safety/dep001b/semantic_safety_model.joblib",
    "Data/evals/safety/dep001b/semantic_safety_calibration.joblib",
    "Data/evals/safety/dep001b/semantic_safety_thresholds.json",
    "Data/evals/safety/dep001b/semantic_safety_model_manifest.json",
    "Data/evals/safety/dep001b/latest_training_evaluation.json",
    "Data/evals/safety/dep001b/latest_overlap_audit.json",
    "Data/evals/safety/dep001b/latest_runtime_assurance.json",
)


def freeze_dep001b_candidate(
    *,
    root: Path = ROOT,
    manifest_path: Path = MANIFEST_PATH,
    frozen_paths: Iterable[str] = FROZEN_PATHS,
    receipt_path: Path = BLIND_RECEIPT_PATH,
) -> dict[str, Any]:
    if receipt_path.exists():
        raise RuntimeError("internal blind receipt already exists; candidate cannot be re-frozen")
    assurance = _read_json(root / "Data/evals/safety/dep001b/latest_runtime_assurance.json")
    overlap = _read_json(root / "Data/evals/safety/dep001b/latest_overlap_audit.json")
    if assurance.get("status") != "eligible_to_freeze":
        raise RuntimeError("runtime assurance is not eligible_to_freeze")
    if overlap.get("status") != "passed":
        raise RuntimeError("overlap audit has not passed")
    exact_counts = overlap.get("exact_overlap_counts")
    if not isinstance(exact_counts, dict) or not exact_counts:
        raise RuntimeError("overlap audit does not use the current exact-overlap schema")
    if any(not isinstance(value, int) or value != 0 for value in exact_counts.values()):
        raise RuntimeError("exact overlap is non-zero")
    if int(overlap.get("exact_burned_external_overlap_count", -1)) != 0:
        raise RuntimeError("burned external overlap is non-zero")
    if overlap.get("used_for_tuning") is not False:
        raise RuntimeError("overlap audit is not marked tuning-isolated")
    if overlap.get("external_text_or_case_ids_emitted") is not False:
        raise RuntimeError("overlap audit emitted external case content")

    artifacts: dict[str, Any] = {}
    for relative in frozen_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        artifacts[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    payload = {
        "schema_version": "dep001b_candidate_freeze_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_for_internal_blind",
        "candidate_frozen": True,
        "dep001_status": "blocked_pending_new_external_no_read_holdout",
        "frozen_artifacts": artifacts,
        "frozen_artifact_count": len(artifacts),
        "internal_blind_completed": False,
        "new_external_holdout_completed": False,
        "burned_external_holdout_rerun": False,
        "was_used_for_tuning": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_frozen_candidate(
    *, root: Path = ROOT, manifest_path: Path = MANIFEST_PATH
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    mismatches = []
    for relative, record in dict(manifest.get("frozen_artifacts") or {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            mismatches.append(relative)
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "manifest": manifest,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["BLIND_RECEIPT_PATH", "FROZEN_PATHS", "MANIFEST_PATH", "freeze_dep001b_candidate", "verify_frozen_candidate"]
