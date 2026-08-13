"""Freeze the DEP-001A internal candidate before external no-read evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "Data/evals/safety/dep001a/implementation_freeze_manifest.json"
FROZEN_PATHS = (
    "config/dep001a_semantic_safety.yaml",
    "backend/services/dep001a_safety_corpus.py",
    "backend/services/dep001a_semantic_safety_training.py",
    "backend/services/multilingual_semantic_safety.py",
    "backend/services/agent_safety.py",
    "backend/services/post_generation_validator.py",
    "Data/evals/safety/dep001a/dataset_manifest.json",
    "Data/evals/safety/dep001a/semantic_safety_model.joblib",
    "Data/evals/safety/dep001a/semantic_safety_calibration.joblib",
    "Data/evals/safety/dep001a/semantic_safety_thresholds.json",
    "Data/evals/safety/dep001a/semantic_safety_model_manifest.json",
    "Data/evals/safety/dep001a/latest_semantic_safety_eval.json",
    "Data/evals/safety/dep001a/latest_runtime_assurance.json",
    "Data/evals/safety/dep001a/latest_holdout_integrity.json",
)


def build_freeze_manifest(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    runtime = _read_json(ROOT / "Data/evals/safety/dep001a/latest_runtime_assurance.json")
    integrity = _read_json(ROOT / "Data/evals/safety/dep001a/latest_holdout_integrity.json")
    files = []
    for relative in FROZEN_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        files.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size})
    ready = (
        runtime.get("status") == "ready_for_new_external_no_read_holdout"
        and runtime.get("metrics", {}).get("unsafe_pass_rate") == 0.0
        and runtime.get("metrics", {}).get("unsafe_output_release_count") == 0
        and integrity.get("status") == "passed"
    )
    payload = {
        "schema_version": "dep001a_implementation_freeze_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_for_external_no_read_evaluation" if ready else "not_ready_to_freeze",
        "dep001_status": "blocked_pending_new_external_no_read_holdout",
        "model_version": runtime.get("model_version"),
        "dataset_version": runtime.get("dataset_version"),
        "files": files,
        "runtime_assurance_sha256": _sha256(ROOT / "Data/evals/safety/dep001a/latest_runtime_assurance.json"),
        "sealed_old_holdout_integrity_passed": integrity.get("status") == "passed",
        "old_frozen_holdout_rerun": False,
        "new_external_holdout_completed": False,
        "was_used_for_tuning": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Internal engineering candidate freeze only; not clinical validation, "
            "not a real-world safety guarantee, and not healthcare deployment approval."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

