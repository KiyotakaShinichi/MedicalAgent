"""Fail-closed preflight for an offline, behavior-only adapter experiment."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "finetune_candidate.json"
DATASET_PATH = ROOT / "Data" / "finetune" / "prepared" / "dataset_train.jsonl"
OUTPUT_PATH = ROOT / "Data" / "evals" / "models" / "latest_finetune_runtime_preflight.json"
ADJUDICATION_PATH = (
    ROOT
    / "Data"
    / "evals"
    / "models"
    / "latest_finetune_contamination_adjudication_readiness.json"
)
DEPENDENCIES = ("torch", "transformers", "peft", "accelerate")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _runtime_probe(timeout_seconds: int) -> dict[str, Any]:
    code = (
        "import json; import torch, transformers, peft, accelerate; "
        "print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,"
        "'peft':peft.__version__,'accelerate':accelerate.__version__,'cuda':torch.cuda.is_available()}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout_seconds, "healthy": False}
    if completed.returncode != 0:
        return {
            "status": "import_failed",
            "healthy": False,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-500:],
        }
    try:
        versions = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "invalid_probe_output", "healthy": False}
    return {"status": "healthy", "healthy": True, "versions": versions}


def build_finetune_runtime_preflight(
    output_path: Path = OUTPUT_PATH,
    *,
    execute_runtime_probe: bool = True,
    timeout_seconds: int = 15,
    adjudication_path: Path = ADJUDICATION_PATH,
) -> dict[str, Any]:
    candidate = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dependency_presence = {name: importlib.util.find_spec(name) is not None for name in DEPENDENCIES}
    runtime = _runtime_probe(timeout_seconds) if execute_runtime_probe else {
        "status": "not_executed_in_test",
        "healthy": False,
    }
    explicit_enable = os.getenv("NLCARE_FINETUNE_EXPERIMENT_ENABLED", "").lower() in {"1", "true", "yes"}
    adjudication = _read_json(adjudication_path)
    adjudication_complete = bool(
        adjudication.get("completed") is True
        and int(adjudication.get("unresolved_count") or 0) == 0
        and int(adjudication.get("critical_unresolved_count") or 0) == 0
    )
    config_ready = all(
        candidate.get(key) for key in ("model_id", "revision", "tokenizer_revision", "license", "official_model_card")
    ) and candidate.get("license_review", "").startswith("recorded_")
    ready = bool(
        config_ready
        and all(dependency_presence.values())
        and runtime["healthy"]
        and DATASET_PATH.exists()
        and explicit_enable
        and adjudication_complete
    )
    report = {
        "schema_version": "finetune_runtime_preflight_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_offline_experiment" if ready else "blocked_runtime",
        "clinical_validation": False,
        "model_trained": False,
        "adapter_created": False,
        "patient_facing_promotion_allowed": False,
        "candidate": candidate,
        "candidate_config_ready": config_ready,
        "dataset": {"path": str(DATASET_PATH.relative_to(ROOT)), "exists": DATASET_PATH.exists(), "sha256": _sha256(DATASET_PATH)},
        "dependency_presence": dependency_presence,
        "runtime_probe": runtime,
        "explicit_experiment_enable": explicit_enable,
        "contamination_adjudication": {
            "path": str(adjudication_path),
            "status": adjudication.get("status") or "missing",
            "completed": adjudication.get("completed") is True,
            "unresolved_count": int(adjudication.get("unresolved_count") or 0),
            "critical_unresolved_count": int(
                adjudication.get("critical_unresolved_count") or 0
            ),
            "cleared_for_runtime": adjudication_complete,
        },
        "ready_for_offline_experiment": ready,
        "next_step": (
            "Complete contamination adjudication, repair or provision an isolated supported training "
            "runtime, install pinned PEFT dependencies, then run baseline and candidate generations "
            "on the internal behavior eval."
        ),
        "claim_boundary": (
            "This is a runtime and lineage preflight only. No adapter was trained, no behavior improvement "
            "was demonstrated, and no clinical or patient-facing use is allowed."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


__all__ = ["build_finetune_runtime_preflight"]
