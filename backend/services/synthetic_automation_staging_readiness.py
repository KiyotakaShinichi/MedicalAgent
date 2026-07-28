"""Validate the local synthetic n8n/MailHog staging contract."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSE = Path("docker-compose.synthetic-automation.yml")
DEFAULT_WORKFLOW = Path("infra/n8n/synthetic_high_risk_review_alert.json")
DEFAULT_OUTPUT = Path("Data/evals/ops/latest_synthetic_automation_staging_readiness.json")


def build_synthetic_automation_staging_readiness(
    *,
    root: str | Path = ROOT,
    compose_path: str | Path = DEFAULT_COMPOSE,
    workflow_path: str | Path = DEFAULT_WORKFLOW,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    compose_file = _resolve(repo, compose_path)
    workflow_file = _resolve(repo, workflow_path)
    output_file = _resolve(repo, output_path)
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    workflow = json.loads(workflow_file.read_text(encoding="utf-8"))
    services = compose.get("services") or {}
    workflow_text = workflow_file.read_text(encoding="utf-8")
    compose_text = compose_file.read_text(encoding="utf-8")

    checks = [
        _check("n8n_service_defined", "n8n" in services),
        _check("mailhog_service_defined", "mailhog" in services),
        _check("ports_bound_to_loopback", "127.0.0.1:5678:5678" in compose_text and "127.0.0.1:8025:8025" in compose_text),
        _check("workflow_inactive", workflow.get("active") is False),
        _check("synthetic_only", workflow.get("meta", {}).get("synthetic_only") is True),
        _check("hmac_verification_present", "createHmac('sha256'" in workflow_text),
        _check("constant_time_compare_present", "timingSafeEqual" in workflow_text),
        _check("replay_window_bounded", workflow.get("meta", {}).get("replay_window_seconds") == 300),
        _check("phi_blocklist_present", "raw_patient_message" in workflow_text and "patient_id" in workflow_text),
        _check("test_recipient_only", "nlcare-synthetic-review@invalid.example" in workflow_text),
        _check("clinical_validation_false", workflow.get("meta", {}).get("clinical_validation") is False),
        _check("production_ready_false", workflow.get("meta", {}).get("healthcare_production_ready") is False),
    ]
    compose_validation = _validate_compose(compose_file)
    passed = all(item["passed"] for item in checks)
    status = (
        "ready_for_synthetic_runtime"
        if passed and compose_validation["valid"]
        else "prepared_needs_attention"
    )
    payload = {
        "schema_version": "synthetic_automation_staging_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "passed_count": sum(int(item["passed"]) for item in checks),
        "check_count": len(checks),
        "compose_validation": compose_validation,
        "workflow_import_completed": False,
        "runtime_completed": False,
        "external_delivery_completed": False,
        "human_acknowledgement_completed": False,
        "synthetic_recipient_only": True,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "next_actions": [
            "Start docker-compose.synthetic-automation.yml on a disposable local machine.",
            "Import the inactive workflow and configure a no-auth MailHog SMTP credential.",
            "Send only signed synthetic events and record a delivery receipt drill.",
            "Keep real email, SMS, and Viber recipients disabled pending organizational review.",
        ],
        "claim_boundary": (
            "This artifact validates static local staging configuration. It does not "
            "claim a running n8n workflow, external delivery, clinician acknowledgement, "
            "production automation, medical safety, or clinical validation."
        ),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _validate_compose(path: Path) -> dict[str, Any]:
    docker = shutil.which("docker")
    if not docker:
        return {"available": False, "executed": False, "valid": False, "reason": "docker CLI unavailable"}
    try:
        result = subprocess.run(
            [docker, "compose", "-f", str(path), "config", "--quiet"],
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "executed": True, "valid": False, "reason": str(exc)}
    return {
        "available": True,
        "executed": True,
        "valid": result.returncode == 0,
        "exit_code": result.returncode,
        "stderr_tail": result.stderr[-500:],
    }


def _check(check_id: str, passed: bool) -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed)}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


__all__ = ["build_synthetic_automation_staging_readiness"]
