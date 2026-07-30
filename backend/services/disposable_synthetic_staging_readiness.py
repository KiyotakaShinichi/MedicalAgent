"""Static and optional runtime readiness for disposable synthetic staging."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSE_PATH = Path("docker-compose.synthetic-staging.yml")
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/ops/latest_disposable_synthetic_staging_readiness.json"
)
REQUIRED_SERVICES = {
    "postgres",
    "redis",
    "backend",
    "worker",
    "frontend",
    "n8n",
    "mailhog",
}
EXTERNALLY_EXPOSED_SERVICES = {
    "postgres",
    "redis",
    "backend",
    "frontend",
    "n8n",
    "mailhog",
}

CLAIM_BOUNDARY = (
    "This artifact validates a disposable loopback-only synthetic staging "
    "definition and, when Docker is available, compose syntax. It does not "
    "prove a running deployment, managed-cloud resilience, real external "
    "delivery, patient-data handling, clinical validation, or production "
    "healthcare readiness."
)


def build_disposable_synthetic_staging_readiness(
    *,
    root: str | Path = ROOT,
    compose_path: str | Path = DEFAULT_COMPOSE_PATH,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    compose_file = _resolve(repo, compose_path)
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    services = compose.get("services") or {}
    checks = [
        _check("required_services_defined", REQUIRED_SERVICES <= set(services)),
        _check(
            "all_published_ports_loopback_only",
            all(
                _ports_are_loopback(services.get(service) or {})
                for service in EXTERNALLY_EXPOSED_SERVICES
            ),
        ),
        _check(
            "backend_and_worker_synthetic_only",
            all(
                str(
                    (services.get(service) or {})
                    .get("environment", {})
                    .get("NLCARE_SYNTHETIC_ONLY", "")
                ).lower()
                == "true"
                for service in ("backend", "worker")
            ),
        ),
        _check(
            "external_automation_delivery_disabled",
            all(
                str(
                    (services.get(service) or {})
                    .get("environment", {})
                    .get("N8N_WEBHOOK_DISPATCH_ENABLED", "")
                ).lower()
                == "false"
                for service in ("backend", "worker")
            ),
        ),
        _check(
            "managed_vector_network_disabled",
            all(
                str(
                    (services.get(service) or {})
                    .get("environment", {})
                    .get("NLCARE_MANAGED_VECTOR_ALLOW_NETWORK", "")
                ).lower()
                == "false"
                for service in ("backend", "worker")
            ),
        ),
        _check(
            "managed_vector_shadow_disabled",
            all(
                str(
                    (services.get(service) or {})
                    .get("environment", {})
                    .get("NLCARE_MANAGED_VECTOR_SHADOW_ENABLED", "")
                ).lower()
                == "false"
                for service in ("backend", "worker")
            ),
        ),
        _check(
            "stateful_services_have_healthchecks",
            all(
                bool((services.get(service) or {}).get("healthcheck"))
                for service in ("postgres", "redis", "backend", "n8n")
            ),
        ),
        _check(
            "named_disposable_state_volumes",
            {
                "staging_postgres",
                "staging_redis",
                "staging_node_modules",
                "staging_n8n",
            }
            <= set(compose.get("volumes") or {}),
        ),
    ]
    compose_validation = _validate_compose(compose_file)
    static_passed = all(check["passed"] for check in checks)
    return {
        "schema_version": "disposable_synthetic_staging_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ready_for_disposable_synthetic_runtime"
            if static_passed
            and (
                compose_validation["valid"]
                or not compose_validation["available"]
            )
            else "needs_attention"
        ),
        "compose_path": str(compose_path).replace("\\", "/"),
        "services": sorted(services),
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "check_count": len(checks),
        "compose_validation": compose_validation,
        "runtime_started": False,
        "runtime_healthchecks_completed": False,
        "postgres_restore_drill_completed": False,
        "n8n_workflow_import_completed": False,
        "mailhog_delivery_receipt_completed": False,
        "managed_vector_network_call_completed": False,
        "real_external_delivery_completed": False,
        "patient_data_processed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_disposable_synthetic_staging_readiness(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_disposable_synthetic_staging_readiness(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _ports_are_loopback(service: dict[str, Any]) -> bool:
    ports = service.get("ports") or []
    return bool(ports) and all(
        isinstance(port, str) and port.startswith("127.0.0.1:")
        for port in ports
    )


def _validate_compose(path: Path) -> dict[str, Any]:
    docker = shutil.which("docker")
    if not docker:
        return {
            "available": False,
            "executed": False,
            "valid": False,
            "reason": "docker CLI unavailable; static checks only",
        }
    try:
        result = subprocess.run(
            [docker, "compose", "-f", str(path), "config", "--quiet"],
            text=True,
            capture_output=True,
            timeout=45,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "available": True,
            "executed": True,
            "valid": False,
            "reason": str(exc),
        }
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


__all__ = [
    "build_disposable_synthetic_staging_readiness",
    "write_disposable_synthetic_staging_readiness",
]
