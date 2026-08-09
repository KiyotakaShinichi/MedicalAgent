"""Static and optional runtime readiness for disposable synthetic staging."""

from __future__ import annotations

import hashlib
import json
import smtplib
import socket
import shutil
import subprocess
import time
import urllib.request
import uuid
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
    "definition and may record executable local runtime drills when explicitly "
    "requested. A healthy local run does not prove managed-cloud resilience, "
    "real external delivery, patient-data handling, clinical validation, or "
    "production healthcare readiness."
)


def build_disposable_synthetic_staging_readiness(
    *,
    root: str | Path = ROOT,
    compose_path: str | Path = DEFAULT_COMPOSE_PATH,
    runtime_observations: dict[str, Any] | None = None,
    runtime_evidence_source: str | None = None,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    compose_file = _resolve(repo, compose_path)
    compose_bytes = compose_file.read_bytes()
    compose = yaml.safe_load(compose_bytes.decode("utf-8"))
    compose_sha256 = hashlib.sha256(compose_bytes).hexdigest()
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
    runtime = runtime_observations or {}
    runtime_started = bool(runtime.get("runtime_started", False))
    runtime_healthy = bool(runtime.get("runtime_healthchecks_completed", False))
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
        "compose_sha256": compose_sha256,
        "services": sorted(services),
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "check_count": len(checks),
        "compose_validation": compose_validation,
        "runtime_started": runtime_started,
        "runtime_healthchecks_completed": runtime_healthy,
        "runtime_evidence_source": (
            runtime_evidence_source
            or ("current_runtime_drill" if runtime else "not_collected")
        ),
        "runtime_observations": runtime,
        "postgres_restore_drill_completed": bool(
            runtime.get("postgres_restore_drill_completed", False)
        ),
        "n8n_workflow_import_completed": bool(
            runtime.get("n8n_workflow_import_completed", False)
        ),
        "mailhog_delivery_receipt_completed": bool(
            runtime.get("mailhog_delivery_receipt_completed", False)
        ),
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
    path = Path(output_path)
    if kwargs.get("runtime_observations") is None:
        preserved = _load_matching_runtime_observations(path, kwargs)
        if preserved is not None:
            kwargs["runtime_observations"] = preserved
            kwargs["runtime_evidence_source"] = "preserved_matching_compose"
    payload = build_disposable_synthetic_staging_readiness(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def collect_disposable_synthetic_runtime_observations(
    *,
    root: str | Path = ROOT,
    compose_path: str | Path = DEFAULT_COMPOSE_PATH,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    compose_file = _resolve(repo, compose_path)
    docker = shutil.which("docker")
    if not docker:
        return {
            "runtime_started": False,
            "runtime_healthchecks_completed": False,
            "reason": "docker CLI unavailable",
        }
    command = [docker, "compose", "-f", str(compose_file)]
    service_result = _run(
        [*command, "ps", "--services", "--filter", "status=running"],
        cwd=repo,
        timeout=45,
    )
    running_services = {
        row.strip()
        for row in service_result.get("stdout", "").splitlines()
        if row.strip()
    }
    runtime_started = REQUIRED_SERVICES <= running_services
    probes = {
        "backend_health": _http_probe("http://127.0.0.1:8017/health"),
        "backend_dependency_import": _backend_dependency_import_probe(
            command,
            repo,
        ),
        "frontend_http": _http_probe("http://127.0.0.1:5173/"),
        "n8n_health": _http_probe("http://127.0.0.1:5678/healthz"),
        "mailhog_api": _http_probe("http://127.0.0.1:8025/api/v2/messages"),
        "postgres_tcp": _tcp_probe("127.0.0.1", 55432),
        "redis_tcp": _tcp_probe("127.0.0.1", 56379),
    }
    healthchecks_completed = runtime_started and all(probes.values())
    postgres_restore = (
        _postgres_restore_drill(command, repo)
        if healthchecks_completed
        else {"completed": False, "reason": "runtime health incomplete"}
    )
    n8n_import = (
        _n8n_import_drill(command, repo)
        if healthchecks_completed
        else {"completed": False, "reason": "runtime health incomplete"}
    )
    mailhog_receipt = (
        _mailhog_receipt_drill()
        if healthchecks_completed
        else {"completed": False, "reason": "runtime health incomplete"}
    )
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "loopback_disposable_synthetic_runtime",
        "runtime_started": runtime_started,
        "runtime_healthchecks_completed": healthchecks_completed,
        "running_services": sorted(running_services),
        "service_listing": service_result,
        "health_probes": probes,
        "postgres_restore_drill_completed": bool(
            postgres_restore.get("completed")
        ),
        "postgres_restore_drill": postgres_restore,
        "n8n_workflow_import_completed": bool(n8n_import.get("completed")),
        "n8n_workflow_import": n8n_import,
        "mailhog_delivery_receipt_completed": bool(
            mailhog_receipt.get("completed")
        ),
        "mailhog_delivery_receipt": mailhog_receipt,
        "real_external_delivery_completed": False,
        "patient_data_processed": False,
    }


def _load_matching_runtime_observations(
    output_path: Path,
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Retain a fresh runtime drill when a static ship refresh uses the same compose file."""

    if not output_path.exists():
        return None
    try:
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        observations = previous.get("runtime_observations") or {}
        observed_at = datetime.fromisoformat(
            str(observations.get("observed_at") or "")
        )
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - observed_at).total_seconds()
        repo = Path(kwargs.get("root", ROOT)).resolve()
        compose_file = _resolve(
            repo,
            kwargs.get("compose_path", DEFAULT_COMPOSE_PATH),
        )
        current_sha256 = hashlib.sha256(compose_file.read_bytes()).hexdigest()
        if (
            previous.get("compose_sha256") == current_sha256
            and bool(observations.get("runtime_started"))
            and bool(observations.get("runtime_healthchecks_completed"))
            and 0 <= age_seconds <= 86_400
        ):
            return dict(observations)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None


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


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "executed": True,
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "executed": True,
        "exit_code": result.returncode,
        "stdout": result.stdout[-1_000:],
        "stderr": result.stderr[-1_000:],
    }


def _http_probe(url: str, attempts: int = 4) -> bool:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:  # noqa: BLE001
            if attempt + 1 < attempts:
                time.sleep(1)
    return False


def _tcp_probe(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def _backend_dependency_import_probe(
    compose_command: list[str],
    repo: Path,
) -> bool:
    result = _run(
        [
            *compose_command,
            "exec",
            "-T",
            "backend",
            "/usr/bin/python3",
            "-c",
            "import fastapi, sqlalchemy; print('runtime-imports-ok')",
        ],
        cwd=repo,
        timeout=30,
    )
    return (
        result.get("exit_code") == 0
        and "runtime-imports-ok" in result.get("stdout", "")
    )


def _postgres_restore_drill(
    compose_command: list[str],
    repo: Path,
) -> dict[str, Any]:
    script = (
        "set -eu; "
        "pg_dump -U nlcare_synthetic -d nlcare_synthetic -Fc "
        "-f /tmp/nlcare_restore.dump; "
        "dropdb -U nlcare_synthetic --if-exists nlcare_restore_drill; "
        "createdb -U nlcare_synthetic nlcare_restore_drill; "
        "pg_restore -U nlcare_synthetic -d nlcare_restore_drill "
        "/tmp/nlcare_restore.dump; "
        "psql -U nlcare_synthetic -d nlcare_restore_drill -tAc "
        "\"SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema='public'\"; "
        "dropdb -U nlcare_synthetic nlcare_restore_drill"
    )
    result = _run(
        [*compose_command, "exec", "-T", "postgres", "sh", "-lc", script],
        cwd=repo,
        timeout=120,
    )
    table_count = None
    for row in result.get("stdout", "").splitlines():
        if row.strip().isdigit():
            table_count = int(row.strip())
    return {
        "completed": result.get("exit_code") == 0 and (table_count or 0) > 0,
        "restored_public_table_count": table_count,
        "command_result": result,
    }


def _n8n_import_drill(
    compose_command: list[str],
    repo: Path,
) -> dict[str, Any]:
    result = _run(
        [
            *compose_command,
            "exec",
            "-T",
            "n8n",
            "n8n",
            "import:workflow",
            "--input=/workflows/synthetic_high_risk_review_alert.json",
        ],
        cwd=repo,
        # n8n CLI startup can take over two minutes on cold Windows Docker
        # Desktop volumes even when the already-running service is healthy.
        timeout=240,
    )
    return {
        "completed": result.get("exit_code") == 0,
        "workflow": "synthetic_high_risk_review_alert",
        "activated": False,
        "external_delivery_enabled": False,
        "command_result": result,
    }


def _mailhog_receipt_drill() -> dict[str, Any]:
    marker = f"nlcare-synthetic-{uuid.uuid4().hex}"
    message = (
        "From: synthetic-runtime@nlcare.invalid\r\n"
        "To: reviewer@nlcare.invalid\r\n"
        f"Subject: {marker}\r\n"
        "\r\n"
        "Synthetic staging receipt only. No patient data.\r\n"
    )
    try:
        with smtplib.SMTP("127.0.0.1", 1025, timeout=5) as smtp:
            smtp.sendmail(
                "synthetic-runtime@nlcare.invalid",
                ["reviewer@nlcare.invalid"],
                message,
            )
        time.sleep(0.5)
        with urllib.request.urlopen(
            "http://127.0.0.1:8025/api/v2/messages",
            timeout=5,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        messages = payload.get("items") or []
        found = any(
            marker
            in str(
                ((item.get("Content") or {}).get("Headers") or {}).get(
                    "Subject",
                    [],
                )
            )
            for item in messages
            if isinstance(item, dict)
        )
        return {
            "completed": found,
            "synthetic_marker": marker,
            "recipient_domain": "nlcare.invalid",
            "external_delivery": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "completed": False,
            "synthetic_marker": marker,
            "external_delivery": False,
            "reason": str(exc),
        }


def _check(check_id: str, passed: bool) -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed)}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


__all__ = [
    "build_disposable_synthetic_staging_readiness",
    "collect_disposable_synthetic_runtime_observations",
    "write_disposable_synthetic_staging_readiness",
]
