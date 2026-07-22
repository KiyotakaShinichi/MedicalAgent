"""Isolated local Postgres/Redis migration and recovery smoke."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "docker-compose.recovery-smoke.yml"
OUTPUT_PATH = ROOT / "Data" / "evals" / "ops" / "latest_container_recovery_smoke.json"
PROJECT = "nlcare-recovery-smoke"
DATABASE_URL = "postgresql+psycopg2://nlcare_smoke:nlcare_recovery_smoke_only_2026@127.0.0.1:55432/nlcare_smoke"
RESTORE_URL = "postgresql+psycopg2://nlcare_smoke:nlcare_recovery_smoke_only_2026@127.0.0.1:55432/nlcare_restore"


def _command(args: list[str], *, timeout: int = 180, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed").strip()[-1200:]
        raise RuntimeError(f"{' '.join(args[:4])} failed: {detail}")
    return completed.stdout.strip()


def _docker_available() -> tuple[bool, str | None]:
    try:
        return True, _command(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30)
    except Exception as exc:  # noqa: BLE001 - readiness artifact records environment block
        return False, str(exc)


def _wait_for_services() -> None:
    for _ in range(45):
        try:
            _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "exec", "-T", "postgres", "pg_isready", "-U", "nlcare_smoke", "-d", "nlcare_smoke"], timeout=15)
            pong = _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "exec", "-T", "redis", "redis-cli", "ping"], timeout=15)
            if pong.strip() == "PONG":
                return
        except Exception:  # noqa: BLE001 - bounded readiness retry
            time.sleep(2)
    raise RuntimeError("Postgres/Redis smoke services did not become ready")


def _database_probe(url: str, *, insert: bool) -> dict[str, Any]:
    from sqlalchemy import create_engine, text

    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as connection:
        if insert:
            connection.execute(text("CREATE TABLE IF NOT EXISTS nlcare_recovery_probe (probe_id INTEGER PRIMARY KEY, marker TEXT NOT NULL)"))
            connection.execute(text("DELETE FROM nlcare_recovery_probe"))
            connection.execute(text("INSERT INTO nlcare_recovery_probe (probe_id, marker) VALUES (1, 'synthetic-recovery-marker')"))
        migration = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        marker = connection.execute(text("SELECT marker FROM nlcare_recovery_probe WHERE probe_id=1")).scalar_one()
        table_count = connection.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")) .scalar_one()
    engine.dispose()
    return {"migration_version": str(migration), "marker": str(marker), "public_table_count": int(table_count)}


def _write(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_container_recovery_smoke(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    generated = datetime.now(timezone.utc).isoformat()
    available, docker_version = _docker_available()
    if not available:
        return _write({
            "schema_version": "container_recovery_smoke_v1",
            "generated_at": generated,
            "status": "blocked_environment",
            "completed": False,
            "docker_available": False,
            "block_reason": "Docker daemon unavailable; no migration or recovery claim is made.",
            "clinical_validation": False,
            "healthcare_production_ready": False,
            "claim_boundary": "Local infrastructure smoke only; not production or healthcare readiness evidence.",
        }, output_path)

    checks: dict[str, Any] = {}
    error: str | None = None
    try:
        _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "up", "-d"], timeout=300)
        _wait_for_services()
        migration_env = {**os.environ, "DATABASE_URL": DATABASE_URL, "ENVIRONMENT": "development"}
        _command(["python", "-m", "alembic", "upgrade", "head"], timeout=240, env=migration_env)
        checks["source_database"] = _database_probe(DATABASE_URL, insert=True)

        _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "exec", "-T", "postgres", "pg_dump", "-U", "nlcare_smoke", "-d", "nlcare_smoke", "-Fc", "-f", "/tmp/nlcare-smoke.dump"], timeout=180)
        _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "exec", "-T", "postgres", "createdb", "-U", "nlcare_smoke", "nlcare_restore"], timeout=60)
        _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "exec", "-T", "postgres", "pg_restore", "-U", "nlcare_smoke", "-d", "nlcare_restore", "/tmp/nlcare-smoke.dump"], timeout=180)
        checks["restored_database"] = _database_probe(RESTORE_URL, insert=False)
        checks["postgres_restore_match"] = checks["source_database"] == checks["restored_database"]

        from redis import Redis

        client = Redis.from_url("redis://127.0.0.1:56379/0", decode_responses=True, socket_timeout=5)
        client.set("nlcare:recovery:probe", "synthetic-redis-marker")
        checks["redis_before_restart"] = client.get("nlcare:recovery:probe")
        _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "restart", "redis"], timeout=120)
        _wait_for_services()
        checks["redis_after_restart"] = client.get("nlcare:recovery:probe")
        checks["redis_persistence_match"] = checks["redis_before_restart"] == checks["redis_after_restart"] == "synthetic-redis-marker"
        completed = bool(checks["postgres_restore_match"] and checks["redis_persistence_match"])
    except Exception as exc:  # noqa: BLE001 - artifact captures a bounded local smoke failure
        completed = False
        error = str(exc)[-1600:]
    finally:
        try:
            _command(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT, "down", "-v", "--remove-orphans"], timeout=180)
            checks["isolated_resources_removed"] = True
        except Exception as cleanup_exc:  # noqa: BLE001
            checks["isolated_resources_removed"] = False
            error = error or str(cleanup_exc)[-1600:]

    return _write({
        "schema_version": "container_recovery_smoke_v1",
        "generated_at": generated,
        "status": "acceptable_local_container_smoke" if completed else "needs_attention",
        "completed": completed,
        "docker_available": True,
        "docker_server_version": docker_version,
        "compose_file": COMPOSE_FILE.relative_to(ROOT).as_posix(),
        "compose_project": PROJECT,
        "checks": checks,
        "error": error,
        "contains_real_patient_data": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "production_traffic_tested": False,
        "claim_boundary": "Disposable local Postgres/Redis migration and recovery smoke only; not production, compliance, or healthcare readiness evidence.",
    }, output_path)


__all__ = ["run_container_recovery_smoke"]
