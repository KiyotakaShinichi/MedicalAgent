"""Executable recovery evidence for the disposable synthetic staging stack."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "docker-compose.synthetic-staging.yml"
DEFAULT_OUTPUT_PATH = (
    ROOT / "Data/evals/ops/latest_synthetic_staging_runtime_recovery.json"
)
CLAIM_BOUNDARY = (
    "This drill uses the loopback disposable stack, synthetic engineering jobs, "
    "and a temporary PostgreSQL restore database. It does not prove managed-cloud "
    "failover, external delivery, clinician acknowledgement, clinical validation, "
    "or production healthcare readiness."
)


def run_runtime_recovery_drill(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    executor: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    started = time.perf_counter()
    drill_id = f"runtime-recovery-{uuid4().hex[:12]}"
    restore_db = f"nlcare_restore_{uuid4().hex[:10]}"
    worker_restarted = False
    cleanup: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    worker_evidence: dict[str, Any] = {}
    postgres_evidence: dict[str, Any] = {}
    error: str | None = None
    try:
        _compose(executor, "stop", "worker", timeout=60)
        prepared = _inside_json(
            executor,
            "prepare",
            drill_id,
            timeout=60,
        )
        task_id = int(prepared["task_id"])
        checks.append(_check("worker_lease_simulated_expired", prepared.get("lease_expired") is True))

        _compose(executor, "start", "worker", timeout=60)
        worker_restarted = True
        completed = _poll_task(
            executor,
            task_id=task_id,
            timeout_seconds=timeout_seconds,
        )
        replay = _inside_json(
            executor,
            "replay",
            drill_id,
            timeout=60,
        )
        worker_evidence = {
            "task_id": task_id,
            "terminal_status": completed.get("status"),
            "attempts": completed.get("attempts"),
            "recovery_count": completed.get("recovery_count"),
            "lease_cleared": not any(
                completed.get(key)
                for key in ("lease_owner", "lease_expires_at", "heartbeat_at")
            ),
            "idempotent_replay_reused_task": replay.get("idempotent_reuse") is True,
            "replay_task_id": replay.get("task_id"),
            "delivery_receipt_is_human_acknowledgement": False,
        }
        checks.extend(
            [
                _check("expired_lease_recovered", int(completed.get("recovery_count") or 0) >= 1),
                _check("recovered_job_completed", completed.get("status") == "completed"),
                _check("worker_lease_cleared", worker_evidence["lease_cleared"]),
                _check(
                    "idempotent_replay_suppressed_duplicate",
                    worker_evidence["idempotent_replay_reused_task"]
                    and int(replay.get("task_id") or -1) == task_id,
                ),
            ]
        )

        with tempfile.TemporaryDirectory(prefix="nlcare-pg-restore-") as tmp:
            directory = Path(tmp)
            archive = directory / "staging.dump"
            source_logical = _pg_dump(executor, database="nlcare_synthetic", logical=True)
            archive.write_bytes(
                _pg_dump(executor, database="nlcare_synthetic", logical=False)
            )
            source_table_count = _pg_table_count(executor, "nlcare_synthetic")
            _pg_exec(executor, ["dropdb", "--if-exists", "-U", "nlcare_synthetic", restore_db])
            _pg_exec(executor, ["createdb", "-U", "nlcare_synthetic", restore_db])
            _pg_restore(executor, archive, restore_db)
            restored_logical = _pg_dump(executor, database=restore_db, logical=True)
            restored_table_count = _pg_table_count(executor, restore_db)
            source_hash = _normalized_dump_sha256(source_logical)
            restored_hash = _normalized_dump_sha256(restored_logical)
            postgres_evidence = {
                "source_database": "nlcare_synthetic",
                "temporary_restore_database": restore_db,
                "source_table_count": source_table_count,
                "restored_table_count": restored_table_count,
                "normalized_logical_dump_sha256_source": source_hash,
                "normalized_logical_dump_sha256_restored": restored_hash,
                "normalized_content_match": source_hash == restored_hash,
                "managed_postgres": False,
                "patient_data_processed": False,
            }
            checks.extend(
                [
                    _check("postgres_restore_table_count_match", source_table_count == restored_table_count),
                    _check("postgres_restore_content_hash_match", source_hash == restored_hash),
                ]
            )
    except Exception as exc:  # noqa: BLE001 - evidence must record the failed drill
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            _pg_exec(executor, ["dropdb", "--if-exists", "-U", "nlcare_synthetic", restore_db])
            cleanup["temporary_restore_database_dropped"] = True
        except Exception as exc:  # noqa: BLE001
            cleanup["temporary_restore_database_dropped"] = False
            cleanup["restore_cleanup_error"] = type(exc).__name__
        try:
            cleanup.update(_inside_json(executor, "cleanup", drill_id, timeout=60))
        except Exception as exc:  # noqa: BLE001
            cleanup["task_cleanup_completed"] = False
            cleanup["task_cleanup_error"] = type(exc).__name__
        try:
            _compose(executor, "start", "worker", timeout=60)
            worker_restarted = True
        except Exception as exc:  # noqa: BLE001
            cleanup["worker_restart_error"] = type(exc).__name__

    passed_count = sum(item["passed"] for item in checks)
    passed = bool(checks) and passed_count == len(checks) and error is None and worker_restarted
    payload = {
        "schema_version": "synthetic_staging_runtime_recovery_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong_local_runtime_only" if passed else "needs_attention",
        "completed": passed,
        "scope": "loopback_disposable_synthetic_staging",
        "checks": checks,
        "passed_count": passed_count,
        "check_count": len(checks),
        "worker_recovery": worker_evidence,
        "postgres_restore": postgres_evidence,
        "cleanup": cleanup,
        "worker_running_after_drill": worker_restarted,
        "error": error,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "real_external_delivery_completed": False,
        "clinician_acknowledgement_proven": False,
        "managed_cloud_failover_proven": False,
        "patient_data_processed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def inside_task_phase(phase: str, drill_id: str) -> dict[str, Any]:
    from backend.database import SessionLocal
    from backend.models import AsyncTask
    from backend.services.automation_job_queue import enqueue_automation_task
    from backend.services.automation_worker import claim_next_automation_task

    db = SessionLocal()
    try:
        if phase == "prepare":
            queued = (
                db.query(AsyncTask)
                .filter(
                    AsyncTask.task_type.like("safe_automation:%"),
                    AsyncTask.status.in_(("queued", "failed", "running")),
                )
                .count()
            )
            if queued:
                raise RuntimeError(
                    f"automation queue is not isolated; {queued} active task(s) exist"
                )
            task = enqueue_automation_task(
                db,
                job_type="publish_trace_quality_digest",
                requested_by="runtime-recovery-drill",
                payload={
                    "run_id": drill_id,
                    "trace_coverage_rate": 1.0,
                    "missing_fields_count": 0,
                    "dashboard_path": "/admin/automation",
                },
                dry_run=True,
                idempotency_key=drill_id,
            )
            claimed = claim_next_automation_task(
                db,
                worker_id="simulated-crashed-worker",
                lease_seconds=5,
            )
            if not claimed or int(claimed["id"]) != int(task["id"]):
                raise RuntimeError("drill task was not claimed deterministically")
            row = db.query(AsyncTask).filter(AsyncTask.id == task["id"]).one()
            row.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            db.commit()
            return {
                "task_id": int(task["id"]),
                "lease_expired": True,
                "clinical_validation": False,
            }
        if phase == "status":
            row = _drill_row(db, drill_id)
            if row is None:
                raise LookupError("drill task not found")
            return {
                "task_id": int(row.id),
                "status": row.status,
                "attempts": int(row.attempts or 0),
                "recovery_count": int(row.recovery_count or 0),
                "lease_owner": row.lease_owner,
                "lease_expires_at": str(row.lease_expires_at) if row.lease_expires_at else None,
                "heartbeat_at": str(row.heartbeat_at) if row.heartbeat_at else None,
            }
        if phase == "replay":
            task = enqueue_automation_task(
                db,
                job_type="publish_trace_quality_digest",
                requested_by="runtime-recovery-drill",
                payload={"run_id": drill_id},
                dry_run=True,
                idempotency_key=drill_id,
            )
            return {
                "task_id": int(task["id"]),
                "idempotent_reuse": task.get("idempotent_reuse") is True,
            }
        if phase == "cleanup":
            row = _drill_row(db, drill_id)
            deleted = 0
            if row is not None:
                db.delete(row)
                db.commit()
                deleted = 1
            return {"task_cleanup_completed": True, "deleted_task_count": deleted}
        raise ValueError(f"unsupported inside phase: {phase}")
    finally:
        db.close()


def _drill_row(db: Any, drill_id: str) -> Any:
    from backend.models import AsyncTask

    rows = (
        db.query(AsyncTask)
        .filter(AsyncTask.task_type.like("safe_automation:%"))
        .order_by(AsyncTask.id.desc())
        .limit(100)
        .all()
    )
    for row in rows:
        try:
            stored = json.loads(row.payload_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        job = stored.get("job") or {}
        if job.get("idempotency_key_hash") == hashlib.sha256(
            drill_id.encode("utf-8")
        ).hexdigest():
            return row
    return None


def _poll_task(
    executor: Callable[..., subprocess.CompletedProcess[Any]],
    *,
    task_id: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _inside_json(executor, "status", str(task_id), timeout=60, status_by_id=True)
        if last.get("status") in {"completed", "dead_lettered"}:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"worker recovery did not finish; last={last}")


def _inside_json(
    executor: Callable[..., subprocess.CompletedProcess[Any]],
    phase: str,
    value: str,
    *,
    timeout: int,
    status_by_id: bool = False,
) -> dict[str, Any]:
    command = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "exec",
        "-T",
        "backend",
        "/usr/bin/python3",
        "scripts/run_synthetic_staging_runtime_drill.py",
        "--inside-phase",
        phase,
        "--drill-id",
        value,
    ]
    if status_by_id:
        command.append("--status-by-task-id")
    completed = executor(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "inside phase failed").strip())
    return json.loads((completed.stdout or "{}").strip())


def status_by_task_id(task_id: int) -> dict[str, Any]:
    from backend.database import SessionLocal
    from backend.models import AsyncTask

    db = SessionLocal()
    try:
        row = db.query(AsyncTask).filter(AsyncTask.id == task_id).one()
        return {
            "task_id": int(row.id),
            "status": row.status,
            "attempts": int(row.attempts or 0),
            "recovery_count": int(row.recovery_count or 0),
            "lease_owner": row.lease_owner,
            "lease_expires_at": str(row.lease_expires_at) if row.lease_expires_at else None,
            "heartbeat_at": str(row.heartbeat_at) if row.heartbeat_at else None,
        }
    finally:
        db.close()


def _compose(executor: Callable[..., subprocess.CompletedProcess[Any]], *args: str, timeout: int) -> None:
    completed = executor(
        ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "docker compose failed").strip())


def _pg_exec(executor: Callable[..., subprocess.CompletedProcess[Any]], command: list[str]) -> str:
    output = _pg_exec_bytes(executor, command)
    return output.decode("utf-8", errors="replace")


def _pg_exec_bytes(
    executor: Callable[..., subprocess.CompletedProcess[Any]],
    command: list[str],
) -> bytes:
    completed = executor(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "exec", "-T", "postgres", *command],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace") if isinstance(completed.stderr, bytes) else str(completed.stderr or "")
        raise RuntimeError(error.strip() or "postgres command failed")
    output = completed.stdout or b""
    return output if isinstance(output, bytes) else str(output).encode("utf-8")


def _pg_dump(executor: Callable[..., subprocess.CompletedProcess[Any]], *, database: str, logical: bool) -> bytes:
    args = ["pg_dump", "-U", "nlcare_synthetic", "-d", database]
    args.extend(["--data-only", "--column-inserts"] if logical else ["--format=custom"])
    return _pg_exec_bytes(executor, args)


def _pg_restore(executor: Callable[..., subprocess.CompletedProcess[Any]], archive: Path, database: str) -> None:
    container_path = f"/tmp/{archive.name}"
    lookup = executor(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "ps",
            "-q",
            "postgres",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    container_id = (lookup.stdout or "").strip()
    if lookup.returncode != 0 or not container_id:
        raise RuntimeError((lookup.stderr or "postgres container not found").strip())
    copy = executor(
        ["docker", "cp", str(archive), f"{container_id}:{container_path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if copy.returncode != 0:
        raise RuntimeError((copy.stderr or "docker cp failed").strip())
    try:
        _pg_exec(
            executor,
            ["pg_restore", "-U", "nlcare_synthetic", "-d", database, "--no-owner", container_path],
        )
    finally:
        _pg_exec(executor, ["rm", "-f", container_path])


def _pg_table_count(executor: Callable[..., subprocess.CompletedProcess[Any]], database: str) -> int:
    output = _pg_exec(
        executor,
        [
            "psql",
            "-U",
            "nlcare_synthetic",
            "-d",
            database,
            "-Atc",
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';",
        ],
    )
    return int(output.strip())


def _normalized_dump_sha256(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="replace")
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip()
        and not line.startswith("--")
        and not line.startswith("\\restrict")
        and not line.startswith("\\unrestrict")
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _check(check_id: str, passed: bool) -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed)}


__all__ = [
    "inside_task_phase",
    "run_runtime_recovery_drill",
    "status_by_task_id",
    "_normalized_dump_sha256",
]
