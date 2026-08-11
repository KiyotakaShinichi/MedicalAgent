"""Leased worker for tenant-scoped synthetic evaluation jobs."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from backend.models import SaaSPlatformJob, SaaSProject
from backend.services.saas_control_plane import (
    SaaSActor,
    append_audit_event,
    append_outbox_event,
    serialize_job,
)


ROOT = Path(__file__).resolve().parents[2]
JOB_COMMANDS: dict[str, tuple[str, ...]] = {
    "rag_baseline_comparison": ("python", "scripts/run_rag_baseline_comparison.py"),
    "adversarial_safety_eval": ("python", "scripts/run_adversarial_safety_regression.py"),
    "agent_workflow_eval": ("python", "scripts/run_agentic_workflow_eval.py"),
    "release_gate": ("python", "scripts/run_release_gate.py"),
    "evidence_packet_export": ("python", "scripts/generate_benchmark_report.py"),
}
CLAIM_BOUNDARY = (
    "This worker executes allowlisted synthetic engineering evaluations. It cannot "
    "perform clinical actions, process real patient data, or establish healthcare production readiness."
)


def recover_expired_platform_jobs(db: Any, *, now: datetime | None = None) -> int:
    current = now or datetime.now(timezone.utc)
    rows = (
        db.query(SaaSPlatformJob)
        .filter(
            SaaSPlatformJob.status == "running",
            SaaSPlatformJob.lease_expires_at.is_not(None),
            SaaSPlatformJob.lease_expires_at < current,
        )
        .all()
    )
    for row in rows:
        row.status = "queued"
        row.available_at = current
        row.lease_owner = None
        row.lease_token = None
        row.lease_expires_at = None
        row.recovery_count = int(row.recovery_count or 0) + 1
        row.error_message = "Expired worker lease recovered."
    db.commit()
    return len(rows)


def claim_next_platform_job(
    db: Any,
    *,
    worker_id: str,
    lease_seconds: int = 300,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    current = now or datetime.now(timezone.utc)
    query = (
        db.query(SaaSPlatformJob)
        .filter(
            SaaSPlatformJob.status == "queued",
            (SaaSPlatformJob.available_at.is_(None) | (SaaSPlatformJob.available_at <= current)),
        )
        .order_by(SaaSPlatformJob.queued_at.asc(), SaaSPlatformJob.id.asc())
    )
    try:
        row = query.with_for_update(skip_locked=True).first()
    except Exception:
        row = query.first()
    if row is None:
        return None
    row.status = "running"
    row.started_at = row.started_at or current
    row.attempts = int(row.attempts or 0) + 1
    row.lease_owner = str(worker_id)
    row.lease_token = f"lease_{uuid4().hex}"
    row.lease_expires_at = current + timedelta(seconds=max(30, min(lease_seconds, 1_800)))
    db.commit()
    db.refresh(row)
    return serialize_job(row) | {"lease_token": row.lease_token}


def heartbeat_platform_job(
    db: Any,
    *,
    job_id: str,
    lease_token: str,
    lease_seconds: int = 300,
    now: datetime | None = None,
) -> bool:
    row = (
        db.query(SaaSPlatformJob)
        .filter(
            SaaSPlatformJob.id == job_id,
            SaaSPlatformJob.status == "running",
            SaaSPlatformJob.lease_token == lease_token,
        )
        .first()
    )
    if row is None:
        return False
    row.lease_expires_at = (now or datetime.now(timezone.utc)) + timedelta(
        seconds=max(30, min(lease_seconds, 1_800))
    )
    db.commit()
    return True


def execute_platform_job(
    db: Any,
    *,
    job_id: str,
    lease_token: str,
    environment: Mapping[str, str] | None = None,
    command_runner: Callable[[Sequence[str], Path, Mapping[str, str], int], Mapping[str, Any]] | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    row = (
        db.query(SaaSPlatformJob)
        .filter(
            SaaSPlatformJob.id == job_id,
            SaaSPlatformJob.status == "running",
            SaaSPlatformJob.lease_token == lease_token,
        )
        .first()
    )
    if row is None:
        raise PermissionError("Job lease is missing, expired, or owned by another worker.")
    scoped_project = (
        db.query(SaaSProject)
        .filter(
            SaaSProject.id == row.project_id,
            SaaSProject.organization_id == row.organization_id,
            SaaSProject.status == "active",
        )
        .first()
    )
    if scoped_project is None:
        return _fail_job(
            db,
            row,
            PermissionError("Worker rejected a job with inconsistent tenant scope."),
        )
    payload = json.loads(row.payload_json or "{}")
    command = JOB_COMMANDS.get(row.job_type)
    if command is None:
        return _fail_job(db, row, ValueError("Job type has no allowlisted worker command."))
    dry_run = bool(payload.get("dry_run", True))
    try:
        if dry_run:
            result = {
                "status": "dry_run_completed",
                "command_preview": list(command),
                "commands_executed": False,
                "clinical_validation": False,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        else:
            values = dict(os.environ if environment is None else environment)
            if not _truthy(values.get("NLCARE_SAAS_JOB_EXECUTION_ENABLED")):
                raise PermissionError("SaaS job execution is disabled; the worker failed closed.")
            runner = command_runner or _run_command
            result = dict(runner(command, ROOT, values, max(30, min(timeout_seconds, 1_800))))
            result.update({
                "commands_executed": True,
                "clinical_validation": False,
                "claim_boundary": CLAIM_BOUNDARY,
            })
    except Exception as exc:
        return _fail_job(db, row, exc)

    current = datetime.now(timezone.utc)
    row.status = "completed"
    row.progress_percent = 100
    row.result_json = json.dumps(result, sort_keys=True, default=str)
    row.error_message = None
    row.finished_at = current
    _clear_lease(row)
    append_outbox_event(
        db,
        organization_id=row.organization_id,
        project_id=row.project_id,
        aggregate_type="platform_job",
        aggregate_id=row.id,
        event_type="evaluation.job.completed",
        payload={"job_id": row.id, "job_type": row.job_type, "status": "completed"},
        idempotency_key=f"job.completed:{row.id}",
    )
    append_audit_event(
        db,
        organization_id=row.organization_id,
        project_id=row.project_id,
        actor=SaaSActor(subject="system:saas-worker", application_role="system", auth_source="worker"),
        action="evaluation_job_completed",
        target_type="platform_job",
        target_id=row.id,
        details={"job_type": row.job_type, "attempts": int(row.attempts or 0)},
    )
    db.commit()
    db.refresh(row)
    return serialize_job(row) | {"result": result}


def run_platform_job_once(db: Any, *, worker_id: str) -> dict[str, Any] | None:
    recover_expired_platform_jobs(db)
    # Allowlisted commands time out after 15 minutes, so a 30-minute lease
    # prevents duplicate recovery while a valid command is still running.
    claimed = claim_next_platform_job(db, worker_id=worker_id, lease_seconds=1_800)
    if claimed is None:
        return None
    return execute_platform_job(
        db,
        job_id=str(claimed["id"]),
        lease_token=str(claimed["lease_token"]),
    )


def _fail_job(db: Any, row: SaaSPlatformJob, exc: Exception) -> dict[str, Any]:
    current = datetime.now(timezone.utc)
    exhausted = int(row.attempts or 0) >= int(row.max_attempts or 3)
    row.status = "dead_lettered" if exhausted else "queued"
    row.error_message = str(exc)[:2_000]
    row.available_at = None if exhausted else current + timedelta(seconds=min(300, 5 * (2 ** int(row.attempts or 1))))
    if exhausted:
        row.finished_at = current
    _clear_lease(row)
    append_outbox_event(
        db,
        organization_id=row.organization_id,
        project_id=row.project_id,
        aggregate_type="platform_job",
        aggregate_id=row.id,
        event_type="evaluation.job.dead_lettered" if exhausted else "evaluation.job.retry_scheduled",
        payload={"job_id": row.id, "job_type": row.job_type, "status": row.status, "attempt": int(row.attempts or 0)},
        idempotency_key=f"job.{row.status}:{row.id}:{int(row.attempts or 0)}",
    )
    db.commit()
    db.refresh(row)
    return serialize_job(row)


def _clear_lease(row: SaaSPlatformJob) -> None:
    row.lease_owner = None
    row.lease_token = None
    row.lease_expires_at = None


def _run_command(command: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: int) -> dict[str, Any]:
    executable = os.environ.get("PYTHON", os.sys.executable) if command[0] == "python" else command[0]
    completed = subprocess.run(
        [executable, *command[1:]],
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        check=False,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip())[-4_000:]
    if completed.returncode != 0:
        raise RuntimeError(f"Allowlisted evaluation failed with exit_code={completed.returncode}: {output}")
    return {"status": "completed", "exit_code": completed.returncode, "output_preview": output}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "CLAIM_BOUNDARY",
    "JOB_COMMANDS",
    "claim_next_platform_job",
    "execute_platform_job",
    "heartbeat_platform_job",
    "recover_expired_platform_jobs",
    "run_platform_job_once",
]
