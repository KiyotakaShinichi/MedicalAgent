"""Single-flight background jobs for synthetic patient-report enrichment.

The patient record response remains the source of truth. This process-local
worker only precomputes synthetic engineering fields and never performs a
clinical action. A distributed task queue would replace it for a multi-worker
deployment; the current implementation is deliberately bounded for the demo.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session

from backend.database import SessionLocal


BuildEnrichment = Callable[[str, Session], dict[str, Any]]
DiscardResult = Callable[[str, dict[str, Any]], None]

_LOCK = threading.RLock()
_JOBS: dict[str, dict[str, Any]] = {}
_GENERATIONS: dict[str, int] = {}
_EXECUTOR: ThreadPoolExecutor | None = None

CLAIM_BOUNDARY = (
    "Background enrichment precomputes synthetic engineering details only. "
    "It is not clinical validation, a clinical prediction service, or a healthcare production worker."
)


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            workers = max(1, min(int(os.getenv("NLCARE_ENRICHMENT_WORKERS", "1")), 2))
            _EXECUTOR = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nlcare-enrichment")
        return _EXECUTOR


def schedule_patient_enrichment(
    patient_id: str,
    *,
    build: BuildEnrichment,
    discard_stale_result: DiscardResult | None = None,
) -> dict[str, Any]:
    """Queue one enrichment job per patient and return its current state."""
    now = datetime.now(timezone.utc).isoformat()
    with _LOCK:
        generation = _GENERATIONS.get(patient_id, 0)
        current = _JOBS.get(patient_id)
        if current and current.get("generation") == generation and current.get("status") in {"queued", "running"}:
            return _public_snapshot(current)
        job = {
            "patient_id": patient_id,
            "generation": generation,
            "status": "queued",
            "requested_at": now,
            "started_at": None,
            "completed_at": None,
            "generated_ms": None,
            "error_code": None,
        }
        _JOBS[patient_id] = job
        _executor().submit(_run_job, patient_id, generation, build, discard_stale_result)
        return _public_snapshot(job)


def get_patient_enrichment_job(patient_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(patient_id)
        return _public_snapshot(job) if job else None


def invalidate_patient_enrichment(patient_id: str | None = None) -> None:
    """Advance generation tokens so an in-flight stale result cannot be trusted."""
    with _LOCK:
        if patient_id is None:
            patient_ids = set(_GENERATIONS) | set(_JOBS)
            for item in patient_ids:
                _GENERATIONS[item] = _GENERATIONS.get(item, 0) + 1
            _JOBS.clear()
            return
        _GENERATIONS[patient_id] = _GENERATIONS.get(patient_id, 0) + 1
        _JOBS.pop(patient_id, None)


def reset_patient_enrichment_jobs_for_tests() -> None:
    with _LOCK:
        _JOBS.clear()
        _GENERATIONS.clear()


def _run_job(
    patient_id: str,
    generation: int,
    build: BuildEnrichment,
    discard_stale_result: DiscardResult | None,
) -> None:
    started = time.perf_counter()
    with _LOCK:
        job = _JOBS.get(patient_id)
        if job is None or job.get("generation") != generation:
            return
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()

    db = SessionLocal()
    result: dict[str, Any] | None = None
    try:
        result = build(patient_id, db)
        db.commit()
        with _LOCK:
            stale = _GENERATIONS.get(patient_id, 0) != generation
            job = _JOBS.get(patient_id)
            if stale or job is None or job.get("generation") != generation:
                if result is not None and discard_stale_result is not None:
                    discard_stale_result(patient_id, result)
                return
            job["status"] = "complete"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["generated_ms"] = round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:  # noqa: BLE001 - expose a bounded engineering status, not raw internals
        db.rollback()
        with _LOCK:
            job = _JOBS.get(patient_id)
            if job is not None and job.get("generation") == generation:
                job["status"] = "failed"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                job["generated_ms"] = round((time.perf_counter() - started) * 1000, 1)
                job["error_code"] = type(exc).__name__
    finally:
        db.close()


def _public_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": job.get("status"),
        "requested_at": job.get("requested_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "generated_ms": job.get("generated_ms"),
        "error_code": job.get("error_code"),
        "retry_after_ms": 750 if job.get("status") in {"queued", "running"} else None,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "get_patient_enrichment_job",
    "invalidate_patient_enrichment",
    "reset_patient_enrichment_jobs_for_tests",
    "schedule_patient_enrichment",
]
