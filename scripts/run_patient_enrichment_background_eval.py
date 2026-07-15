from __future__ import annotations

import inspect
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.routers import patient as patient_router
from backend.services import patient_report_enrichment_jobs as jobs


OUTPUT = ROOT / "Data/evals/ops/latest_patient_enrichment_background_eval.json"


class _Session:
    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _wait_for(patient_id: str, status: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = jobs.get_patient_enrichment_job(patient_id)
        if snapshot and snapshot["status"] == status:
            return snapshot
        time.sleep(0.01)
    return None


def main() -> int:
    original_session_local = jobs.SessionLocal
    jobs.SessionLocal = _Session
    jobs.reset_patient_enrichment_jobs_for_tests()
    try:
        release = threading.Event()
        started = threading.Event()
        calls = []

        def build(patient_id, db):
            calls.append(patient_id)
            started.set()
            release.wait(1.0)
            return {"patient_id": patient_id}

        first = jobs.schedule_patient_enrichment("eval-single-flight", build=build)
        started.wait(1.0)
        second = jobs.schedule_patient_enrichment("eval-single-flight", build=build)
        release.set()
        completed = _wait_for("eval-single-flight", "complete")
        single_flight = len(calls) == 1 and first["status"] in {"queued", "running"} and second["status"] == "running"

        stale_started = threading.Event()
        stale_release = threading.Event()
        discarded = []

        def stale_build(patient_id, db):
            stale_started.set()
            stale_release.wait(1.0)
            return {"patient_id": patient_id, "stale": True}

        jobs.schedule_patient_enrichment(
            "eval-stale",
            build=stale_build,
            discard_stale_result=lambda patient_id, result: discarded.append((patient_id, result)),
        )
        stale_started.wait(1.0)
        jobs.invalidate_patient_enrichment("eval-stale")
        stale_release.set()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not discarded:
            time.sleep(0.01)
        stale_result_discarded = bool(discarded) and jobs.get_patient_enrichment_job("eval-stale") is None

        def failed_build(patient_id, db):
            raise RuntimeError("private evaluation detail")

        jobs.schedule_patient_enrichment("eval-failure", build=failed_build)
        failed = _wait_for("eval-failure", "failed") or {}
        failure_redacted = failed.get("error_code") == "RuntimeError" and "private evaluation detail" not in str(failed)
        request_path_nonblocking_contract = "build_patient_report_response" not in inspect.getsource(
            patient_router.get_my_patient_report_enrichment
        )
        checks = {
            "single_flight": single_flight,
            "completion_recorded": bool(completed and completed.get("generated_ms") is not None),
            "stale_result_discarded": stale_result_discarded,
            "failure_redacted": failure_redacted,
            "request_path_nonblocking_contract": request_path_nonblocking_contract,
        }
    finally:
        jobs.SessionLocal = original_session_local
        jobs.reset_patient_enrichment_jobs_for_tests()

    payload = {
        "schema_version": "patient_enrichment_background_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if all(checks.values()) else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "process_local_worker": True,
        "distributed_queue_proven": False,
        "checks": checks,
        "claim_boundary": (
            "This is a local engineering concurrency regression. It does not establish distributed-worker "
            "durability, production latency, clinical validation, or healthcare production readiness."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2))
    return 0 if payload["status"] == "strong" else 1


if __name__ == "__main__":
    raise SystemExit(main())
