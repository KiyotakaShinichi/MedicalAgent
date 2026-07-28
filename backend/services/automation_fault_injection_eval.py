"""Executable fault-injection checks for redacted engineering automation.

The scenarios use disposable SQLite state and in-process webhook transports.
They exercise durability contracts without sending messages or automating any
clinical action.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import AsyncTask
from backend.services.automation_job_queue import (
    enqueue_automation_task,
    requeue_automation_task,
)
from backend.services.automation_worker import (
    claim_next_automation_task,
    record_automation_delivery_receipt,
    recover_expired_automation_leases,
)
from backend.services.background_eval_worker import execute_job
from backend.services.n8n_webhook_dispatcher import (
    build_signed_dispatch,
    validate_signed_dispatch_envelope,
    validate_signed_dispatch_envelope_with_keyring,
)


DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_automation_fault_injection.json")
CLAIM_BOUNDARY = (
    "Disposable engineering fault injection only. A passing result does not prove external channel "
    "reliability, clinician acknowledgement, emergency coverage, patient benefit, or healthcare production readiness."
)


def _sessions(path: Path):
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _scenario(name: str, function: Callable[[], tuple[bool, dict[str, Any]]]) -> dict[str, Any]:
    try:
        passed, evidence = function()
        return {"id": name, "passed": bool(passed), "evidence": evidence, "error": None}
    except Exception as exc:  # every injected failure must become reviewable evidence
        return {
            "id": name,
            "passed": False,
            "evidence": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _queue_scenarios(database_path: Path) -> list[dict[str, Any]]:
    sessions = _sessions(database_path)

    def reset(db) -> None:
        db.query(AsyncTask).delete()
        db.commit()

    def duplicate_enqueue() -> tuple[bool, dict[str, Any]]:
        db = sessions()
        try:
            reset(db)
            first = enqueue_automation_task(
                db, job_type="refresh_trace_envelope_v2_eval", requested_by="fault-test",
                dry_run=True, idempotency_key="repeatable-operation",
            )
            second = enqueue_automation_task(
                db, job_type="refresh_trace_envelope_v2_eval", requested_by="fault-test",
                dry_run=True, idempotency_key="repeatable-operation",
            )
            count = db.query(AsyncTask).count()
            return first["id"] == second["id"] and count == 1, {
                "first_task_id": first["id"], "second_task_id": second["id"], "row_count": count,
            }
        finally:
            db.close()

    def lease_contention() -> tuple[bool, dict[str, Any]]:
        db = sessions()
        try:
            reset(db)
            enqueue_automation_task(
                db, job_type="refresh_trace_envelope_v2_eval", requested_by="fault-test", dry_run=True,
            )
            first = claim_next_automation_task(db, worker_id="worker-a")
            second = claim_next_automation_task(db, worker_id="worker-b")
            return first is not None and second is None, {
                "first_claimed": first is not None, "second_claimed": second is not None,
            }
        finally:
            db.close()

    def stale_lease_and_token() -> tuple[bool, dict[str, Any]]:
        db = sessions()
        try:
            reset(db)
            task = enqueue_automation_task(
                db, job_type="refresh_trace_envelope_v2_eval", requested_by="fault-test", dry_run=True,
            )
            claimed = claim_next_automation_task(db, worker_id="crashed-worker")
            row = db.query(AsyncTask).filter(AsyncTask.id == task["id"]).one()
            old_token = str(claimed["lease_token"])
            row.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            db.commit()
            recovery = recover_expired_automation_leases(db)
            reclaimed = claim_next_automation_task(db, worker_id="replacement-worker")
            return (
                recovery["recovered"] == 1
                and reclaimed is not None
                and reclaimed["lease_token"] != old_token
                and reclaimed["lease_owner"] == "replacement-worker"
            ), {
                "recovery": recovery,
                "old_token_reused": bool(reclaimed and reclaimed["lease_token"] == old_token),
                "replacement_claimed": reclaimed is not None,
            }
        finally:
            db.close()

    def dead_letter_and_manual_requeue() -> tuple[bool, dict[str, Any]]:
        db = sessions()
        try:
            reset(db)
            task = enqueue_automation_task(
                db, job_type="refresh_trace_envelope_v2_eval", requested_by="fault-test", dry_run=True,
            )
            row = db.query(AsyncTask).filter(AsyncTask.id == task["id"]).one()
            row.status = "running"
            row.attempts = 3
            row.lease_owner = "failed-worker"
            row.lease_token = "failed-token"
            row.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            db.commit()
            recovery = recover_expired_automation_leases(db)
            db.refresh(row)
            dead_lettered = row.status == "dead_lettered"
            requeued = requeue_automation_task(db, row.id, requested_by="fault-operator")
            history = requeued.get("requeue_history") or []
            return dead_lettered and requeued["status"] == "queued" and len(history) == 1, {
                "recovery": recovery,
                "dead_lettered_before_requeue": dead_lettered,
                "requeue_history": history,
            }
        finally:
            db.close()

    def receipt_conflict() -> tuple[bool, dict[str, Any]]:
        db = sessions()
        try:
            reset(db)
            task = enqueue_automation_task(
                db, job_type="publish_trace_quality_digest", requested_by="fault-test", dry_run=True,
            )
            row = db.query(AsyncTask).filter(AsyncTask.id == task["id"]).one()
            row.delivery_event_id = "stable-event"
            db.commit()
            first = record_automation_delivery_receipt(
                db, event_id="stable-event", receipt_id="receipt-1", delivery_status="delivered",
                occurred_at=datetime.now(timezone.utc),
            )
            duplicate = record_automation_delivery_receipt(
                db, event_id="stable-event", receipt_id="receipt-1", delivery_status="delivered",
                occurred_at=datetime.now(timezone.utc),
            )
            conflict_rejected = False
            try:
                record_automation_delivery_receipt(
                    db, event_id="stable-event", receipt_id="receipt-2", delivery_status="failed",
                    occurred_at=datetime.now(timezone.utc),
                )
            except ValueError:
                conflict_rejected = True
            return first.id == duplicate.id and conflict_rejected, {
                "same_receipt_idempotent": first.id == duplicate.id,
                "conflicting_receipt_rejected": conflict_rejected,
                "human_acknowledgement_inferred": False,
            }
        finally:
            db.close()

    results = [
        _scenario("duplicate_enqueue_idempotency", duplicate_enqueue),
        _scenario("database_lease_contention", lease_contention),
        _scenario("stale_lease_recovery_rotates_token", stale_lease_and_token),
        _scenario("bounded_attempts_dead_letter_and_audited_requeue", dead_letter_and_manual_requeue),
        _scenario("delivery_receipt_idempotency_and_conflict", receipt_conflict),
    ]
    sessions.kw["bind"].dispose()
    return results


def _webhook_scenarios() -> list[dict[str, Any]]:
    secret_old = "old-secret-for-fault-injection-0001"
    secret_new = "new-secret-for-fault-injection-0002"
    now = datetime.now(timezone.utc)

    def crash_after_side_effect() -> tuple[bool, dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def transport(url: str, body: str, headers: dict[str, str], timeout: float):
            calls.append({"url": url, "body": body, "headers": dict(headers), "timeout": timeout})
            return {"status_code": 202}

        job = {
            "accepted": True,
            "job_id": "durable-job-42",
            "job_type": "publish_trace_quality_digest",
            "sanitized_payload": {"status": "needs_attention"},
            "dry_run": False,
        }
        env = {
            "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
            "N8N_WEBHOOK_BASE_URL": "http://127.0.0.1:5678/webhook",
            "N8N_WEBHOOK_SIGNING_SECRET": secret_new,
            "N8N_WEBHOOK_SIGNING_KEY_ID": "new",
        }
        # The same durable job is executed twice to simulate a crash after the
        # remote side effect but before the local completion commit.
        from backend.services import n8n_webhook_dispatcher as dispatcher

        original = dispatcher._urllib_transport
        dispatcher._urllib_transport = transport
        try:
            first = execute_job(job, env=env)
            second = execute_job(job, env=env)
        finally:
            dispatcher._urllib_transport = original
        ids = [first.get("event_id"), second.get("event_id")]
        receiver_seen: set[str] = set()
        accepted = validate_signed_dispatch_envelope(
            body=calls[0]["body"], signature=calls[0]["headers"]["X-NLCare-Signature"],
            secret=secret_new, now=now, max_age_seconds=600, seen_event_ids=receiver_seen,
        )
        replay = validate_signed_dispatch_envelope(
            body=calls[1]["body"], signature=calls[1]["headers"]["X-NLCare-Signature"],
            secret=secret_new, now=now, max_age_seconds=600, seen_event_ids=receiver_seen,
        )
        return ids == ["durable-job-42", "durable-job-42"] and accepted["valid"] and replay["reason"] == "replay", {
            "event_ids": ids,
            "first_receiver_result": accepted["reason"],
            "second_receiver_result": replay["reason"],
        }

    def signature_rotation_and_tamper() -> tuple[bool, dict[str, Any]]:
        signed = build_signed_dispatch(
            workflow_id="trace_quality_digest", payload={"status": "ok"}, secret=secret_old,
            event_id="rotation-event", key_id="old", timestamp=now.isoformat(),
        )
        accepted = validate_signed_dispatch_envelope_with_keyring(
            body=signed["body"], signature=signed["headers"]["X-NLCare-Signature"], key_id="old",
            secrets={"old": secret_old, "new": secret_new}, now=now,
        )
        unknown = validate_signed_dispatch_envelope_with_keyring(
            body=signed["body"], signature=signed["headers"]["X-NLCare-Signature"], key_id="retired",
            secrets={"new": secret_new}, now=now,
        )
        tampered = validate_signed_dispatch_envelope_with_keyring(
            body=signed["body"] + " ", signature=signed["headers"]["X-NLCare-Signature"], key_id="old",
            secrets={"old": secret_old}, now=now,
        )
        return accepted["valid"] and not unknown["valid"] and not tampered["valid"], {
            "active_old_key_accepted": accepted["valid"],
            "retired_key_rejected": unknown["reason"],
            "tampered_body_rejected": tampered["reason"],
        }

    def stale_event() -> tuple[bool, dict[str, Any]]:
        old = now - timedelta(minutes=20)
        signed = build_signed_dispatch(
            workflow_id="trace_quality_digest", payload={"status": "old"}, secret=secret_new,
            event_id="stale-event", timestamp=old.isoformat(),
        )
        result = validate_signed_dispatch_envelope(
            body=signed["body"], signature=signed["headers"]["X-NLCare-Signature"],
            secret=secret_new, now=now, max_age_seconds=300,
        )
        return not result["valid"] and result["reason"] == "expired", {"receiver_result": result["reason"]}

    return [
        _scenario("crash_after_side_effect_uses_stable_event_id", crash_after_side_effect),
        _scenario("signing_key_rotation_and_tamper_rejection", signature_rotation_and_tamper),
        _scenario("delayed_stale_event_rejected", stale_event),
    ]


def build_automation_fault_injection_eval(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="nlcare-automation-fault-") as directory:
        scenarios = _queue_scenarios(Path(directory) / "faults.db") + _webhook_scenarios()
    passed = sum(bool(row["passed"]) for row in scenarios)
    report = {
        "schema_version": "automation_fault_injection_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if passed == len(scenarios) else "needs_attention",
        "scenario_count": len(scenarios),
        "passed_count": passed,
        "pass_rate": round(passed / len(scenarios), 6) if scenarios else 0.0,
        "scenarios": scenarios,
        "external_delivery_performed": False,
        "human_acknowledgement_proven": False,
        "clinical_action_automated": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["build_automation_fault_injection_eval"]
