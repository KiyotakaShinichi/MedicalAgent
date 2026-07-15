"""Generated contract test for patient-confirmed support-chat writes.

This evaluates the persistence boundary directly. It does not call an LLM and
does not measure natural-language understanding. The separate large-scale
prompt bank covers routing language; this artifact covers the database contract
that must hold after extraction: no write before confirmation, cancellation,
idempotency, patient isolation, duplicate prevention, and undo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import Patient, PatientRecordWriteAudit, SymptomReport
from backend.services.confirmed_record_write import (
    queue_record_write,
    resolve_pending_record_write,
    undo_record_write,
)


OUTPUT_PATH = Path("Data/evals/agentic_tool_use/latest_confirmed_write_contract_eval.json")
CONFIRMATIONS = ("yes", "yes please", "confirm", "confirm save", "save it", "go ahead", "oo", "opo")
CANCELLATIONS = ("no", "cancel", "cancel save", "don't save", "never mind", "huwag")


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _patient(db, patient_id: str) -> None:
    db.add(Patient(id=patient_id, name=f"Contract {patient_id}", diagnosis="Synthetic demo record"))
    db.commit()


def _payload(index: int) -> dict[str, Any]:
    return {
        "date": "2026-07-14",
        "symptom": f"contract symptom {index}",
        "severity": (index % 10) + 1,
    }


def _result(case_id: str, passed: bool, checks: dict[str, bool], note: str) -> dict[str, Any]:
    return {"case_id": case_id, "passed": bool(passed), "checks": checks, "note": note}


def build_report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for index, phrase in enumerate(CONFIRMATIONS):
        db = _session()
        patient_id = f"CONFIRM-{index:03d}"
        _patient(db, patient_id)
        queue_record_write(patient_id, "symptom", _payload(index), source_message="candidate", source_chat_message_id=index)
        before = db.query(SymptomReport).count() == 0
        actions = resolve_pending_record_write(db, patient_id, phrase) or []
        checks = {
            "no_write_before_confirmation": before,
            "one_write_after_confirmation": db.query(SymptomReport).count() == 1,
            "audit_created": db.query(PatientRecordWriteAudit).count() == 1,
            "undo_available": bool(actions and actions[0].get("undo_available")),
        }
        rows.append(_result(f"confirm_{index:03d}", all(checks.values()), checks, phrase))
        db.close()

    for index, phrase in enumerate(CANCELLATIONS):
        db = _session()
        patient_id = f"CANCEL-{index:03d}"
        _patient(db, patient_id)
        queue_record_write(patient_id, "symptom", _payload(100 + index), source_message="candidate", source_chat_message_id=index)
        actions = resolve_pending_record_write(db, patient_id, phrase) or []
        checks = {
            "no_record_created": db.query(SymptomReport).count() == 0,
            "no_audit_created": db.query(PatientRecordWriteAudit).count() == 0,
            "cancel_action_returned": bool(actions and actions[0].get("type") == "record_write_cancelled"),
        }
        rows.append(_result(f"cancel_{index:03d}", all(checks.values()), checks, phrase))
        db.close()

    for index in range(20):
        db = _session()
        patient_id = f"DUP-{index:03d}"
        _patient(db, patient_id)
        payload = _payload(200 + index)
        queue_record_write(patient_id, "symptom", payload, source_message="first", source_chat_message_id=index)
        resolve_pending_record_write(db, patient_id, "confirm")
        queue_record_write(patient_id, "symptom", payload, source_message="repeat", source_chat_message_id=index + 1000)
        actions = resolve_pending_record_write(db, patient_id, "confirm") or []
        checks = {
            "one_active_record": db.query(SymptomReport).count() == 1,
            "one_audit_envelope": db.query(PatientRecordWriteAudit).count() == 1,
            "duplicate_prevented": bool(actions and actions[0].get("type") == "duplicate_record_prevented"),
        }
        rows.append(_result(f"duplicate_{index:03d}", all(checks.values()), checks, "same payload twice"))
        db.close()

    for index in range(20):
        db = _session()
        patient_id = f"UNDO-{index:03d}"
        _patient(db, patient_id)
        queue_record_write(patient_id, "symptom", _payload(300 + index), source_message="candidate", source_chat_message_id=index)
        actions = resolve_pending_record_write(db, patient_id, "confirm") or []
        audit_id = int(actions[0]["audit_action_id"])
        undo = undo_record_write(db, patient_id, audit_id)
        audit = db.query(PatientRecordWriteAudit).one()
        checks = {
            "record_removed": db.query(SymptomReport).count() == 0,
            "audit_retained": audit.status == "undone" and audit.undone_at is not None,
            "undo_action_returned": undo.get("type") == "record_write_undone",
        }
        rows.append(_result(f"undo_{index:03d}", all(checks.values()), checks, "confirmed then undone"))
        db.close()

    db = _session()
    _patient(db, "ISO-A")
    _patient(db, "ISO-B")
    queue_record_write("ISO-A", "symptom", _payload(400), source_message="candidate", source_chat_message_id=1)
    other_actions = resolve_pending_record_write(db, "ISO-B", "confirm")
    owner_actions = resolve_pending_record_write(db, "ISO-A", "confirm") or []
    checks = {
        "other_patient_cannot_confirm": other_actions is None,
        "no_cross_patient_write": db.query(SymptomReport).filter(SymptomReport.patient_id == "ISO-B").count() == 0,
        "owner_can_confirm": bool(owner_actions and owner_actions[0].get("type") == "saved_symptom"),
    }
    rows.append(_result("patient_isolation", all(checks.values()), checks, "patient-scoped pending state"))
    db.close()

    db = _session()
    _patient(db, "AMBIGUOUS")
    queue_record_write("AMBIGUOUS", "symptom", _payload(500), source_message="candidate", source_chat_message_id=1)
    ambiguous = resolve_pending_record_write(db, "AMBIGUOUS", "maybe, let me think")
    checks = {
        "ambiguous_reply_does_not_write": ambiguous is None and db.query(SymptomReport).count() == 0,
        "ambiguous_reply_does_not_create_audit": db.query(PatientRecordWriteAudit).count() == 0,
    }
    resolve_pending_record_write(db, "AMBIGUOUS", "cancel")
    rows.append(_result("ambiguous_confirmation", all(checks.values()), checks, "ambiguous reply"))
    db.close()

    passed = sum(1 for row in rows if row["passed"])
    return {
        "schema_version": "confirmed_write_contract_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if passed == len(rows) else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "external_author_eval_completed": False,
        "internal_vs_external_authored": "internal_generated_contract_cases",
        "was_used_for_tuning": True,
        "full_live_generation_n": 0,
        "case_count": len(rows),
        "passed_n": passed,
        "failed_n": len(rows) - passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "rows": rows,
        "failures": [row for row in rows if not row["passed"]],
        "claim_boundary": (
            "Internal persistence-contract evaluation only. It tests confirmation and audit invariants, "
            "not natural-language coverage, clinical correctness, real-world safety, or clinical validation."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "build_report", "write_report"]
