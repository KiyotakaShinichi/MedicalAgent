"""Confirmed, provenance-stamped support-chat writes.

The support assistant may extract candidate values, but this module is the
only path that persists them. Every write requires a separate user
confirmation turn, receives an idempotency key, is checked for an active
duplicate, and can be undone by the same patient.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

from backend.models import (
    ImagingReport,
    LabResult,
    MedicationLog,
    PatientRecordWriteAudit,
    SymptomReport,
)
from backend.processing.radiology_analysis import detect_possible_metastatic_indicators
from backend.services.conversation_state import clear_pending_action, get_pending_action, set_pending_action


PENDING_KEY = "confirmed_record_write"
SUPPORTED_RECORD_TYPES = {"symptom", "cbc", "imaging", "medication"}

_CONFIRM_RE = re.compile(
    r"^(?:yes|yes please|confirm|confirm save|save it|save this|go ahead|"
    r"oo|opo|sige|i-confirm|kumpirmahin)(?:[.! ]*)$",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"^(?:no|cancel|cancel save|do not save|don't save|never mind|nevermind|"
    r"huwag|wag|hindi)(?:[.! ]*)$",
    re.IGNORECASE,
)


def is_confirmation_message(message: str) -> bool:
    return bool(_CONFIRM_RE.fullmatch(str(message or "").strip()))


def is_cancellation_message(message: str) -> bool:
    return bool(_CANCEL_RE.fullmatch(str(message or "").strip()))


def queue_record_write(
    patient_id: str,
    record_type: str,
    payload: dict[str, Any],
    *,
    source_message: str,
    source_chat_message_id: int | None,
) -> dict[str, Any]:
    if record_type not in SUPPORTED_RECORD_TYPES:
        raise ValueError(f"Unsupported record type: {record_type}")
    serial_payload = _jsonable(payload)
    pending = get_pending_action(patient_id, PENDING_KEY)
    if not pending or pending.get("source_chat_message_id") != source_chat_message_id:
        pending = {
            "confirmation_id": str(uuid4()),
            "source_message": str(source_message or "")[:2000],
            "source_chat_message_id": source_chat_message_id,
            "items": [],
        }
    items = list(pending.get("items") or [])
    candidate = {"record_type": record_type, "payload": serial_payload}
    if candidate not in items:
        items.append(candidate)
    pending["items"] = items
    set_pending_action(patient_id, PENDING_KEY, pending)
    return {
        "type": "pending_record_confirmation",
        "record_type": record_type,
        "confirmation_id": pending["confirmation_id"],
        "preview": _preview(record_type, serial_payload),
        "message": "Nothing has been saved yet. Confirm or cancel this record preview.",
        "requires_confirmation": True,
    }


def resolve_pending_record_write(db, patient_id: str, message: str) -> list[dict[str, Any]] | None:
    pending = get_pending_action(patient_id, PENDING_KEY)
    if not pending:
        return None
    if is_cancellation_message(message):
        clear_pending_action(patient_id, PENDING_KEY)
        return [{
            "type": "record_write_cancelled",
            "confirmation_id": pending.get("confirmation_id"),
            "message": "Cancelled. No patient record was changed.",
        }]
    if not is_confirmation_message(message):
        return None

    actions: list[dict[str, Any]] = []
    for index, item in enumerate(pending.get("items") or []):
        persisted = _persist_item(
            db,
            patient_id,
            pending,
            item,
            confirmation_message=message,
            item_index=index,
        )
        actions.append(persisted)
        if persisted.get("type") == "saved_imaging_report":
            payload = dict(item.get("payload") or {})
            indicators = detect_possible_metastatic_indicators(
                f"{payload.get('findings', '')} {payload.get('impression', '')}"
            )
            if indicators:
                actions.append({
                    "type": "possible_metastatic_indicator",
                    "sites": sorted({indicator["site"] for indicator in indicators}),
                    "message": (
                        "The confirmed report wording contains a review flag. "
                        "This is not a diagnosis and should be reviewed by the oncology team."
                    ),
                })
    clear_pending_action(patient_id, PENDING_KEY)
    return actions or [{
        "type": "record_write_cancelled",
        "message": "There was no complete record waiting to be saved.",
    }]


def undo_record_write(db, patient_id: str, audit_id: int) -> dict[str, Any]:
    audit = (
        db.query(PatientRecordWriteAudit)
        .filter(
            PatientRecordWriteAudit.id == audit_id,
            PatientRecordWriteAudit.patient_id == patient_id,
        )
        .first()
    )
    if audit is None:
        raise ValueError("Confirmed record write was not found for this patient.")
    if audit.status == "undone":
        return {
            "type": "record_write_undone",
            "audit_action_id": audit.id,
            "record_type": audit.record_type,
            "already_undone": True,
            "message": "This save was already undone.",
        }
    model = _record_model(audit.record_type)
    row = None
    if model is not None and audit.record_id is not None:
        row = (
            db.query(model)
            .filter(model.id == audit.record_id, model.patient_id == patient_id)
            .first()
        )
    if row is not None:
        db.delete(row)
    audit.status = "undone"
    audit.undone_at = datetime.now(timezone.utc)
    db.commit()
    return {
        "type": "record_write_undone",
        "audit_action_id": audit.id,
        "record_type": audit.record_type,
        "already_undone": False,
        "message": "The confirmed portal entry was removed. The audit envelope remains for traceability.",
    }


def _persist_item(db, patient_id, pending, item, *, confirmation_message: str, item_index: int):
    record_type = str(item.get("record_type") or "")
    payload = dict(item.get("payload") or {})
    fingerprint = _fingerprint(patient_id, record_type, payload)
    duplicate = (
        db.query(PatientRecordWriteAudit)
        .filter(
            PatientRecordWriteAudit.patient_id == patient_id,
            PatientRecordWriteAudit.record_fingerprint == fingerprint,
            PatientRecordWriteAudit.status == "saved",
        )
        .order_by(PatientRecordWriteAudit.id.desc())
        .first()
    )
    if duplicate is not None:
        return {
            "type": "duplicate_record_prevented",
            "record_type": record_type,
            "audit_action_id": duplicate.id,
            "existing_record_id": duplicate.record_id,
            "message": "The same active portal entry already exists, so NLCare did not create a duplicate.",
        }

    confirmation_id = str(pending.get("confirmation_id") or uuid4())
    idempotency_key = _hash(f"{patient_id}|{confirmation_id}|{item_index}|{record_type}|{_canonical(payload)}")
    existing_attempt = (
        db.query(PatientRecordWriteAudit)
        .filter(PatientRecordWriteAudit.idempotency_key == idempotency_key)
        .first()
    )
    if existing_attempt is not None:
        return {
            "type": "duplicate_record_prevented",
            "record_type": record_type,
            "audit_action_id": existing_attempt.id,
            "existing_record_id": existing_attempt.record_id,
            "message": "This confirmation was already processed, so NLCare did not repeat the write.",
        }

    row = _build_record(patient_id, record_type, payload, pending.get("source_message"))
    db.add(row)
    db.flush()
    provenance = {
        "source": "patient_confirmed_support_chat",
        "confirmation_required": True,
        "confirmation_id": confirmation_id,
        "source_chat_message_id": pending.get("source_chat_message_id"),
        "clinical_validation": False,
    }
    audit = PatientRecordWriteAudit(
        patient_id=patient_id,
        record_type=record_type,
        record_id=row.id,
        idempotency_key=idempotency_key,
        record_fingerprint=fingerprint,
        source_chat_message_id=pending.get("source_chat_message_id"),
        source_message=str(pending.get("source_message") or "")[:2000],
        confirmation_message=str(confirmation_message or "")[:500],
        payload_json=_canonical(payload),
        provenance_json=_canonical(provenance),
        status="saved",
    )
    db.add(audit)
    db.flush()
    return {
        "type": _saved_action_type(record_type),
        **payload,
        "audit_action_id": audit.id,
        "record_id": row.id,
        "undo_available": True,
        "provenance": provenance,
    }


def _build_record(patient_id: str, record_type: str, payload: dict[str, Any], source_message: str):
    record_date = date.fromisoformat(str(payload["date"]))
    source_note = f"Patient-confirmed support chat entry. Source text: {str(source_message or '')[:500]}"
    if record_type == "symptom":
        return SymptomReport(
            patient_id=patient_id,
            date=record_date,
            symptom=str(payload["symptom"]),
            severity=int(payload["severity"]),
            notes=source_note,
        )
    if record_type == "cbc":
        return LabResult(
            patient_id=patient_id,
            date=record_date,
            wbc=float(payload["wbc"]),
            hemoglobin=float(payload["hemoglobin"]),
            platelets=float(payload["platelets"]),
            source="support_chat_confirmed",
            source_note=source_note,
        )
    if record_type == "imaging":
        return ImagingReport(
            patient_id=patient_id,
            date=record_date,
            modality=str(payload["modality"]),
            report_type=str(payload["report_type"]),
            body_site=payload.get("body_site"),
            findings=str(payload["findings"]),
            impression=str(payload["impression"]),
        )
    if record_type == "medication":
        return MedicationLog(
            patient_id=patient_id,
            date=record_date,
            medication=str(payload["medication"]),
            dose=payload.get("dose"),
            frequency=payload.get("frequency"),
            notes=source_note,
            source="support_chat_confirmed",
        )
    raise ValueError(f"Unsupported record type: {record_type}")


def _record_model(record_type: str):
    return {
        "symptom": SymptomReport,
        "cbc": LabResult,
        "imaging": ImagingReport,
        "medication": MedicationLog,
    }.get(record_type)


def _saved_action_type(record_type: str) -> str:
    return {
        "symptom": "saved_symptom",
        "cbc": "saved_labs",
        "imaging": "saved_imaging_report",
        "medication": "saved_medication",
    }[record_type]


def _preview(record_type: str, payload: dict[str, Any]) -> str:
    if record_type == "symptom":
        return f"Symptom: {payload.get('symptom')}; severity {payload.get('severity')}/10; date {payload.get('date')}"
    if record_type == "cbc":
        return (
            f"CBC on {payload.get('date')}: WBC {payload.get('wbc')}, hemoglobin "
            f"{payload.get('hemoglobin')}, platelets {payload.get('platelets')}"
        )
    if record_type == "imaging":
        return f"{payload.get('modality')} report dated {payload.get('date')}"
    if record_type == "medication":
        details = " ".join(str(value) for value in (payload.get("dose"), payload.get("frequency")) if value)
        return f"Medication: {payload.get('medication')}{f' ({details})' if details else ''}; date {payload.get('date')}"
    return record_type


def _fingerprint(patient_id: str, record_type: str, payload: dict[str, Any]) -> str:
    return _hash(f"{patient_id}|{record_type}|{_canonical(payload)}")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _jsonable(value: Any):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
