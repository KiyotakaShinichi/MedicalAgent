from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import Patient, PatientRecordWriteAudit, SymptomReport
from backend.services.confirmed_record_write import queue_record_write, resolve_pending_record_write, undo_record_write
from backend.services.conversation_state import get_pending_action


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Patient(id="CONFIRM-P001", name="Confirm Patient", diagnosis="Demo"))
    session.commit()
    return session


def test_record_requires_separate_confirmation_and_carries_provenance():
    db = _db()
    try:
        action = queue_record_write(
            "CONFIRM-P001",
            "symptom",
            {"date": "2026-07-14", "symptom": "nausea", "severity": 6},
            source_message="I have nausea severity 6/10",
            source_chat_message_id=None,
        )
        assert action["type"] == "pending_record_confirmation"
        assert action["confirmation_digest"]
        assert action["expires_at_epoch"] > 0
        assert db.query(SymptomReport).count() == 0

        saved = resolve_pending_record_write(db, "CONFIRM-P001", "Confirm save")
        assert saved and saved[0]["type"] == "saved_symptom"
        assert saved[0]["undo_available"] is True
        assert db.query(SymptomReport).count() == 1
        audit = db.query(PatientRecordWriteAudit).one()
        assert audit.status == "saved"
        assert "patient_confirmed_support_chat" in audit.provenance_json
    finally:
        db.close()


def test_tampered_preview_is_rejected_without_a_write():
    db = _db()
    try:
        queue_record_write(
            "CONFIRM-P001",
            "symptom",
            {"date": "2026-07-14", "symptom": "nausea", "severity": 4},
            source_message="nausea 4/10",
            source_chat_message_id=44,
        )
        pending = get_pending_action("CONFIRM-P001", "confirmed_record_write")
        pending["items"][0]["payload"]["severity"] = 9
        result = resolve_pending_record_write(db, "CONFIRM-P001", "confirm")
        assert result[0]["reason"] == "confirmation_payload_integrity_check_failed"
        assert db.query(SymptomReport).count() == 0
        assert db.query(PatientRecordWriteAudit).count() == 0
    finally:
        db.close()


def test_duplicate_active_record_is_prevented():
    db = _db()
    try:
        payload = {"date": "2026-07-14", "symptom": "nausea", "severity": 6}
        queue_record_write("CONFIRM-P001", "symptom", payload, source_message="first", source_chat_message_id=None)
        resolve_pending_record_write(db, "CONFIRM-P001", "yes")
        queue_record_write("CONFIRM-P001", "symptom", payload, source_message="again", source_chat_message_id=None)
        duplicate = resolve_pending_record_write(db, "CONFIRM-P001", "yes")
        assert duplicate and duplicate[0]["type"] == "duplicate_record_prevented"
        assert db.query(SymptomReport).count() == 1
        assert db.query(PatientRecordWriteAudit).count() == 1
    finally:
        db.close()


def test_confirmed_write_can_be_undone_without_deleting_audit():
    db = _db()
    try:
        queue_record_write(
            "CONFIRM-P001",
            "symptom",
            {"date": "2026-07-14", "symptom": "fatigue", "severity": 4},
            source_message="fatigue 4/10",
            source_chat_message_id=None,
        )
        saved = resolve_pending_record_write(db, "CONFIRM-P001", "confirm")
        audit_id = saved[0]["audit_action_id"]
        undone = undo_record_write(db, "CONFIRM-P001", audit_id)
        assert undone["type"] == "record_write_undone"
        assert db.query(SymptomReport).count() == 0
        audit = db.query(PatientRecordWriteAudit).one()
        assert audit.status == "undone"
        assert audit.undone_at is not None
    finally:
        db.close()
