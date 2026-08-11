from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import ImagingReport, LabResult, Patient, SymptomReport
from backend.services.support_chat_agent import handle_patient_chat


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Patient(id="CHANGE-P001", name="Change Demo Patient", diagnosis="Synthetic breast cancer demo"))
    session.commit()
    return session


def _preview_and_confirm(db, message, expected_type):
    preview = handle_patient_chat(db=db, patient_id="CHANGE-P001", message=message)
    assert any(action["type"] == "pending_record_confirmation" for action in preview["saved_actions"])
    confirmed = handle_patient_chat(db=db, patient_id="CHANGE-P001", message="Confirm save")
    assert any(action["type"] == expected_type for action in confirmed["saved_actions"])
    return confirmed


def test_confirmed_multimodal_records_produce_bounded_change_explanation(monkeypatch):
    monkeypatch.setenv("LLM_ADJUDICATION_ENABLED", "false")

    def _unexpected_external_rag(**_kwargs):
        raise AssertionError("patient-scoped record comparison must not use external-evidence RAG")

    monkeypatch.setattr(
        "backend.services.support_chat_agent.run_patient_agent_pipeline",
        _unexpected_external_rag,
    )
    db = _db()
    try:
        _preview_and_confirm(db, "Log nausea severity 8/10 on 2026-07-01", "saved_symptom")
        _preview_and_confirm(db, "Log nausea severity 4/10 on 2026-08-01", "saved_symptom")
        _preview_and_confirm(
            db,
            "CBC on 2026-07-01: WBC 3.0, hemoglobin 10.0, platelets 120",
            "saved_labs",
        )
        _preview_and_confirm(
            db,
            "CBC on 2026-08-01: WBC 5.2, hemoglobin 12.4, platelets 190",
            "saved_labs",
        )
        _preview_and_confirm(
            db,
            "MRI report on 2026-07-01 impression: left breast mass measures 3.2 cm.",
            "saved_imaging_report",
        )
        latest = _preview_and_confirm(
            db,
            "MRI report on 2026-08-01 impression: left breast mass measures 2.4 cm.",
            "saved_imaging_report",
        )

        assert db.query(SymptomReport).count() == 2
        assert db.query(LabResult).count() == 2
        assert db.query(ImagingReport).count() == 2
        assert "does not show whether treatment is working" in latest["reply"].lower()

        status = handle_patient_chat(
            db=db,
            patient_id="CHANGE-P001",
            message="Am I improving? Is the treatment working?",
        )
        reply = status["reply"].lower()
        assert "fewer fixed portal review concerns" in reply
        assert "highest recorded symptom severity changed from 8/10 to 4/10" in reply
        assert "3.2 cm to 2.4 cm" in reply
        assert "does not show whether treatment is working" in reply
        assert "treatment-response score" not in reply
        assert "treatment is effective" not in reply
        assert status["agent_pipeline"]["citations"] == []
        assert "patient_record_context" in status["agent_pipeline"]["pipeline_trace"]["steps"]
        assert status["evidence_envelope"]["evidence_required"] is False
        assert status["release_authorization"]["disposition"] == "ALLOW"
    finally:
        db.close()
        db.bind.dispose()


def test_record_mentions_never_write_without_confirmation(monkeypatch):
    monkeypatch.setenv("LLM_ADJUDICATION_ENABLED", "false")
    db = _db()
    try:
        preview = handle_patient_chat(
            db=db,
            patient_id="CHANGE-P001",
            message="CBC on 2026-08-10: WBC 4.2, hemoglobin 12.1, platelets 170",
        )
        assert any(action["type"] == "pending_record_confirmation" for action in preview["saved_actions"])
        assert db.query(LabResult).count() == 0

        cancelled = handle_patient_chat(db=db, patient_id="CHANGE-P001", message="Cancel")
        assert any(action["type"] == "record_write_cancelled" for action in cancelled["saved_actions"])
        assert db.query(LabResult).count() == 0
    finally:
        db.close()
        db.bind.dispose()
