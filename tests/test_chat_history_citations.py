import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.crud import get_chat_messages
from backend.database import Base
from backend.models import ChatMessage, Patient


def test_chat_history_restores_governed_citations_from_trace_envelope():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        db.add(Patient(id="CITE-P001", name="Citation Demo"))
        db.add(ChatMessage(
            patient_id="CITE-P001",
            role="assistant",
            message="Source-backed educational answer.",
            intent="patient_support_response",
            saved_actions_json=json.dumps({
                "saved_actions": [],
                "agent_pipeline": {
                    "citations": [{
                        "id": "paper-chunk-1",
                        "title": "Use of PRO-CTCAE in oncology clinical trials",
                        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12452844/",
                    }]
                },
            }),
        ))
        db.commit()

        messages = get_chat_messages(db, "CITE-P001")
        assert messages[-1]["citations"][0]["id"] == "paper-chunk-1"
        assert messages[-1]["citations"][0]["title"] == "Use of PRO-CTCAE in oncology clinical trials"
    finally:
        db.close()
        engine.dispose()


def test_chat_history_tolerates_legacy_malformed_trace_json():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        db.add(Patient(id="CITE-P002", name="Legacy Demo"))
        db.add(ChatMessage(
            patient_id="CITE-P002",
            role="assistant",
            message="Legacy answer.",
            intent="patient_support_response",
            saved_actions_json="{not-json",
        ))
        db.commit()
        assert get_chat_messages(db, "CITE-P002")[-1]["citations"] == []
    finally:
        db.close()
        engine.dispose()
