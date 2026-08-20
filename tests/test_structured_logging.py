from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.services.admin_action_audit import append_admin_action
from backend.services.structured_logging import JsonEventFormatter, build_event


def test_structured_event_has_correlation_fields():
    event = build_event("rag_retrieval_completed", user_role="patient", patient_id="P001")

    assert event["schema_version"] == "structured_event_v2"
    assert event["event_type"] == "rag_retrieval_completed"
    assert event["request_id"]
    assert event["correlation_id"]
    assert event["timestamp"]
    assert event["patient_id"] == "[REDACTED]"


def test_structured_event_redacts_sensitive_nested_fields():
    event = build_event(
        "agent_turn",
        details={
            "prompt": "private medical question",
            "nested": {"access_token": "secret", "password": "secret"},
            "route": "/patient/chat",
        },
    )

    assert event["details"]["prompt"] == "[REDACTED]"
    assert event["details"]["nested"]["access_token"] == "[REDACTED]"
    assert event["details"]["nested"]["password"] == "[REDACTED]"
    assert event["details"]["route"] == "/patient/chat"


def test_json_formatter_emits_machine_readable_event():
    event = build_event("health_probe", request_id="req-1", details={"status": "ok"})
    record = logging.LogRecord("nlcare.events", logging.INFO, __file__, 1, "health_probe", (), None)
    record.nlcare_event = event

    payload = json.loads(JsonEventFormatter().format(record))

    assert payload["event_type"] == "health_probe"
    assert payload["request_id"] == "req-1"
    assert payload["details"] == {"status": "ok"}


def test_admin_action_audit_appends_jsonl(tmp_path: Path):
    path = tmp_path / "admin_actions.jsonl"
    event = append_admin_action("rerun_benchmark", artifact_id="live_rag_eval", output_path=path)

    assert path.exists()
    assert "rerun_benchmark" in path.read_text(encoding="utf-8")
    assert event["user_role"] == "admin"
