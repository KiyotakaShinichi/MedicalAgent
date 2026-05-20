from __future__ import annotations

from pathlib import Path

from backend.services.admin_action_audit import append_admin_action
from backend.services.structured_logging import build_event


def test_structured_event_has_correlation_fields():
    event = build_event("rag_retrieval_completed", user_role="patient", patient_id="P001")

    assert event["schema_version"] == "structured_event_v1"
    assert event["event_type"] == "rag_retrieval_completed"
    assert event["request_id"]
    assert event["correlation_id"]
    assert event["timestamp"]


def test_admin_action_audit_appends_jsonl(tmp_path: Path):
    path = tmp_path / "admin_actions.jsonl"
    event = append_admin_action("rerun_benchmark", artifact_id="live_rag_eval", output_path=path)

    assert path.exists()
    assert "rerun_benchmark" in path.read_text(encoding="utf-8")
    assert event["user_role"] == "admin"
