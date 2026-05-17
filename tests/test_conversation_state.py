import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.conversation_state import (
    clear_pending_action,
    get_pending_action,
    remember_turn,
    set_pending_action,
    state_snapshot,
)


def test_conversation_state_tracks_pending_symptom_and_recent_actions():
    patient_id = "TEST-MEMORY-P001"
    clear_pending_action(patient_id, "symptom_save")

    remember_turn(patient_id, "user", "I have nausea")
    set_pending_action(patient_id, "symptom_save", {"symptom": "nausea", "type": "partial_symptom_detected"})

    pending = get_pending_action(patient_id, "symptom_save")
    assert pending is not None
    assert pending["symptom"] == "nausea"

    remember_turn(patient_id, "assistant", "I logged nausea severity 7.", actions=[
        {"type": "saved_symptom", "symptom": "nausea", "severity": 7}
    ])
    clear_pending_action(patient_id, "symptom_save")

    snapshot = state_snapshot(patient_id)
    assert snapshot["message_count"] == 2
    assert snapshot["pending_actions"] == {}
    assert snapshot["recent_saved_actions"][-1]["type"] == "saved_symptom"
