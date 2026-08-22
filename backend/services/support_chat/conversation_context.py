import json

from backend.models import ChatMessage
from backend.services.conversation_state import get_pending_action
from backend.services.support_chat_extraction import (
    _extract_complete_labs,
    _extract_imaging_report,
    _extract_medication,
    _extract_severity,
    _extract_symptom,
    _is_general_lab_question,
    _looks_like_partial_imaging,
    _looks_like_partial_labs,
)


def extract_candidate_inputs(message):
    try:
        from backend.services.multilingual_tool_router import (
            normalize_lab_value_string,
            normalize_user_text,
        )

        normalized = normalize_user_text(message)
        lab_message = normalize_lab_value_string(normalized) if normalized else message
    except Exception:  # noqa: BLE001 - never break chat on normalization
        lab_message = message

    labs = _extract_complete_labs(lab_message) or _extract_complete_labs(message)
    imaging_report = _extract_imaging_report(lab_message) or _extract_imaging_report(message)
    symptom = _extract_symptom(message)
    return {
        "symptom": symptom,
        "labs": labs,
        "partial_labs": bool(
            not labs
            and (_looks_like_partial_labs(lab_message) or _looks_like_partial_labs(message))
            and not _is_general_lab_question(message)
        ),
        "imaging_report": imaging_report,
        "partial_imaging": bool(
            not imaging_report
            and (_looks_like_partial_imaging(lab_message) or _looks_like_partial_imaging(message))
        ),
        "medication": _extract_medication(lab_message) or _extract_medication(message),
    }


def resume_pending_symptom_if_possible(db, patient_id, message, extracted):
    """Resume a prior partial symptom save when the user replies with severity only."""
    if extracted.get("symptom"):
        return None
    severity = _extract_severity(message.lower())
    if severity is None:
        return None
    pending = get_pending_action(patient_id, "symptom_save")
    if pending and pending.get("symptom"):
        return {
            "symptom": str(pending["symptom"]),
            "severity": severity,
            "severity_provided": True,
            "resumed_from_memory": True,
            "memory_source": "conversation_state",
        }
    pending_symptom = latest_pending_symptom(db, patient_id)
    if not pending_symptom:
        return None
    return {
        "symptom": pending_symptom,
        "severity": severity,
        "severity_provided": True,
        "resumed_from_memory": True,
    }


def latest_pending_symptom(db, patient_id):
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.patient_id == patient_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(8)
        .all()
    )
    for row in rows:
        try:
            payload = json.loads(row.saved_actions_json or "{}")
        except (TypeError, ValueError):
            continue
        for action in payload.get("saved_actions") or []:
            action_type = action.get("type")
            if action_type == "saved_symptom":
                return None
            if action_type == "partial_symptom_detected" and action.get("symptom"):
                return str(action["symptom"])
    return None
