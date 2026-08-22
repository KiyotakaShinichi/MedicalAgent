import re

from backend.services.support_chat_safety import _immediate_danger_reply


def apply_emotional_distress_mode(reply, emotional_distress):
    if emotional_distress is None or not getattr(emotional_distress, "detected", False):
        return reply
    mode = emotional_distress.response_mode
    if mode == "crisis_support":
        return (
            "I'm really sorry you're carrying this much right now. I can't provide crisis care here, "
            "but if you might hurt yourself or feel unsafe, please contact local emergency services or "
            "a crisis hotline now, and reach out to someone near you. I can stay in the lane of helping "
            "organize questions or records for your care team, but this needs immediate human support."
        )
    if mode in {"urgent_clinician_review", "clinician_review_with_warm_handoff"}:
        if mode == "urgent_clinician_review":
            return _immediate_danger_reply()
        prefix = (
            "I'm sorry this feels so heavy. I can't diagnose or predict what is happening from here, "
            "but this is worth bringing to your oncology team for review. "
        )
    else:
        prefix = (
            "I'm sorry you're carrying that fear. I can't diagnose or predict what is happening from here, "
            "but I can help organize the information and questions for your care team. "
        )
    if str(reply).startswith(prefix[:24]):
        return reply
    return f"{prefix}{reply}"


def append_alert_notice(reply, actions):
    notices = [
        str(action.get("message") or "").strip()
        for action in (actions or [])
        if action.get("type") in {"high_risk_review_alert", "high_risk_review_alert_failed"}
        and action.get("message")
    ]
    text = str(reply or "").strip()
    for notice in notices:
        if notice and notice not in text:
            text = f"{text} {notice}".strip()
    return text


def compound_intent_payload(envelope, llm_verdict):
    """Build the trace payload for compound-intent routing."""
    if envelope is None:
        return None
    payload = envelope.to_dict()
    if llm_verdict is not None:
        payload["llm"] = {
            "available": True,
            "language": llm_verdict.get("language"),
            "llm_confidence": llm_verdict.get("llm_confidence"),
            "provider": llm_verdict.get("provider"),
            "model": llm_verdict.get("model"),
        }
    else:
        payload["llm"] = {"available": False, "reason": "llm_unavailable_or_disabled"}
    return payload


def tool_request_followup_message(tool_targets, casual_opener):
    target_to_prompt = {
        "save_symptom": (
            "I can log a symptom — please send the symptom name AND a "
            'severity from 0–10 (e.g. "nausea severity 6/10 today").'
        ),
        "save_complete_cbc": (
            "I can save a CBC row — please send WBC, hemoglobin, and "
            'platelets together (e.g. "WBC 2.1, hemoglobin 10.4, '
            'platelets 145").'
        ),
        "save_imaging_report": (
            "I can save an imaging report — please paste the report "
            "date plus the findings or impression text."
        ),
        "save_medication": (
            "I can log a medication — please send the medication name "
            "and (if known) the dose and frequency."
        ),
    }
    prompts: list[str] = []
    for target in tool_targets or ["save_symptom"]:
        prompt = target_to_prompt.get(target)
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    body = " ".join(prompts) if prompts else target_to_prompt["save_symptom"]
    if casual_opener:
        return f"Hi! Sure, I can help with that. {body}"
    return body


def has_tool_action(actions):
    return any(
        action.get("type")
        in {
            "saved_symptom",
            "saved_labs",
            "saved_medication",
            "saved_imaging_report",
            "partial_symptom_detected",
            "partial_labs_detected",
            "partial_imaging_detected",
            "symptom_save_failed",
            "pending_record_confirmation",
            "record_write_cancelled",
            "duplicate_record_prevented",
        }
        for action in actions
    )


def should_bypass_rag_for_tool_actions(actions, routing_intent):
    if routing_intent != "data_entry_confirmation":
        return False
    return has_tool_action(actions)


def should_bypass_rag_for_patient_context(routing_intent, message=""):
    """Keep patient-record summaries separate from external evidence RAG."""
    if routing_intent in {"patient_memory", "patient_timeline_monitoring"}:
        return True
    normalized = re.sub(r"\s+", " ", str(message or "").lower()).strip()
    record_comparison_phrases = (
        "am i improving",
        "am i getting better",
        "am i getting worse",
        "is my treatment working",
        "how is my treatment",
        "my treatment progress",
        "what changed in my record",
        "what changed in my results",
        "gumagaling ba ako",
        "lumalala ba ako",
        "epektibo ba ang treatment ko",
        "umuuubra ba treatment ko",
    )
    return any(phrase in normalized for phrase in record_comparison_phrases)


def portal_help_reply():
    return (
        "Use the left navigation to open Overview, Labs, Signals, Timeline, Family & Genetics, or Support. "
        "In Support, select the plus button beside the message box to open a structured form for symptoms, "
        "CBC values, imaging reports, medications, or treatment notes. Nothing is saved until you review and "
        "submit or explicitly confirm it."
    )
