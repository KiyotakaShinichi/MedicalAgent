import json

from backend.models import ChatMessage
from backend.services.app_logging import log_app_event
from backend.services.conversation_state import remember_turn, state_snapshot
from backend.services.support_chat.response_helpers import compound_intent_payload


def persist_support_turn(
    *,
    db,
    patient_id,
    response,
    actions,
    tool_plan,
    agent_result,
    agentic_shadow,
    compound_intent,
    llm_compound_verdict,
    urgent_flags,
    high_risk_alert,
):
    assistant_record = ChatMessage(
        patient_id=patient_id,
        role="assistant",
        message=response,
        intent="patient_support_response",
        saved_actions_json=json.dumps(
            {
                "saved_actions": actions,
                "tool_plan": tool_plan,
                "conversation_state": state_snapshot(patient_id),
                "agent_pipeline": {
                    "intent": agent_result.get("intent"),
                    "safety": agent_result.get("safety"),
                    "citations": agent_result.get("citations") or [],
                    "cache": agent_result.get("cache"),
                    "validation": agent_result.get("validation"),
                    "guardrails": agent_result.get("guardrails"),
                    "rag_evaluation": agent_result.get("rag_evaluation"),
                    "emotional_distress": agent_result.get("emotional_distress"),
                    "agentic_shadow": agentic_shadow,
                    "evidence_envelope": agent_result.get("evidence_envelope"),
                    "release_authorization": agent_result.get("release_authorization"),
                },
            }
        ),
    )
    db.add(assistant_record)
    db.flush()
    if high_risk_alert is not None:
        try:
            from backend.services.high_risk_conversation_alerts import attach_assistant_message

            attach_assistant_message(db, high_risk_alert.id, assistant_record.id)
        except Exception:  # noqa: BLE001 - alert linkage is secondary to preserving chat
            pass
    db.commit()
    db.refresh(assistant_record)
    remember_turn(patient_id, "assistant", response, actions=actions)
    log_app_event(
        db=db,
        event_type="agent_rag",
        patient_id=patient_id,
        route="/me/chat",
        status="ok",
        input_payload={
            "intent": agent_result.get("intent"),
            "safety_level": (agent_result.get("safety") or {}).get("level"),
            "cache": agent_result.get("cache"),
            "tool_plan": tool_plan,
            "agentic_shadow": agentic_shadow,
            "conversation_state": {
                "pending_actions": state_snapshot(patient_id).get("pending_actions"),
            },
        },
        output_payload={
            "citation_count": len(agent_result.get("citations") or []),
            "validation": agent_result.get("validation"),
            "agentic_shadow_review_required": bool(agentic_shadow.get("review_required")),
        },
    )

    return {
        "reply": response,
        "saved_actions": actions,
        "tool_plan": tool_plan,
        "conversation_state": state_snapshot(patient_id),
        "urgent_flags": urgent_flags,
        "evidence_envelope": agent_result.get("evidence_envelope"),
        "release_authorization": agent_result.get("release_authorization"),
        "agent_pipeline": {
            "intent": agent_result.get("intent"),
            "safety": agent_result.get("safety"),
            "citations": agent_result.get("citations") or [],
            "cache": agent_result.get("cache"),
            "validation": agent_result.get("validation"),
            "guardrails": agent_result.get("guardrails"),
            "rag_evaluation": agent_result.get("rag_evaluation"),
            "pipeline_trace": agent_result.get("pipeline_trace"),
            "compound_intent": compound_intent_payload(compound_intent, llm_compound_verdict),
            "emotional_distress": agent_result.get("emotional_distress"),
            "agentic_shadow": agentic_shadow,
            "evidence_envelope": agent_result.get("evidence_envelope"),
            "release_authorization": agent_result.get("release_authorization"),
        },
        "assistant_message_id": assistant_record.id,
        "safety_note": (
            "This assistant logs and summarizes information only. "
            "It does not diagnose or give treatment instructions."
        ),
    }
