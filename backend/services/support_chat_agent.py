import json
import re
from backend.models import (
    ChatMessage,
)
from backend.processing.radiology_analysis import detect_possible_metastatic_indicators
from backend.services.agent_rag import run_patient_agent_pipeline, route_intent, safety_scope_check
from backend.services.agent_input_gate import input_guardrail_check
from backend.services.agent_output_gate import output_guardrail_check
from backend.services.agent_post_gen import apply_post_gen_validator
from backend.services.app_logging import log_app_event
from backend.services.input_validation import validate_cbc_values, validate_imaging_report_payload, validate_symptom_payload
from backend.services.local_llm import select_support_tools_with_local_llm
from backend.services.support_chat_context import _recent_patient_context
from backend.services.rag_evidence_envelope import (
    build_fail_closed_error_result,
    enforce_evidence_release,
    enforce_transport_release,
    parse_evidence_envelope,
)
from backend.services.conversation_state import (
    clear_pending_action,
    get_pending_action,
    remember_turn,
    set_pending_action,
    state_snapshot,
)
from backend.services.confirmed_record_write import (
    queue_record_write,
    resolve_pending_record_write,
)


from backend.services.support_chat_policy import (
    ALLOWED_SUPPORT_INTENTS,
    ALLOWED_SUPPORT_TOOLS,
)

def handle_patient_chat(db, patient_id, message):
    normalized = message.strip()
    if not normalized:
        raise ValueError("Message cannot be empty")

    prior_state = state_snapshot(patient_id)
    prior_user_messages = [
        str(row.get("message") or "")
        for row in prior_state.get("recent_messages") or []
        if row.get("role") == "user"
    ][-3:]
    safety_followup = _resolve_safety_location_followup(db, patient_id, normalized)
    immediate_danger = _is_immediate_danger_statement(normalized)
    remember_turn(patient_id, "user", normalized)

    urgent_flags = _detect_urgent_flags(normalized)
    if immediate_danger:
        urgent_flags.append("immediate_danger_statement")
    if safety_followup:
        urgent_flags.append("safety_location_followup")
    urgent_flags = sorted(set(urgent_flags))
    routing_safety = safety_scope_check(
        normalized,
        urgent_flags,
        previous_user_messages=prior_user_messages,
    )
    input_guardrail_preview = input_guardrail_check(normalized, routing_safety)
    terminal_input_block = input_guardrail_preview.get("status") == "failed"
    try:
        from backend.services.emotional_distress_detection import detect_emotional_distress
        emotional_distress = detect_emotional_distress(normalized, safety=routing_safety)
        if emotional_distress.response_mode == "crisis_support":
            urgent_flags.append("emotional_crisis")
            routing_safety = safety_scope_check(
                normalized,
                urgent_flags,
                previous_user_messages=prior_user_messages,
            )
            emotional_distress = detect_emotional_distress(normalized, safety=routing_safety)
    except Exception:  # noqa: BLE001 - affective detection must never block chat
        emotional_distress = None
    # Compound-intent envelope: detects messages that mix a casual
    # opener ("hi") with a tool request ("can you log my symptoms?")
    # so the chat layer can still surface the tool ask instead of
    # collapsing to a bare greeting reply.  When the LLM adjudicator
    # is reachable, the 70B router model adds a multilingual second
    # opinion that catches languages our hard-coded tables don't.
    compound_intent = None
    llm_compound_verdict: dict | None = None
    try:
        if terminal_input_block:
            raise RuntimeError("terminal input safety block")
        from backend.services.compound_intent_router import detect_compound_intents_with_llm
        compound_intent, llm_compound_verdict = detect_compound_intents_with_llm(message)
    except Exception:  # noqa: BLE001 — never break chat on the router
        try:
            if terminal_input_block:
                raise RuntimeError("terminal input safety block")
            from backend.services.compound_intent_router import detect_compound_intents
            compound_intent = detect_compound_intents(message)
        except Exception:  # noqa: BLE001
            compound_intent = None
    extracted = _extract_candidate_inputs(normalized)
    resumed_symptom = _resume_pending_symptom_if_possible(db, patient_id, normalized, extracted)
    if resumed_symptom:
        extracted["symptom"] = resumed_symptom
    deterministic_plan = _deterministic_tool_plan(normalized, extracted, routing_safety)
    tool_plan = _select_tool_plan(normalized, extracted, deterministic_plan, routing_safety)

    user_record = ChatMessage(
        patient_id=patient_id,
        role="user",
        message=normalized,
        intent="patient_support",
    )
    db.add(user_record)
    db.flush()

    high_risk_alert = None
    alert_action = None
    confirmation_actions = (
        None
        if terminal_input_block
        else resolve_pending_record_write(db, patient_id, normalized)
    )
    actions = list(confirmation_actions or [])
    selected_tools = (
        set()
        if terminal_input_block or confirmation_actions is not None
        else set(tool_plan["selected_tools"])
    )
    if terminal_input_block:
        tool_plan = {
            **tool_plan,
            "intent": "safety_boundary",
            "selected_tools": ["none"],
            "forced_tools": [],
            "source": "terminal_input_safety_block",
            "confidence": 1.0,
            "reason": "input safety boundary is terminal; no record action or tool follow-up is allowed",
        }
    if confirmation_actions is not None:
        tool_plan = {
            **tool_plan,
            "intent": "data_entry_confirmation",
            "selected_tools": ["none"],
            "source": "confirmed_write_state",
            "confidence": 1.0,
            "reason": "resolved an explicit confirm/cancel turn against patient-scoped pending state",
        }

    symptom = extracted["symptom"]
    if "save_symptom" in selected_tools and symptom:
        # Honest-save rule: only persist when severity was explicitly provided.
        # Auto-defaulting silently caused "I've logged your symptom" replies
        # for messages that gave no severity, which is misleading.
        if not symptom.get("severity_provided") or symptom["severity"] is None:
            actions.append({
                "type": "partial_symptom_detected",
                "symptom": symptom["symptom"],
                "message": (
                    f"I noticed you mentioned {symptom['symptom']}, but I need a severity "
                    f"before I can save it. Please send it on a 0-10 scale, for example: "
                    f"\"{symptom['symptom']} severity 6/10 today\"."
                ),
            })
            set_pending_action(patient_id, "symptom_save", {
                "type": "partial_symptom_detected",
                "symptom": symptom["symptom"],
                "source": "support_chat_agent",
            })
        else:
            severity = int(symptom["severity"])
            try:
                validate_symptom_payload(symptom["symptom"], severity)
                actions.append(queue_record_write(
                    patient_id,
                    "symptom",
                    {
                        "date": _extract_date(normalized),
                        "symptom": symptom["symptom"],
                        "severity": severity,
                    },
                    source_message=normalized,
                    source_chat_message_id=user_record.id,
                ))
                clear_pending_action(patient_id, "symptom_save")
            except Exception as exc:
                actions.append({
                    "type": "symptom_save_failed",
                    "symptom": symptom["symptom"],
                    "severity": severity,
                    "reason": str(exc)[:200],
                    "message": (
                        f"I couldn't save the {symptom['symptom']} entry just now — "
                        f"there was a problem with the record. Please try again or "
                        f"log it from the portal manually."
                    ),
                })
    elif "request_symptom_details" in selected_tools and symptom:
        actions.append({
            "type": "partial_symptom_detected",
            "symptom": symptom["symptom"],
            "message": (
                f"I noticed you mentioned {symptom['symptom']}. If you want me to log it, "
                f"send the severity from 0-10, for example: "
                f"\"{symptom['symptom']} severity 6/10 today\"."
            ),
        })
        set_pending_action(patient_id, "symptom_save", {
            "type": "partial_symptom_detected",
            "symptom": symptom["symptom"],
            "source": "support_chat_agent",
        })

    labs = extracted["labs"]
    if "save_complete_cbc" in selected_tools and labs:
        validate_cbc_values(labs["wbc"], labs["hemoglobin"], labs["platelets"])
        lab_alerts = _clinical_lab_alerts(labs)
        actions.append(queue_record_write(
            patient_id,
            "cbc",
            {"date": _extract_date(normalized), **labs},
            source_message=normalized,
            source_chat_message_id=user_record.id,
        ))
        if lab_alerts:
            actions.append({
                "type": "clinical_rule_alert",
                "alerts": lab_alerts,
                "message": "CBC safety rule triggered before RAG retrieval.",
            })
            urgent_flags.extend([alert["rule"] for alert in lab_alerts])
    elif "request_missing_cbc_fields" in selected_tools and extracted["partial_labs"]:
        actions.append({
            "type": "partial_labs_detected",
            "message": "I saw lab information, but I need WBC, hemoglobin, and platelets together to save a CBC row.",
        })

    imaging_report = extracted["imaging_report"]
    if "save_imaging_report" in selected_tools and imaging_report:
        validate_imaging_report_payload(
            imaging_report["modality"],
            imaging_report["report_type"],
            imaging_report["findings"],
            imaging_report["impression"],
            body_site=imaging_report["body_site"],
        )
        actions.append(queue_record_write(
            patient_id,
            "imaging",
            imaging_report,
            source_message=normalized,
            source_chat_message_id=user_record.id,
        ))
        indicators = detect_possible_metastatic_indicators(
            f"{imaging_report['findings']} {imaging_report['impression']}"
        )
        if indicators:
            sites = sorted({indicator["site"] for indicator in indicators})
            actions.append({
                "type": "possible_metastatic_indicator",
                "sites": sites,
                "message": (
                    "Report wording mentions possible distant-disease indicators. "
                    "This is not a diagnosis and should be reviewed by the oncology team."
                ),
            })
            urgent_flags.extend([f"imaging_{site}" for site in sites])
    elif "request_missing_imaging_details" in selected_tools and extracted["partial_imaging"]:
        actions.append({
            "type": "partial_imaging_detected",
            "message": "I saw imaging wording. To save it as a report, paste the report date plus findings or impression text.",
        })

    medication = extracted["medication"]
    if "save_medication" in selected_tools and medication:
        actions.append(queue_record_write(
            patient_id,
            "medication",
            {"date": _extract_date(normalized), **medication},
            source_message=normalized,
            source_chat_message_id=user_record.id,
        ))

    # Compound-intent: user asked to log something ("hi, can you log
    # my symptoms?") but no extractor produced concrete data.  Surface
    # a follow-up action so the user gets asked for the missing detail
    # instead of receiving a bare casual reply.
    has_concrete_save = any(
        a.get("type", "").startswith("saved_")
        or a.get("type", "").startswith("partial_")
        or a.get("type") == "pending_record_confirmation"
        for a in actions
    )
    if (
        compound_intent is not None
        and compound_intent.has_tool_request
        and not has_concrete_save
        and not urgent_flags
        and not terminal_input_block
        and tool_plan.get("intent") != "portal_help"
    ):
        actions.append({
            "type": "partial_tool_request_detected",
            "tool_targets": compound_intent.tool_request_targets,
            "casual_opener": compound_intent.has_casual_opener,
            "suggested_acknowledgment": compound_intent.suggested_acknowledgment,
            "message": _tool_request_followup_message(
                compound_intent.tool_request_targets,
                compound_intent.has_casual_opener,
            ),
        })

    # Lab/imaging extractors may have added urgent_flags after the initial
    # safety scope check.  Recompute so the bypass path and the RAG pipeline
    # both see the elevated safety level (e.g. very_low_wbc → high_risk).
    if urgent_flags:
        routing_safety = safety_scope_check(
            normalized,
            urgent_flags,
            previous_user_messages=prior_user_messages,
        )

    # Queue the review alert only after every extractor has had a chance to
    # add safety flags. This covers explicit danger language as well as urgent
    # CBC/imaging evidence discovered during structured input parsing.
    try:
        from backend.services.high_risk_conversation_alerts import queue_and_dispatch_alert

        high_risk_alert, alert_action = queue_and_dispatch_alert(
            db,
            patient_id=patient_id,
            source_chat_message_id=user_record.id,
            immediate_danger=immediate_danger,
            urgent_flags=sorted(set(urgent_flags)),
            emotional_distress=emotional_distress,
        )
    except Exception:  # noqa: BLE001 - chat safety response must survive alert-outbox failure
        alert_action = {
            "type": "high_risk_review_alert_failed",
            "message": (
                "The review-alert queue could not confirm that it recorded this turn. "
                "Do not wait for a portal reply if you feel unsafe or in immediate danger."
            ),
        } if (immediate_danger or urgent_flags) else None
    if alert_action is not None:
        actions.append(alert_action)

    patient_context = _recent_patient_context(db, patient_id)
    direct_safety_reply = None
    direct_scope_reply = None
    direct_portal_reply = None
    if safety_followup:
        direct_safety_reply = _safety_location_followup_reply()
    elif immediate_danger:
        direct_safety_reply = _immediate_danger_reply()
    elif _is_out_of_domain_request(
        normalized,
        actions=actions,
        safety=routing_safety,
        emotional_distress=emotional_distress,
    ):
        direct_scope_reply = _out_of_domain_reply()
    elif tool_plan.get("intent") == "portal_help":
        direct_portal_reply = _portal_help_reply()

    fallback_response = (
        direct_safety_reply
        or direct_scope_reply
        or direct_portal_reply
        or _build_response(normalized, actions, urgent_flags, patient_context)
    )
    routing_intent = (
        "safety_boundary"
        if direct_safety_reply
        else "scope_boundary"
        if direct_scope_reply
        else "portal_help"
        if direct_portal_reply
        else tool_plan["intent"]
        if tool_plan.get("intent") in ALLOWED_SUPPORT_INTENTS
        else route_intent(normalized, actions=actions, safety=routing_safety)
    )
    from backend.services.live_agentic_shadow import build_live_agentic_shadow

    agentic_shadow = build_live_agentic_shadow(
        normalized,
        patient_context=state_snapshot(patient_id),
        live_intent=routing_intent,
        live_safety=routing_safety,
        live_tools=list(tool_plan.get("selected_tools") or []),
    )
    contextual_record_route = _should_bypass_rag_for_patient_context(routing_intent, normalized)
    if (
        _should_use_llm_direct_reply(routing_intent, routing_safety, actions, urgent_flags)
        and not _has_tool_action(actions)
        and not contextual_record_route
    ):
        fallback_response = _generate_llm_response(normalized, actions, urgent_flags, patient_context, fallback_response)
    fallback_response = _apply_emotional_distress_mode(fallback_response, emotional_distress)
    if (
        direct_safety_reply
        or direct_scope_reply
        or direct_portal_reply
        or contextual_record_route
        or _should_bypass_rag_for_tool_actions(actions, routing_intent)
    ):
        direct_reason = (
            "deterministic safety clarification; no RAG generation"
            if direct_safety_reply
            else "out-of-domain request; no LLM or RAG generation"
            if direct_scope_reply
            else "deterministic portal navigation help; no external-evidence RAG generation"
            if direct_portal_reply
            else "patient-scoped record comparison; no external-evidence RAG generation"
            if contextual_record_route
            else "deterministic tool confirmation; no RAG generation"
        )
        agent_result = {
            "reply": fallback_response,
            "intent": routing_intent,
            "safety": routing_safety,
            "citations": [],
            "cache": {"status": "bypassed_for_deterministic_tool_action"},
            "validation": {"status": "not_needed_for_tool_confirmation"},
            "guardrails": {
                "input_passed": routing_safety.get("level") != "blocked",
                "output_passed": True,
                "reason": direct_reason,
            },
            "rag_evaluation": None,
            "pipeline_trace": {
                "steps": [
                    "safety_gate",
                    "intent_routing",
                    (
                        "deterministic_safety_reply"
                        if direct_safety_reply
                        else "scope_boundary_reply"
                        if direct_scope_reply
                        else "portal_help_reply"
                        if direct_portal_reply
                        else "patient_record_context"
                        if contextual_record_route
                        else "deterministic_tool_action"
                    ),
                    (
                        "safety_clarification"
                        if direct_safety_reply
                        else "scope_redirect"
                        if direct_scope_reply
                        else "portal_navigation"
                        if direct_portal_reply
                        else "record_change_explanation"
                        if contextual_record_route
                        else "confirmation_reply"
                    ),
                ],
                "terminal_step": "direct_support",
                "safety_level": routing_safety.get("level"),
                "intent": routing_intent,
                "subquery_count": 0,
                "retrieved_count": 0,
                "reranked_count": 0,
                "compressed_count": 0,
            },
        }
    else:
        agent_result = run_patient_agent_pipeline(
            db=db,
            patient_id=patient_id,
            query=normalized,
            patient_context=patient_context,
            fallback_response=fallback_response,
            actions=actions,
            urgent_flags=urgent_flags,
            preselected_intent=routing_intent,
            compound_intent=compound_intent,
            precomputed_safety=routing_safety,
        )
    agent_result["emotional_distress"] = emotional_distress.to_dict() if emotional_distress is not None else None
    agent_result["reply"] = _apply_emotional_distress_mode(agent_result["reply"], emotional_distress)
    response = _enforce_record_provenance(agent_result["reply"], actions)
    response = _ensure_complete_response(response, fallback_response)
    response = _ensure_complete_safety_reply(response, routing_safety)
    response = _append_alert_notice(response, actions)
    agent_result["reply"] = response
    agent_result = _authorize_final_support_response(
        agent_result,
        query=normalized,
        routing_safety=routing_safety,
        deterministic_tool_confirmation=_has_tool_action(actions),
    )
    response = agent_result["reply"]
    assistant_record = ChatMessage(
        patient_id=patient_id,
        role="assistant",
        message=response,
        intent="patient_support_response",
        saved_actions_json=json.dumps({
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
        }),
    )
    db.add(assistant_record)
    db.flush()
    if high_risk_alert is not None:
        try:
            from backend.services.high_risk_conversation_alerts import attach_assistant_message

            attach_assistant_message(db, high_risk_alert.id, assistant_record.id)
        except Exception:  # noqa: BLE001 - alert linkage is secondary to preserving the chat turn
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
            "compound_intent": _compound_intent_payload(compound_intent, llm_compound_verdict),
            "emotional_distress": agent_result.get("emotional_distress"),
            "agentic_shadow": agentic_shadow,
            "evidence_envelope": agent_result.get("evidence_envelope"),
            "release_authorization": agent_result.get("release_authorization"),
        },
        "assistant_message_id": assistant_record.id,
        "safety_note": "This assistant logs and summarizes information only. It does not diagnose or give treatment instructions.",
    }


def _authorize_final_support_response(
    agent_result,
    *,
    query,
    routing_safety,
    deterministic_tool_confirmation=False,
):
    """Authorize the exact reply that the support API persists and sends.

    Evidence-dependent answers cannot be re-authorized after the outer support
    layer mutates their text because claim/citation validation applied to the
    original candidate. Deterministic support replies are rechecked from
    scratch because they do not depend on retrieved medical evidence.
    """

    if not isinstance(agent_result, dict):
        return build_fail_closed_error_result(
            query=query,
            error_code="support_result_malformed",
        )
    existing_envelope, _ = parse_evidence_envelope(agent_result.get("evidence_envelope"))
    if existing_envelope is not None and existing_envelope.evidence_required:
        return enforce_transport_release(agent_result, query=query)

    try:
        input_guardrails = input_guardrail_check(query, routing_safety or {})
        validation = agent_result.get("validation")
        if not isinstance(validation, dict) or validation.get("status") != "passed":
            agent_result["validation"] = {
                "status": "passed",
                "issues": [],
                "citation_count": 0,
                "validation_scope": "deterministic_non_evidence_support",
            }
        output_candidate = agent_result
        if deterministic_tool_confirmation:
            # The confirmation text reports a completed record action; it is
            # not a medical answer to the prior turn that supplied the data.
            # It still passes post-generation validation below.
            output_candidate = dict(agent_result)
            output_candidate["safety"] = {
                **(agent_result.get("safety") or {}),
                "level": "deterministic_tool_confirmation",
            }
        output_guardrails = output_guardrail_check(output_candidate)
        output_guardrails, _ = apply_post_gen_validator(agent_result, output_guardrails)
        agent_result["guardrails"] = {
            "input": input_guardrails,
            "output": output_guardrails,
        }
        errors = []
        if (output_guardrails or {}).get("status") != "passed":
            errors.append("support_output_guardrail_failed")
        enforce_evidence_release(
            agent_result,
            query=query,
            input_guardrails=input_guardrails,
            validation_errors=errors,
            evidence_required=False,
        )
        return enforce_transport_release(agent_result, query=query)
    except Exception as exc:  # noqa: BLE001 - alternate entry point must deny
        return build_fail_closed_error_result(
            query=query,
            error_code=f"support_final_authorization_exception:{type(exc).__name__}",
            result=agent_result,
        )


def _apply_emotional_distress_mode(reply, emotional_distress):
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


def _append_alert_notice(reply, actions):
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


def _compound_intent_payload(envelope, llm_verdict):
    """Build the agent_pipeline.compound_intent JSON.  Returns None when
    no envelope exists; otherwise the envelope plus a small ``llm``
    sub-block describing whether (and which) LLM contributed and the
    detected language, so the admin trace replay can render it."""
    if envelope is None:
        return None
    payload = envelope.to_dict()
    if llm_verdict is not None:
        payload["llm"] = {
            "available":       True,
            "language":        llm_verdict.get("language"),
            "llm_confidence":  llm_verdict.get("llm_confidence"),
            "provider":        llm_verdict.get("provider"),
            "model":           llm_verdict.get("model"),
        }
    else:
        payload["llm"] = {"available": False, "reason": "llm_unavailable_or_disabled"}
    return payload


def _tool_request_followup_message(tool_targets, casual_opener):
    """Build the user-facing follow-up prompt for a tool request that
    arrived without concrete data.  ``tool_targets`` is the list of
    save_* tool names the compound router inferred.  ``casual_opener``
    controls whether to lead with a brief greeting."""
    target_to_prompt = {
        "save_symptom": (
            "I can log a symptom — please send the symptom name AND a "
            "severity from 0–10 (e.g. \"nausea severity 6/10 today\")."
        ),
        "save_complete_cbc": (
            "I can save a CBC row — please send WBC, hemoglobin, and "
            "platelets together (e.g. \"WBC 2.1, hemoglobin 10.4, "
            "platelets 145\")."
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


def _has_tool_action(actions):
    return any(
        action.get("type") in {
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


def _should_bypass_rag_for_tool_actions(actions, routing_intent):
    if routing_intent != "data_entry_confirmation":
        return False
    return _has_tool_action(actions)


def _should_bypass_rag_for_patient_context(routing_intent, message=""):
    """Keep patient-record summaries separate from external evidence RAG.

    Confirmed portal records are the evidence for memory and timeline turns.
    Requiring knowledge-base citations for those turns both misstates the
    provenance and can replace a valid record comparison with a generic RAG
    abstention. Medical education still uses the source-governed RAG path.
    """
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


def _extract_candidate_inputs(message):
    # Multilingual preprocessing: rewrite Taglish / Spanish / typo'd
    # wording into the English shape that the existing extractors
    # already understand.  The original `message` is preserved for
    # symptom extraction (which has its own multilingual table) and
    # for downstream display; the normalized `lab_message` is used
    # only for lab + imaging + medication extraction so we don't
    # mangle anything else.
    try:
        from backend.services.multilingual_tool_router import (
            normalize_lab_value_string,
            normalize_user_text,
        )
        normalized = normalize_user_text(message)
        lab_message = normalize_lab_value_string(normalized) if normalized else message
    except Exception:  # noqa: BLE001 — never break chat on the router
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


def _resume_pending_symptom_if_possible(db, patient_id, message, extracted):
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
    pending_symptom = _latest_pending_symptom(db, patient_id)
    if not pending_symptom:
        return None
    return {
        "symptom": pending_symptom,
        "severity": severity,
        "severity_provided": True,
        "resumed_from_memory": True,
    }


def _latest_pending_symptom(db, patient_id):
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


def _deterministic_tool_plan(message, extracted, safety):
    safety = safety or {}
    safety_limited = (
        safety.get("level") in {"high_risk", "blocked"}
        or safety.get("scope") in {
            "treatment_decision_request",
            "urgent_or_safety_related",
            "diagnosis_or_outcome_claim",
        }
    )

    lower = message.lower()
    explicit_write_cues = (
        "log my", "save my", "record my", "add my", "enter my",
        "i have", "i had", "i took", "my result", "my report says",
    )
    education_only_cues = (
        "paper", "research", "study", "article", "publication",
        "knowledge base", "what does", "explain", "how does", "why ",
    )
    has_structured_candidate = any(
        extracted.get(key)
        for key in ("symptom", "labs", "partial_labs", "imaging_report", "partial_imaging", "medication")
    )
    if (
        route_intent(message, actions=[], safety=safety) == "education"
        and not any(cue in lower for cue in explicit_write_cues)
        and (not has_structured_candidate or any(cue in lower for cue in education_only_cues))
    ):
        return {
            "intent": "education",
            "selected_tools": ["none"],
            "force_tools": [],
            "source": "education_precedence",
            "confidence": 1.0,
            "reason": "education or research question lacks an explicit patient record-write request",
        }

    tools = []
    force_tools = []

    if extracted.get("symptom"):
        if _should_save_symptom(message, extracted["symptom"]):
            tools.append("save_symptom")
            force_tools.append("save_symptom")
        elif _should_request_symptom_details(message, extracted["symptom"]):
            tools.append("request_symptom_details")

    if extracted.get("labs"):
        tools.append("save_complete_cbc")
        force_tools.append("save_complete_cbc")
    elif extracted.get("partial_labs"):
        tools.append("request_missing_cbc_fields")
        if _is_short_record_hint(message):
            force_tools.append("request_missing_cbc_fields")

    if extracted.get("imaging_report"):
        tools.append("save_imaging_report")
        force_tools.append("save_imaging_report")
    elif extracted.get("partial_imaging"):
        tools.append("request_missing_imaging_details")
        if _is_short_record_hint(message):
            force_tools.append("request_missing_imaging_details")

    if extracted.get("medication"):
        tools.append("save_medication")
        force_tools.append("save_medication")

    if safety_limited:
        # A safety boundary blocks action-shaped medication requests and
        # incidental symptom mentions. Structured CBC/imaging text is still
        # allowed to create a patient confirmation and clinical-review flag;
        # neither path writes a record without an explicit follow-up confirm.
        blocked_tools = {"save_medication"}
        explicit_record_command = _has_explicit_record_command(message)
        explicit_imaging_record = bool(
            extracted.get("imaging_report")
            and re.search(r"\b(?:mri|ct|ultrasound|mammogram|imaging|scan)\b", lower)
            and re.search(r"\b(?:report|impression|findings?)\b", lower)
        )
        if not explicit_imaging_record:
            blocked_tools.update({"save_imaging_report", "request_missing_imaging_details"})
        if not explicit_record_command:
            blocked_tools.update({"save_symptom", "request_symptom_details"})
        tools = [tool for tool in tools if tool not in blocked_tools]
        force_tools = [tool for tool in force_tools if tool not in blocked_tools]

    tools = _dedupe_tools(tools) or ["none"]
    return {
        "intent": "data_entry_confirmation" if tools != ["none"] else _rough_chat_intent(message, safety),
        "selected_tools": tools,
        "force_tools": _dedupe_tools(force_tools),
        "source": "safety_filtered_extractors" if safety_limited else "deterministic_extractors",
        "confidence": 1.0 if tools != ["none"] else 0.55,
        "reason": (
            "safety-filtered local extractors; medication and incidental symptom writes are blocked"
            if safety_limited
            else "validated local extractors and record-hint heuristics"
        ),
    }


def _select_tool_plan(message, extracted, deterministic_plan, safety):
    deterministic_tools = [tool for tool in deterministic_plan.get("selected_tools", []) if tool != "none"]
    llm = {"available": False}
    if not deterministic_tools:
        # Latency optimization: only consult the LLM tool router when the
        # multilingual hint function spots tool-shaped wording.  A bare
        # greeting / identity / education / safety question has no hints,
        # so skipping the (up to 3s) Ollama HTTP call there saves real time.
        try:
            from backend.services.multilingual_tool_router import tool_intent_hints_from_text
            hints = tool_intent_hints_from_text(message)
        except Exception:  # noqa: BLE001 — never break the chat on the hint helper
            hints = []
        if hints:
            llm = select_support_tools_with_local_llm(
                message,
                deterministic_tools=deterministic_plan["selected_tools"],
                deterministic_intent=deterministic_plan["intent"],
                safety=safety,
            )
    selected = deterministic_plan["selected_tools"]
    source = deterministic_plan["source"]
    confidence = deterministic_plan["confidence"]
    reason = deterministic_plan["reason"]
    planned_intent = deterministic_plan["intent"]

    if llm.get("available") and float(llm.get("confidence") or 0) >= 0.6:
        selected = _normalize_selected_tools(llm.get("selected_tools") or llm.get("tools") or [])
        selected = _reconcile_selected_tools(selected, extracted, message)
        candidate_intent = str(llm.get("intent") or "").strip()
        if candidate_intent in ALLOWED_SUPPORT_INTENTS:
            planned_intent = candidate_intent
        source = f"llm_{llm.get('provider')}"
        confidence = float(llm.get("confidence") or 0)
        reason = llm.get("reason") or "LLM support-tool router"

    selected = _reconcile_selected_tools(selected, extracted, message)
    selected = _dedupe_tools([tool for tool in selected if tool != "none"] + deterministic_plan.get("force_tools", []))
    if _is_safety_limited_turn(safety) and not _has_explicit_record_command(message):
        selected = [
            tool
            for tool in selected
            if tool not in {"save_symptom", "request_symptom_details", "save_medication"}
        ]
    if not selected:
        selected = ["none"]
    if selected != ["none"]:
        planned_intent = "data_entry_confirmation"
    elif planned_intent == "data_entry_confirmation":
        planned_intent = _rough_chat_intent(message, safety)
    elif safety.get("scope") == "treatment_decision_request":
        planned_intent = "treatment_decision_boundary"
    elif safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        planned_intent = "safety_boundary"

    return {
        "intent": planned_intent if planned_intent in ALLOWED_SUPPORT_INTENTS else _rough_chat_intent(message, safety),
        "selected_tools": selected,
        "deterministic_tools": deterministic_plan["selected_tools"],
        "forced_tools": deterministic_plan.get("force_tools", []),
        "source": source,
        "confidence": round(confidence, 3),
        "reason": reason,
    }


def _normalize_selected_tools(raw_tools):
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    tools = []
    for tool in raw_tools or []:
        normalized = str(tool or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in ALLOWED_SUPPORT_TOOLS:
            tools.append(normalized)
    return _dedupe_tools(tools) or ["none"]


def _is_safety_limited_turn(safety):
    safety = safety or {}
    return (
        safety.get("level") in {"high_risk", "blocked"}
        or safety.get("scope") in {
            "treatment_decision_request",
            "urgent_or_safety_related",
            "diagnosis_or_outcome_claim",
        }
    )


def _has_explicit_record_command(message):
    return re.search(r"\b(?:log|save|record|add|enter)\b", str(message or "").lower()) is not None


def _portal_help_reply():
    return (
        "Use the left navigation to open Overview, Labs, Signals, Timeline, Family & Genetics, or Support. "
        "In Support, select the plus button beside the message box to open a structured form for symptoms, "
        "CBC values, imaging reports, medications, or treatment notes. Nothing is saved until you review and "
        "submit or explicitly confirm it."
    )


def _reconcile_selected_tools(selected, extracted, message):
    reconciled = []
    selected = set(selected or [])
    symptom = extracted.get("symptom")

    if "save_symptom" in selected:
        if symptom and _should_save_symptom(message, symptom):
            reconciled.append("save_symptom")
        elif symptom and _should_request_symptom_details(message, symptom):
            reconciled.append("request_symptom_details")
    if "request_symptom_details" in selected and symptom and _should_request_symptom_details(message, symptom):
        reconciled.append("request_symptom_details")

    if "save_complete_cbc" in selected:
        if extracted.get("labs"):
            reconciled.append("save_complete_cbc")
        elif extracted.get("partial_labs"):
            reconciled.append("request_missing_cbc_fields")
    if "request_missing_cbc_fields" in selected and extracted.get("partial_labs"):
        reconciled.append("request_missing_cbc_fields")

    if "save_imaging_report" in selected:
        if extracted.get("imaging_report"):
            reconciled.append("save_imaging_report")
        elif extracted.get("partial_imaging"):
            reconciled.append("request_missing_imaging_details")
    if "request_missing_imaging_details" in selected and extracted.get("partial_imaging"):
        reconciled.append("request_missing_imaging_details")

    if "save_medication" in selected and extracted.get("medication"):
        reconciled.append("save_medication")

    return _dedupe_tools(reconciled) or ["none"]


def _dedupe_tools(tools):
    seen = set()
    deduped = []
    for tool in tools or []:
        if tool == "none" and len(tools) > 1:
            continue
        if tool not in seen:
            seen.add(tool)
            deduped.append(tool)
    return deduped


def _rough_chat_intent(message, safety):
    lower = message.lower()
    if safety.get("scope") == "treatment_decision_request":
        return "treatment_decision_boundary"
    if safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        return "safety_boundary"
    if _is_conversational_prompt(message):
        return "conversation"
    if any(term in lower for term in ["remember", "what did i tell", "what did i say", "last message", "previous message", "chat history"]):
        return "patient_memory"
    if any(term in lower for term in ["last 14", "timeline", "cycle", "toxicity", "score", "my treatment", "working", "progress"]):
        return "patient_timeline_monitoring"
    if any(term in lower for term in ["upload", "site", "portal", "dashboard", "where can i", "how do i add"]):
        return "portal_help"
    if any(term in lower for term in ["pcr", "response", "mri", "cbc", "wbc", "hemoglobin", "platelets", "chemo", "chemotherapy", "side effect", "breast cancer", "neutropenia", "infection risk"]):
        return "education"
    if any(term in lower for term in ["anxious", "worried", "sad", "scared", "depressed"]):
        return "emotional_support"
    return "general_support"


from backend.services.support_chat_extraction import (
    _clinical_lab_alerts,
    _extract_complete_labs,
    _extract_date,
    _extract_imaging_report,
    _extract_medication,
    _extract_severity,
    _extract_symptom,
    _is_general_lab_question,
    _is_short_record_hint,
    _looks_like_partial_imaging,
    _looks_like_partial_labs,
    _should_request_symptom_details,
    _should_save_symptom,
)


# These three blocks are a deliberate re-export facade, not ordinary imports:
# the helpers were split out of this module and callers (including tests) still
# reach them as `support_chat_agent.<name>`. Names not referenced inside this
# file are therefore marked `noqa: F401` — removing them silently breaks those
# callers, which is exactly what happened when the rule was first enabled.
from backend.services.support_chat_safety import (
    _detect_urgent_flags,
    _enforce_record_provenance,
    _ensure_complete_response,
    _ensure_complete_safety_reply,
    _immediate_danger_reply,
    _is_immediate_danger_statement,
    _is_out_of_domain_request,
    _looks_truncated_reply,  # noqa: F401 - re-exported facade name
    _out_of_domain_reply,
    _prefer_deterministic_reply,  # noqa: F401 - re-exported facade name
    _resolve_safety_location_followup,
    _safety_location_followup_reply,
)

from backend.services.support_chat_response import (
    _build_response,
    _is_conversational_prompt,
    _is_small_talk,  # noqa: F401 - re-exported facade name
    _should_use_llm_direct_reply,
    _generate_llm_response,
)
