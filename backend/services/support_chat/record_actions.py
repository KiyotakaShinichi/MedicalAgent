from backend.processing.radiology_analysis import detect_possible_metastatic_indicators
from backend.services.agent_rag import safety_scope_check
from backend.services.confirmed_record_write import queue_record_write
from backend.services.conversation_state import clear_pending_action, set_pending_action
from backend.services.input_validation import (
    validate_cbc_values,
    validate_imaging_report_payload,
    validate_symptom_payload,
)
from backend.services.support_chat.response_helpers import tool_request_followup_message
from backend.services.support_chat_extraction import _clinical_lab_alerts, _extract_date


def apply_record_actions(
    *,
    db,
    patient_id,
    normalized,
    extracted,
    selected_tools,
    actions,
    urgent_flags,
    terminal_input_block,
    tool_plan,
    compound_intent,
    user_record,
    routing_safety,
    prior_user_messages,
):
    symptom = extracted["symptom"]
    if "save_symptom" in selected_tools and symptom:
        if not symptom.get("severity_provided") or symptom["severity"] is None:
            actions.append(
                {
                    "type": "partial_symptom_detected",
                    "symptom": symptom["symptom"],
                    "message": (
                        f"I noticed you mentioned {symptom['symptom']}, but I need a severity "
                        f"before I can save it. Please send it on a 0-10 scale, for example: "
                        f'"{symptom["symptom"]} severity 6/10 today".'
                    ),
                }
            )
            set_pending_action(
                patient_id,
                "symptom_save",
                {
                    "type": "partial_symptom_detected",
                    "symptom": symptom["symptom"],
                    "source": "support_chat_agent",
                },
            )
        else:
            severity = int(symptom["severity"])
            try:
                validate_symptom_payload(symptom["symptom"], severity)
                actions.append(
                    queue_record_write(
                        patient_id,
                        "symptom",
                        {
                            "date": _extract_date(normalized),
                            "symptom": symptom["symptom"],
                            "severity": severity,
                        },
                        source_message=normalized,
                        source_chat_message_id=user_record.id,
                    )
                )
                clear_pending_action(patient_id, "symptom_save")
            except Exception as exc:
                actions.append(
                    {
                        "type": "symptom_save_failed",
                        "symptom": symptom["symptom"],
                        "severity": severity,
                        "reason": str(exc)[:200],
                        "message": (
                            f"I couldn't save the {symptom['symptom']} entry just now — "
                            f"there was a problem with the record. Please try again or "
                            f"log it from the portal manually."
                        ),
                    }
                )
    elif "request_symptom_details" in selected_tools and symptom:
        actions.append(
            {
                "type": "partial_symptom_detected",
                "symptom": symptom["symptom"],
                "message": (
                    f"I noticed you mentioned {symptom['symptom']}. If you want me to log it, "
                    f"send the severity from 0-10, for example: "
                    f'"{symptom["symptom"]} severity 6/10 today".'
                ),
            }
        )
        set_pending_action(
            patient_id,
            "symptom_save",
            {
                "type": "partial_symptom_detected",
                "symptom": symptom["symptom"],
                "source": "support_chat_agent",
            },
        )

    labs = extracted["labs"]
    if "save_complete_cbc" in selected_tools and labs:
        validate_cbc_values(labs["wbc"], labs["hemoglobin"], labs["platelets"])
        lab_alerts = _clinical_lab_alerts(labs)
        actions.append(
            queue_record_write(
                patient_id,
                "cbc",
                {"date": _extract_date(normalized), **labs},
                source_message=normalized,
                source_chat_message_id=user_record.id,
            )
        )
        if lab_alerts:
            actions.append(
                {
                    "type": "clinical_rule_alert",
                    "alerts": lab_alerts,
                    "message": "CBC safety rule triggered before RAG retrieval.",
                }
            )
            urgent_flags.extend([alert["rule"] for alert in lab_alerts])
    elif "request_missing_cbc_fields" in selected_tools and extracted["partial_labs"]:
        actions.append(
            {
                "type": "partial_labs_detected",
                "message": "I saw lab information, but I need WBC, hemoglobin, and platelets together to save a CBC row.",
            }
        )

    imaging_report = extracted["imaging_report"]
    if "save_imaging_report" in selected_tools and imaging_report:
        validate_imaging_report_payload(
            imaging_report["modality"],
            imaging_report["report_type"],
            imaging_report["findings"],
            imaging_report["impression"],
            body_site=imaging_report["body_site"],
        )
        actions.append(
            queue_record_write(
                patient_id,
                "imaging",
                imaging_report,
                source_message=normalized,
                source_chat_message_id=user_record.id,
            )
        )
        indicators = detect_possible_metastatic_indicators(
            f"{imaging_report['findings']} {imaging_report['impression']}"
        )
        if indicators:
            sites = sorted({indicator["site"] for indicator in indicators})
            actions.append(
                {
                    "type": "possible_metastatic_indicator",
                    "sites": sites,
                    "message": (
                        "Report wording mentions possible distant-disease indicators. "
                        "This is not a diagnosis and should be reviewed by the oncology team."
                    ),
                }
            )
            urgent_flags.extend([f"imaging_{site}" for site in sites])
    elif "request_missing_imaging_details" in selected_tools and extracted["partial_imaging"]:
        actions.append(
            {
                "type": "partial_imaging_detected",
                "message": "I saw imaging wording. To save it as a report, paste the report date plus findings or impression text.",
            }
        )

    medication = extracted["medication"]
    if "save_medication" in selected_tools and medication:
        actions.append(
            queue_record_write(
                patient_id,
                "medication",
                {"date": _extract_date(normalized), **medication},
                source_message=normalized,
                source_chat_message_id=user_record.id,
            )
        )

    has_concrete_save = any(
        action.get("type", "").startswith("saved_")
        or action.get("type", "").startswith("partial_")
        or action.get("type") == "pending_record_confirmation"
        for action in actions
    )
    if (
        compound_intent is not None
        and compound_intent.has_tool_request
        and not has_concrete_save
        and not urgent_flags
        and not terminal_input_block
        and tool_plan.get("intent") != "portal_help"
    ):
        actions.append(
            {
                "type": "partial_tool_request_detected",
                "tool_targets": compound_intent.tool_request_targets,
                "casual_opener": compound_intent.has_casual_opener,
                "suggested_acknowledgment": compound_intent.suggested_acknowledgment,
                "message": tool_request_followup_message(
                    compound_intent.tool_request_targets,
                    compound_intent.has_casual_opener,
                ),
            }
        )

    if urgent_flags:
        routing_safety = safety_scope_check(
            normalized,
            urgent_flags,
            previous_user_messages=prior_user_messages,
        )
    return actions, urgent_flags, routing_safety
