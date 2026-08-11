"""Patient-support response composition and bounded LLM rephrasing.

This module does not decide routes or execute writes. It renders deterministic
actions and may use the configured LLM only to rephrase a precomputed safe
fallback; provenance and truncation guards remain fail-closed.
"""
from __future__ import annotations

import json
import re

from groq import Groq

from backend.config import get_groq_api_key, get_groq_model, get_llm_adjudication_enabled
from backend.services.llm_telemetry import LLMCallTimer, provider_usage, record_llm_call
from backend.services.support_chat_policy import CHAT_SYSTEM_PROMPT
from backend.services.support_chat_safety import _looks_truncated_reply
from backend.services.local_llm import (
    provider_circuit_status,
    record_provider_failure,
    record_provider_success,
)

def _build_response(message, actions, urgent_flags, patient_context):
    if not actions and not urgent_flags and _is_conversational_prompt(message):
        return _conversation_reply(patient_context)

    parts = []
    if urgent_flags:
        parts.append(
            "I noticed possible urgent wording. If symptoms feel severe, sudden, or unsafe, contact your oncology team or local emergency services now."
        )

    saved = [action for action in actions if action["type"].startswith("saved_")]
    failed = [action for action in actions if action["type"].endswith("_save_failed")]
    partial_actions = [action for action in actions if action["type"].startswith("partial_")]
    pending_confirmations = [action for action in actions if action["type"] == "pending_record_confirmation"]
    cancellations = [action for action in actions if action["type"] == "record_write_cancelled"]
    duplicates = [action for action in actions if action["type"] == "duplicate_record_prevented"]
    if pending_confirmations:
        previews = "; ".join(action.get("preview", "record") for action in pending_confirmations)
        parts.append(
            "I prepared this record preview: " + previews + ". Nothing has been saved yet. "
            "Choose Confirm save to add it, or Cancel to leave your record unchanged."
        )
    elif cancellations:
        parts.append("Cancelled. No patient record was changed.")
    elif duplicates:
        parts.append(
            "I did not create a duplicate because the same active portal entry already exists. "
            "You can review the existing entry in the relevant dashboard section."
        )
    elif saved:
        labels = []
        for action in saved:
            if action["type"] == "saved_symptom":
                labels.append(f"symptom: {action['symptom']} severity {action['severity']}/10")
            elif action["type"] == "saved_labs":
                labels.append("CBC values")
            elif action["type"] == "saved_medication":
                labels.append(f"medication: {action['medication']}")
            elif action["type"] == "saved_imaging_report":
                labels.append(f"{action['modality']} report from {action['date']}")
        parts.append("I saved this to your patient record: " + "; ".join(labels) + ".")
        change = patient_context.get("record_change_explanation") or {}
        if change.get("patient_summary"):
            summary = str(change["patient_summary"])
            parts.append(
                "After this confirmed update, "
                + summary[:1].lower()
                + summary[1:]
            )
    if failed:
        parts.extend(action["message"] for action in failed if action.get("message"))
    if not saved and not failed and not pending_confirmations and not cancellations and not duplicates:
        if partial_actions:
            parts.extend(action["message"] for action in partial_actions if action.get("message"))
        else:
            contextual = _contextual_reply(message, patient_context)
            parts.append(contextual or "I heard you. I can chat normally, answer low-risk education questions with sources when needed, or help log symptoms, CBC values, medications, and imaging report text.")

    partial_labs = [action for action in actions if action["type"] == "partial_labs_detected"]
    if partial_labs and saved:
        parts.append(partial_labs[0]["message"])

    partial_imaging = [action for action in actions if action["type"] == "partial_imaging_detected"]
    if partial_imaging and saved:
        parts.append(partial_imaging[0]["message"])

    lab_alerts = [action for action in actions if action["type"] == "clinical_rule_alert"]
    if lab_alerts:
        labels = [
            f"{alert['label']} {alert['value']} ({alert['severity']}, threshold {alert['threshold']})"
            for alert in lab_alerts[0]["alerts"]
        ]
        parts.append(
            "A deterministic CBC safety rule flagged this for clinician review: "
            + "; ".join(labels)
            + ". Please contact your oncology care team for medical guidance."
        )

    imaging_alerts = [action for action in actions if action["type"] == "possible_metastatic_indicator"]
    if imaging_alerts:
        sites = ", ".join(imaging_alerts[0].get("sites") or ["unspecified"])
        parts.append(
            "The imaging text includes wording that may need clinician review "
            f"({sites}). I am only logging the report text and cannot diagnose metastasis."
        )

    parts.append(
        "I can help track what you are feeling and summarize it for review, but I cannot diagnose or decide treatment."
    )
    return " ".join(parts)


def _is_small_talk(message):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    small_talk = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "kumusta",
        "kamusta",
        "thanks",
        "thank you",
        "salamat",
    }
    return cleaned in small_talk or cleaned.startswith(("hi ", "hello ", "hey "))


def _is_conversational_prompt(message):
    return _is_small_talk(message) or _is_identity_or_capability_question(message) or _is_social_checkin(message)


def _is_identity_or_capability_question(message):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    patterns = [
        "who are you",
        "what are you",
        "what can you do",
        "what do you do",
        "how can you help",
        "help me",
        "can you help",
        "are you a doctor",
        "are you ai",
        "are you an ai",
    ]
    return any(pattern in cleaned for pattern in patterns)


def _is_social_checkin(message):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    patterns = [
        "how are you",
        "how are u",
        "how you doing",
        "how are you doing",
        "are you ok",
        "what s up",
        "whats up",
    ]
    return any(pattern in cleaned for pattern in patterns)


def _conversation_reply(patient_context):
    memory_hint = _latest_memory_hint(patient_context)
    base = (
        "I am the portal support agent for this breast cancer monitoring demo. "
        "I can chat normally, remember recent patient-scoped messages, log symptoms, save complete CBC values, "
        "record medications, and save report-like MRI/imaging notes for clinician review."
    )
    if memory_hint:
        base += f" I can also refer back to recent portal context, like {memory_hint}."
    base += " I cannot diagnose or choose treatment."
    return base


def _should_use_llm_direct_reply(intent, safety, actions, urgent_flags):
    if urgent_flags or safety.get("level") == "high_risk":
        return False
    if intent == "data_entry_confirmation":
        return bool(actions)
    return intent in {
        "conversation",
        "patient_memory",
        "patient_timeline_monitoring",
        "general_support",
        "emotional_support",
    }


def _generate_llm_response(message, actions, urgent_flags, patient_context, fallback_response):
    if not get_llm_adjudication_enabled():
        return fallback_response
    api_key = get_groq_api_key()
    if not api_key:
        return fallback_response
    if provider_circuit_status("groq").get("open"):
        return fallback_response

    user_prompt = {
        "patient_message": message,
        "saved_actions": actions,
        "urgent_flags": urgent_flags,
        "recent_context": patient_context,
        "fallback_reply": fallback_response,
    }
    model = get_groq_model()
    prompt_json = json.dumps(user_prompt, default=str)
    timer = LLMCallTimer.start()
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=220,
            messages=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_json},
            ],
        )
        choice = completion.choices[0]
        reply = (choice.message.content or "").strip()
        record_llm_call(
            provider="groq",
            model=model,
            operation="patient_support_answer",
            latency_ms=timer.elapsed_ms(),
            prompt_parts=[CHAT_SYSTEM_PROMPT, prompt_json],
            completion_text=reply,
            usage=provider_usage(completion),
        )
        if getattr(choice, "finish_reason", None) not in {None, "stop"}:
            return fallback_response
    except Exception as exc:
        record_provider_failure("groq", exc.__class__.__name__)
        record_llm_call(
            provider="groq",
            model=model,
            operation="patient_support_answer",
            latency_ms=timer.elapsed_ms(),
            prompt_parts=[CHAT_SYSTEM_PROMPT, prompt_json],
            success=False,
            error_type=exc.__class__.__name__,
        )
        return fallback_response

    record_provider_success("groq")
    if not reply or _looks_truncated_reply(reply):
        return fallback_response
    if urgent_flags and "emergency" not in reply.lower() and "oncology" not in reply.lower():
        return fallback_response
    # Honest-save guard: the LLM is allowed to rephrase the deterministic
    # fallback, but it must never claim to have logged/saved/recorded something
    # when no save action actually succeeded.  If the LLM hallucinates a save,
    # discard its output and return the deterministic reply.
    saved_count = sum(1 for action in actions if str(action.get("type", "")).startswith("saved_"))
    if saved_count == 0:
        lower_reply = reply.lower()
        save_claim_terms = (
            "i logged", "i've logged", "i have logged",
            "i saved", "i've saved", "i have saved",
            "i recorded", "i've recorded", "i have recorded",
            "i added", "i've added", "i have added",
            "added to your record", "saved to your record", "logged to your record",
        )
        if any(term in lower_reply for term in save_claim_terms):
            return fallback_response
    if _is_identity_or_capability_question(message) and "support" not in reply.lower():
        return fallback_response
    return reply


from backend.services.support_chat_context import (
    _chat_toxicity_summary,
    _last_14_day_changes,
    _recent_patient_context,
    _synthetic_model_context,
    _timeline_context,
)

def _contextual_reply(message, context):
    lower = message.lower()
    status_terms = ["how am i", "how is my treatment", "am i improving", "working", "progress", "score"]
    explain_terms = ["why", "explain", "factor", "contribute", "model"]
    doctor_terms = ["doctor", "oncologist", "tell them", "bring", "ask"]
    timeline_terms = ["last 14", "last fourteen", "what changed", "timeline", "tumor board", "toxicity", "cycle 2"]
    memory_terms = ["remember", "what did i tell", "what did i say", "last message", "previous message", "chat history"]

    if any(term in lower for term in memory_terms):
        return _memory_reply(message, context)

    if any(term in lower for term in timeline_terms):
        timeline = context.get("timeline_context") or {}
        if "tumor board" in lower:
            return timeline.get("tumor_board_brief")
        if "toxicity" in lower or "cycle 2" in lower:
            return timeline.get("toxicity_summary")
        if "last 14" in lower or "last fourteen" in lower or "what changed" in lower:
            changes = timeline.get("last_14_day_changes") or []
            if changes:
                return "In the last represented 14 days: " + "; ".join(changes[:5]) + "."
            return "I do not see timeline events in the last represented 14-day window. This may mean the record is missing recent updates."

    if any(term in lower for term in status_terms):
        change = context.get("record_change_explanation") or {}
        observations = change.get("observations") or []
        detail = " ".join(str(item.get("summary") or "") for item in observations[:3]).strip()
        summary = change.get("patient_summary") or (
            "The portal does not have enough comparable dated records to summarize a change. "
            "This does not show whether treatment is working."
        )
        return f"{summary} {detail}".strip()

    if any(term in lower for term in explain_terms):
        xai = context.get("synthetic_model_explanation") or {}
        positives = xai.get("positive_contributions") or []
        negatives = xai.get("negative_contributions") or []
        if positives or negatives:
            toward = ", ".join(item["feature"] for item in positives[:3]) or "none"
            away = ", ".join(item["feature"] for item in negatives[:3]) or "none"
            return (
                f"The demo explanation says features pushing toward the higher synthetic class include: {toward}. "
                f"Features pushing toward the lower synthetic class include: {away}. "
                "These explain model behavior on synthetic data, not medical causality."
            )

    if any(term in lower for term in doctor_terms):
        latest_lab = context.get("latest_lab")
        interventions = context.get("recent_interventions") or []
        symptoms = context.get("recent_symptoms") or []
        items = []
        if latest_lab:
            items.append(f"latest CBC WBC {latest_lab.get('wbc')}, hemoglobin {latest_lab.get('hemoglobin')}, platelets {latest_lab.get('platelets')}")
        if symptoms:
            items.append(f"recent symptom {symptoms[0].get('symptom')} severity {symptoms[0].get('severity')}/10")
        if interventions:
            items.append(f"recent support intervention: {interventions[0].get('type')}")
        if items:
            return "For your care team, bring up " + "; ".join(items) + "."

    return None


def _memory_reply(message, context):
    current = message.strip().lower()
    recent_chat = context.get("recent_chat") or []
    previous_user_messages = []
    seen = set()
    for item in recent_chat:
        if item.get("role") != "user":
            continue
        text = item.get("message", "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if not text or key == current or key in seen or _is_small_talk(text):
            continue
        seen.add(key)
        previous_user_messages.append(text)
    if previous_user_messages:
        snippets = [text.strip() for text in previous_user_messages[-3:] if text.strip()]
        return (
            "From this chat, the recent things you told me were: "
            + " | ".join(snippets)
            + ". I use this only as portal memory for tracking and review, not diagnosis."
        )

    hint = _latest_memory_hint(context)
    if hint:
        return f"I do not see earlier chat notes yet, but your portal context includes {hint}. This is for tracking and clinician review only."
    return "I do not see earlier chat details for this patient yet. Tell me symptoms, CBC values, medications, or imaging report text and I can help log them."


def _latest_memory_hint(context):
    latest_lab = context.get("latest_lab")
    symptoms = context.get("recent_symptoms") or []
    imaging = context.get("recent_imaging") or []
    medications = context.get("recent_medications") or []
    if latest_lab:
        return f"latest CBC WBC {latest_lab.get('wbc')}, hemoglobin {latest_lab.get('hemoglobin')}, platelets {latest_lab.get('platelets')}"
    if symptoms:
        return f"recent symptom {symptoms[0].get('symptom')} severity {symptoms[0].get('severity')}/10"
    if imaging:
        return f"recent imaging note: {imaging[0].get('impression')}"
    if medications:
        return f"recent medication: {medications[0].get('medication')}"
    return None
