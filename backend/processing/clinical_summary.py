"""
Clinician summary generator — template-based, safety-validated.

Architecture (defense in depth):
1.  Compute data availability from patient_state.
2.  Build a deterministic 10-section template summary that always covers every
    mandatory section, including explicit missing-data language for any
    unavailable input.
3.  Optionally enrich the template with an LLM-generated narrative pulled from
    the same patient_state. The LLM never replaces mandatory sections.
4.  Run a post-generation safety validator that detects banned advice phrases
    ("stop chemotherapy", "increase your dose", "diagnosis confirmed", etc.).
    Detected phrases are redacted in place and recorded in safety_validator.
5.  Score completeness against a critical-info checklist and emit a confidence
    level (Low / Moderate / High) with a human-readable reason.

The deterministic template guarantees the benchmark elements (timeline anchor,
lab trend, symptom signal, imaging signal, risk flags, clinician action,
uncertainty language, non-diagnostic disclaimer) are always present, even when
the LLM is unavailable or returns junk.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from groq import Groq
except Exception:
    Groq = None

from backend.config import get_groq_api_key, get_groq_model


# ─── Safety vocabulary ────────────────────────────────────────────────────────

BANNED_PHRASES: tuple[tuple[str, str], ...] = (
    # (regex pattern, replacement)
    (r"\bstart(?:ing)?\s+(?:chemo|chemotherapy|treatment|medication)\b", "[advice redacted — requires clinician review]"),
    (r"\bstop(?:ping)?\s+(?:chemo|chemotherapy|treatment|medication)\b", "[advice redacted — requires clinician review]"),
    (r"\bincrease\s+(?:the\s+|your\s+)?dose\b", "[advice redacted — requires clinician review]"),
    (r"\bdecrease\s+(?:the\s+|your\s+)?dose\b", "[advice redacted — requires clinician review]"),
    (r"\bchange\s+(?:the\s+|your\s+)?dose\b", "[advice redacted — requires clinician review]"),
    (r"\bskip\s+(?:chemo|chemotherapy|treatment|the\s+next\s+cycle)\b", "[advice redacted — requires clinician review]"),
    (r"\bsafe\s+to\s+continue\b", "may warrant clinician review before continuing"),
    (r"\bno\s+need\s+to\s+worry\b", "flagged for follow-up"),
    (r"\bdefinitely\s+responding\b", "trend may warrant clinician review"),
    (r"\bdefinitely\s+progressing\b", "trend may warrant clinician review"),
    (r"\bdiagnosis\s+(?:is\s+)?confirmed\b", "diagnosis is not made by this system"),
    (r"\bcancer\s+detected\b", "imaging/lab signals requiring clinician interpretation"),
    (r"\bmetastasis\s+(?:is\s+)?confirmed\b", "metastasis cannot be confirmed by this system"),
    (r"\bprescribe\s+\w+\b", "[advice redacted — requires clinician review]"),
    (r"\byou\s+have\s+cancer\b", "diagnosis is not made by this system"),
    (r"\byou\s+do\s+not\s+have\s+cancer\b", "diagnosis is not made by this system"),
)

# Patterns the safety validator flags but does not auto-redact (informational).
SOFT_FLAGS: tuple[str, ...] = (
    "you should stop",
    "you should start",
    "treatment recommendation",
)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_clinical_summary(patient_state: dict) -> dict:
    """Generate structured clinician + patient summaries from fused patient state.

    Always returns a complete structured payload covering all 10 mandatory
    sections, even when the LLM is unavailable.
    """
    availability = _compute_data_availability(patient_state)
    template = _build_template_summary(patient_state, availability)

    llm_result, llm_meta = _try_llm_enrichment(patient_state)
    if llm_result:
        merged = _merge_llm_into_template(template, llm_result)
        summary_source = "template+llm"
    else:
        merged = template
        summary_source = "template"

    validated = _apply_safety_validator(merged)
    completeness = _compute_completeness(validated, availability)
    confidence = _compute_confidence(availability, completeness, validated["safety_validator"])

    return {
        "clinical_summary": validated["clinical_summary"],
        "patient_explanation": validated["patient_explanation"],
        "changes_since_baseline": validated["changes_since_baseline"],
        "review_reasons": validated["review_reasons"],
        "limitations": validated["limitations"],
        "data_availability": availability,
        "completeness_score": completeness["score"],
        "checklist": completeness["checklist"],
        "confidence_level": confidence["level"],
        "confidence_reason": confidence["reason"],
        "safety_validator": validated["safety_validator"],
        "summary_source": summary_source,
        "model": llm_meta.get("model"),
        "llm_status": llm_meta.get("status"),
    }


# ─── Data availability ────────────────────────────────────────────────────────

def _compute_data_availability(state: dict) -> dict:
    labs = state.get("latest_labs")
    baseline_labs = state.get("baseline_labs")
    symptoms = state.get("symptoms") or []
    radiology = state.get("radiology") or {}
    treatment_effects = state.get("treatment_effects") or {}
    risk_flags = state.get("risk_flags") or []

    cbc_available = bool(labs) and any(
        labs.get(field) not in (None, "") for field in ("wbc", "hemoglobin", "platelets")
    )
    baseline_cbc_available = bool(baseline_labs) and any(
        baseline_labs.get(field) not in (None, "") for field in ("wbc", "hemoglobin", "platelets")
    )
    symptoms_available = bool(symptoms)
    imaging_available = bool(radiology) and (
        radiology.get("size_status") not in (None, "", "unknown")
        or radiology.get("most_recent_report")
        or radiology.get("possible_metastatic_indicators")
    )
    medication_available = bool(treatment_effects) and bool(
        treatment_effects.get("medication_log") or treatment_effects.get("recent_medications")
    )
    treatment_cycle_available = bool(treatment_effects) and (
        treatment_effects.get("latest_cycle") is not None
        or bool(treatment_effects.get("cycles"))
    )
    risk_flags_available = bool(risk_flags)
    profile_available = bool(state.get("breast_cancer_profile"))

    return {
        "cbc_available": cbc_available,
        "baseline_cbc_available": baseline_cbc_available,
        "symptoms_available": symptoms_available,
        "imaging_available": imaging_available,
        "medication_available": medication_available,
        "treatment_cycle_available": treatment_cycle_available,
        "risk_flags_available": risk_flags_available,
        "profile_available": profile_available,
    }


# ─── Deterministic 10-section template ────────────────────────────────────────

def _build_template_summary(state: dict, availability: dict) -> dict:
    sections = []

    sections.append(_section_patient_context(state, availability))
    sections.append(_section_treatment_cycle(state, availability))
    sections.append(_section_cbc_trends(state, availability))
    sections.append(_section_symptoms(state, availability))
    sections.append(_section_imaging(state, availability))
    sections.append(_section_medications(state, availability))
    sections.append(_section_safety_flags(state, availability))
    sections.append(_section_missing_data(availability))
    sections.append(_section_review_focus(state, availability))
    sections.append(_section_non_diagnostic_disclaimer())

    clinical_summary = "Clinical Review Summary\n\n" + "\n\n".join(sections)
    patient_explanation = _build_patient_explanation(state, availability)
    changes = _build_changes_since_baseline(state, availability)
    review_reasons = _build_review_reasons(state, availability)
    limitations = _build_limitations()

    return {
        "clinical_summary": clinical_summary,
        "patient_explanation": patient_explanation,
        "changes_since_baseline": changes,
        "review_reasons": review_reasons,
        "limitations": limitations,
    }


def _section_patient_context(state: dict, availability: dict) -> str:
    patient = state.get("patient") or {}
    profile = state.get("breast_cancer_profile") or {}
    lines = ["1. Patient context"]
    diagnosis = patient.get("diagnosis") or "diagnosis not recorded"
    lines.append(f"   - Diagnosis: {diagnosis}")
    if availability["profile_available"]:
        parts = []
        if profile.get("cancer_stage"):
            parts.append(f"stage {profile['cancer_stage']}")
        for receptor in ("er_status", "pr_status", "her2_status"):
            if profile.get(receptor):
                parts.append(f"{receptor.replace('_status', '').upper()} {profile[receptor]}")
        if profile.get("molecular_subtype"):
            parts.append(f"subtype {profile['molecular_subtype']}")
        if profile.get("treatment_intent"):
            parts.append(f"intent: {profile['treatment_intent']}")
        if parts:
            lines.append(f"   - Profile: {', '.join(parts)}")
    else:
        lines.append("   - Profile: tumor profile (stage, ER/PR/HER2) is not available in the provided record.")
    return "\n".join(lines)


def _section_treatment_cycle(state: dict, availability: dict) -> str:
    lines = ["2. Treatment cycle and timeline"]
    if not availability["treatment_cycle_available"]:
        lines.append("   - Treatment cycle and date information are not available in the provided record, so timing context cannot be inferred from this summary.")
        return "\n".join(lines)
    effects = state.get("treatment_effects") or {}
    latest_cycle = effects.get("latest_cycle")
    latest_date = effects.get("latest_treatment_date")
    if latest_cycle is not None:
        lines.append(f"   - Latest recorded cycle: {latest_cycle}")
    if latest_date:
        lines.append(f"   - Latest treatment date: {latest_date}")
    cycle_count = effects.get("cycle_count") or (len(effects.get("cycles") or []))
    if cycle_count:
        lines.append(f"   - Recorded cycles to date: {cycle_count}")
    if len(lines) == 1:
        lines.append("   - Cycle/date fields present but not parseable; flagged for review.")
    return "\n".join(lines)


def _section_cbc_trends(state: dict, availability: dict) -> str:
    lines = ["3. Key CBC trends"]
    if not availability["cbc_available"]:
        lines.append("   - CBC values are not available in the provided record, so toxicity inference from blood counts is not possible from this summary.")
        return "\n".join(lines)
    latest = state.get("latest_labs") or {}
    baseline = state.get("baseline_labs") or {}
    for field, label in (("wbc", "WBC"), ("hemoglobin", "Hemoglobin"), ("platelets", "Platelets")):
        latest_val = latest.get(field)
        baseline_val = baseline.get(field) if availability["baseline_cbc_available"] else None
        if latest_val is None:
            lines.append(f"   - {label}: not recorded in latest CBC.")
            continue
        if baseline_val is None:
            lines.append(f"   - {label} latest: {latest_val} (no baseline available for trend).")
            continue
        try:
            delta = float(latest_val) - float(baseline_val)
            direction = _trend_direction(delta)
            lines.append(f"   - {label} trend: baseline {baseline_val} → latest {latest_val} ({direction}).")
        except (TypeError, ValueError):
            lines.append(f"   - {label}: baseline {baseline_val}, latest {latest_val} (trend not computable).")
    trends = state.get("lab_trends") or {}
    if trends.get("warnings"):
        lines.append(f"   - Trend warnings: {', '.join(trends['warnings'][:3])}.")
    return "\n".join(lines)


def _section_symptoms(state: dict, availability: dict) -> str:
    lines = ["4. Symptoms reported"]
    symptoms = state.get("symptoms") or []
    if not symptoms:
        lines.append("   - No patient-reported symptoms are recorded; symptom toxicity cannot be inferred from this summary.")
        return "\n".join(lines)
    top = sorted(symptoms, key=lambda s: -(s.get("severity") or 0))[:5]
    for entry in top:
        name = entry.get("symptom") or entry.get("name") or "symptom"
        severity = entry.get("severity")
        date = entry.get("date")
        suffix = f" (severity {severity})" if severity is not None else ""
        date_suffix = f" on {date}" if date else ""
        lines.append(f"   - {name}{suffix}{date_suffix}")
    return "\n".join(lines)


def _section_imaging(state: dict, availability: dict) -> str:
    lines = ["5. Imaging and MRI notes"]
    if not availability["imaging_available"]:
        lines.append("   - Imaging and MRI information is not available in the provided record, so response assessment cannot be inferred from imaging data in this summary.")
        return "\n".join(lines)
    radiology = state.get("radiology") or {}
    size_status = radiology.get("size_status")
    percent = radiology.get("percent_change")
    if size_status:
        line = f"   - Imaging trend: tumor size status {size_status}"
        if percent is not None:
            line += f" ({percent}% change where measurable)"
        line += " — extracted from report text and requires clinician interpretation."
        lines.append(line)
    indicators = radiology.get("possible_metastatic_indicators") or []
    for indicator in indicators[:3]:
        message = indicator.get("message") if isinstance(indicator, dict) else str(indicator)
        if message:
            lines.append(f"   - NLP flag: {message}")
    if len(lines) == 1:
        lines.append("   - Imaging fields present but not parseable; flagged for review.")
    return "\n".join(lines)


def _section_medications(state: dict, availability: dict) -> str:
    lines = ["6. Medications and interventions"]
    if not availability["medication_available"]:
        lines.append("   - Medication and intervention log is not available in the provided record, so treatment-response context is limited in this summary.")
        return "\n".join(lines)
    effects = state.get("treatment_effects") or {}
    meds = effects.get("recent_medications") or effects.get("medication_log") or []
    if isinstance(meds, list):
        for med in meds[:5]:
            if isinstance(med, dict):
                name = med.get("drug") or med.get("name") or "medication"
                date = med.get("date") or ""
                date_suffix = f" on {date}" if date else ""
                lines.append(f"   - {name}{date_suffix}")
            else:
                lines.append(f"   - {med}")
    if len(lines) == 1:
        lines.append("   - Medication records present but not listable in this summary.")
    return "\n".join(lines)


def _section_safety_flags(state: dict, availability: dict) -> str:
    lines = ["7. Safety flags"]
    risks = state.get("risk_flags") or []
    if not risks:
        lines.append("   - No automated risk flags raised on this record. Absence of a flag is not the same as absence of risk; clinician review is still required.")
        return "\n".join(lines)
    for risk in risks[:6]:
        if isinstance(risk, dict):
            severity = risk.get("severity") or "info"
            message = risk.get("message") or "risk flag"
            lines.append(f"   - [{severity}] {message}")
        else:
            lines.append(f"   - {risk}")
    return "\n".join(lines)


def _section_missing_data(availability: dict) -> str:
    lines = ["8. Missing or uncertain data"]
    missing = []
    if not availability["cbc_available"]:
        missing.append("CBC values are missing")
    if not availability["symptoms_available"]:
        missing.append("patient-reported symptoms are missing")
    if not availability["medication_available"]:
        missing.append("medication log is missing")
    if not availability["imaging_available"]:
        missing.append("imaging/MRI report data is missing")
    if not availability["treatment_cycle_available"]:
        missing.append("treatment cycle and date information is missing")
    if not availability["profile_available"]:
        missing.append("tumor profile (stage/ER/PR/HER2) is missing")
    if not availability["risk_flags_available"]:
        missing.append("no automated risk flags were raised, but this does not imply absence of risk")
    if not missing:
        lines.append("   - No mandatory inputs were detected as missing. Data limitations may still exist that this system does not check.")
    else:
        for item in missing:
            lines.append(f"   - {item}.")
        lines.append("   - Summary completeness is limited by the unavailable inputs listed above.")
    return "\n".join(lines)


def _section_review_focus(state: dict, availability: dict) -> str:
    lines = ["9. Suggested clinician review focus"]
    suggestions = []
    if availability["risk_flags_available"]:
        suggestions.append("review the safety flags above and confirm whether they reflect a real clinical concern")
    if availability["cbc_available"]:
        suggestions.append("verify recent CBC trend in relation to chemotherapy nadir timing")
    if availability["imaging_available"]:
        suggestions.append("read the radiology report text directly rather than relying on NLP-extracted size status")
    if availability["symptoms_available"]:
        suggestions.append("check patient-reported symptom severity for toxicity escalation patterns")
    if not availability["imaging_available"]:
        suggestions.append("obtain imaging/MRI data so response assessment can be completed")
    if not availability["medication_available"]:
        suggestions.append("populate the medication log to enable toxicity attribution")
    if not suggestions:
        suggestions.append("schedule a routine clinician review; insufficient data to recommend a specific focus area")
    for item in suggestions[:6]:
        lines.append(f"   - {item}")
    return "\n".join(lines)


def _section_non_diagnostic_disclaimer() -> str:
    return (
        "10. Non-diagnostic disclaimer\n"
        "   - This summary is generated by an engineering decision-support system. "
        "It is not a diagnosis, does not confirm or rule out cancer, recurrence, or "
        "metastasis, and does not recommend specific treatment, dose, or scheduling. "
        "All conclusions require clinician review."
    )


def _trend_direction(delta: float) -> str:
    if delta > 0.5:
        return "rising"
    if delta < -0.5:
        return "falling"
    return "stable"


# ─── Patient explanation / changes / reasons / limitations ────────────────────

def _build_patient_explanation(state: dict, availability: dict) -> str:
    parts = []
    if availability["cbc_available"]:
        parts.append("Your latest blood counts have been recorded in the system and trends are being tracked.")
    else:
        parts.append("Your blood counts (CBC) are not in the record yet, so the system cannot show CBC trends right now.")
    if availability["symptoms_available"]:
        parts.append("Reported symptoms have been logged for clinician review.")
    else:
        parts.append("No symptoms are logged. You can add new ones in the portal if needed.")
    if not availability["imaging_available"]:
        parts.append("Imaging or MRI information is not available, so response assessment cannot be inferred from imaging data.")
    parts.append("This system does not diagnose. Please follow up with your oncology care team for medical decisions.")
    return " ".join(parts)


def _build_changes_since_baseline(state: dict, availability: dict) -> list[str]:
    changes = []
    radiology = state.get("radiology") or {}
    if availability["imaging_available"] and radiology.get("size_status"):
        percent = radiology.get("percent_change")
        if percent is not None:
            changes.append(f"CT/MRI NLP reports tumor size status {radiology['size_status']} with {percent}% change where measurable.")
        else:
            changes.append(f"CT/MRI NLP reports tumor size status {radiology['size_status']}.")
    if availability["cbc_available"] and availability["baseline_cbc_available"]:
        latest = state.get("latest_labs") or {}
        baseline = state.get("baseline_labs") or {}
        for field, label in (("wbc", "WBC"), ("hemoglobin", "Hemoglobin"), ("platelets", "Platelets")):
            lv = latest.get(field)
            bv = baseline.get(field)
            try:
                delta = float(lv) - float(bv)
                changes.append(f"{label}: baseline {bv} → latest {lv} ({_trend_direction(delta)}).")
            except (TypeError, ValueError):
                continue
    if not changes:
        changes.append("Insufficient data to enumerate longitudinal changes from baseline.")
    return changes


def _build_review_reasons(state: dict, availability: dict) -> list[str]:
    reasons = []
    for risk in state.get("risk_flags") or []:
        if isinstance(risk, dict):
            msg = risk.get("message")
            if msg:
                reasons.append(msg)
    radiology = state.get("radiology") or {}
    for indicator in radiology.get("possible_metastatic_indicators") or []:
        if isinstance(indicator, dict) and indicator.get("message"):
            reasons.append(indicator["message"])
    if not availability["imaging_available"]:
        reasons.append("Imaging/MRI data is missing; clinician should request follow-up imaging to complete response assessment.")
    if not availability["cbc_available"]:
        reasons.append("CBC data is missing; clinician should review whether labs were skipped.")
    return reasons


def _build_limitations() -> list[str]:
    return [
        "This system does not diagnose, recommend treatment, or replace clinician judgment.",
        "Imaging findings are extracted from report text by NLP pattern matching and require clinician interpretation.",
        "Risk flags are decision-support signals only, not medical orders.",
        "Summary completeness depends on data uploaded to the patient record; absence of data is not absence of clinical risk.",
    ]


# ─── LLM enrichment (optional) ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical support AI for longitudinal breast cancer monitoring.

Safety rules:
- Do not diagnose cancer, metastasis, recurrence, or treatment response.
- Do not recommend a specific treatment, dose, or schedule change.
- Use uncertainty language when findings come from NLP pattern matching.
- Use phrases like "may warrant review", "requires clinician review",
  "flagged for follow-up", "insufficient information".
- Never use phrases like "start chemotherapy", "stop chemotherapy", "increase
  your dose", "decrease your dose", "safe to continue", "no need to worry",
  "definitely responding", "definitely progressing", "diagnosis confirmed".
- Keep patient-facing language calm, simple, and non-alarming.
"""

USER_PROMPT_TEMPLATE = """\
Here is a structured patient state for longitudinal breast cancer monitoring:

{patient_state}

Return strict JSON only (no markdown fences, no raw line breaks inside string values) with these keys:
- narrative_clinical: a 3-5 sentence narrative supplement to the structured clinical summary, using uncertainty language.
- narrative_patient: a calm patient-friendly paraphrase, 2-3 sentences.
- additional_review_reasons: optional list of additional clinician review reasons not already obvious from risk flags.
"""


def _try_llm_enrichment(patient_state: dict) -> tuple[dict | None, dict]:
    """Try to enrich the template with LLM narrative. Returns (result, meta)."""
    if Groq is None:
        return None, {"status": "groq_not_installed", "model": None}
    api_key = get_groq_api_key()
    if not api_key:
        return None, {"status": "no_api_key", "model": None}
    model = get_groq_model()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        patient_state=json.dumps(patient_state, indent=2, default=str),
    )
    try:
        client = Groq(api_key=api_key, timeout=8)
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        return None, {"status": f"llm_error:{exc.__class__.__name__}", "model": model}

    content = response.choices[0].message.content if response.choices else None
    parsed = _parse_summary_json(content)
    if not parsed:
        return None, {"status": "non_json_response", "model": model}
    return parsed, {"status": "ok", "model": model}


def _merge_llm_into_template(template: dict, llm: dict) -> dict:
    """Add LLM narrative as a supplementary section. Never replace mandatory sections."""
    narrative_clinical = (llm.get("narrative_clinical") or "").strip()
    narrative_patient = (llm.get("narrative_patient") or "").strip()
    extra_reasons = llm.get("additional_review_reasons") or []

    merged = dict(template)
    if narrative_clinical:
        merged["clinical_summary"] = (
            template["clinical_summary"]
            + "\n\n11. Supplementary narrative (LLM-generated, advisory only)\n   - "
            + narrative_clinical
        )
    if narrative_patient:
        merged["patient_explanation"] = template["patient_explanation"] + " " + narrative_patient
    if isinstance(extra_reasons, list):
        for reason in extra_reasons:
            if isinstance(reason, str) and reason.strip():
                merged.setdefault("review_reasons", list(template["review_reasons"]))
                merged["review_reasons"].append(reason.strip())
    return merged


# ─── Safety validator ─────────────────────────────────────────────────────────

def _apply_safety_validator(summary: dict) -> dict:
    text_fields = ("clinical_summary", "patient_explanation")
    list_fields = ("changes_since_baseline", "review_reasons", "limitations")

    detected: list[dict] = []
    soft_flags: list[dict] = []
    output = dict(summary)

    for field in text_fields:
        original = output.get(field) or ""
        cleaned, hits = _redact_banned(original)
        for hit in hits:
            detected.append({"field": field, **hit})
        for soft in _detect_soft_flags(original):
            soft_flags.append({"field": field, "phrase": soft})
        output[field] = cleaned

    for field in list_fields:
        items = output.get(field) or []
        new_items = []
        for index, item in enumerate(items):
            if not isinstance(item, str):
                new_items.append(item)
                continue
            cleaned, hits = _redact_banned(item)
            for hit in hits:
                detected.append({"field": f"{field}[{index}]", **hit})
            for soft in _detect_soft_flags(item):
                soft_flags.append({"field": f"{field}[{index}]", "phrase": soft})
            new_items.append(cleaned)
        output[field] = new_items

    output["safety_validator"] = {
        "status": "blocked_unsafe_advice" if detected else "passed",
        "redactions": detected,
        "soft_flags": soft_flags,
        "policy_version": "clinician_summary_safety_v1",
    }
    return output


def _redact_banned(text: str) -> tuple[str, list[dict]]:
    hits: list[dict] = []
    result = text
    for pattern, replacement in BANNED_PHRASES:
        matches = list(re.finditer(pattern, result, flags=re.IGNORECASE))
        if not matches:
            continue
        for match in matches:
            hits.append({"pattern": pattern, "matched": match.group(0)})
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result, hits


def _detect_soft_flags(text: str) -> list[str]:
    lower = text.lower()
    return [phrase for phrase in SOFT_FLAGS if phrase in lower]


# ─── Completeness & confidence ────────────────────────────────────────────────

CRITICAL_CHECKLIST = (
    "latest_cbc_values",
    "trend_direction",
    "urgent_symptoms",
    "imaging_status",
    "medication_or_treatment_cycle",
    "risk_flags",
    "missing_data_statement",
    "uncertainty_statement",
    "clinician_review_recommendation",
)


def _compute_completeness(summary: dict, availability: dict) -> dict:
    text = (summary.get("clinical_summary") or "").lower()
    checklist: dict[str, bool] = {}

    checklist["latest_cbc_values"] = any(token in text for token in ("wbc", "hemoglobin", "platelet"))
    checklist["trend_direction"] = any(token in text for token in ("rising", "falling", "stable", "baseline", "trend"))
    checklist["urgent_symptoms"] = any(token in text for token in ("symptom", "fever", "fatigue", "nausea", "pain", "no patient-reported symptoms"))
    checklist["imaging_status"] = any(token in text for token in ("mri", "imaging", "radiology", "tumor", "ct"))
    checklist["medication_or_treatment_cycle"] = any(token in text for token in ("cycle", "medication", "treatment", "intervention"))
    checklist["risk_flags"] = any(token in text for token in ("flag", "risk", "alert", "urgent", "no automated risk flags"))
    checklist["missing_data_statement"] = any(
        token in text for token in ("missing", "not available", "not recorded", "insufficient", "unavailable", "no patient-reported")
    )
    checklist["uncertainty_statement"] = any(
        token in text for token in (
            "may warrant", "requires clinician review", "flagged for follow-up",
            "not enough data", "cannot be inferred", "limited", "requires clinician interpretation",
        )
    )
    checklist["clinician_review_recommendation"] = any(
        token in text for token in ("clinician review", "review focus", "follow-up", "follow up", "schedule a")
    )

    hits = sum(1 for value in checklist.values() if value)
    score = round(hits / len(checklist), 3)
    return {"score": score, "checklist": checklist}


def _compute_confidence(availability: dict, completeness: dict, safety_validator: dict) -> dict:
    score = completeness["score"]
    missing_count = sum(1 for value in availability.values() if not value)
    reasons = []

    if safety_validator.get("status") != "passed":
        reasons.append("post-generation safety validator redacted unsafe phrasing")

    if not availability["imaging_available"]:
        reasons.append("imaging/MRI data missing")
    if not availability["cbc_available"]:
        reasons.append("CBC values missing")
    if not availability["symptoms_available"]:
        reasons.append("patient symptoms missing")

    if safety_validator.get("status") != "passed" or score < 0.6 or missing_count >= 4:
        level = "Low"
    elif score < 0.85 or missing_count >= 2:
        level = "Moderate"
    else:
        level = "High"

    if not reasons:
        reasons.append("all checklist items satisfied and mandatory inputs present")

    return {"level": level, "reason": "; ".join(reasons)}


# ─── JSON parsing helpers (kept for backwards compatibility) ──────────────────

def _parse_summary_json(content: Any) -> dict | None:
    text = str(content or "").strip()
    if not text:
        return None

    candidates: list[str] = [text]
    if text.startswith("```"):
        stripped = text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        for repaired in (candidate, _escape_control_chars_inside_json_strings(candidate)):
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _escape_control_chars_inside_json_strings(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in str(text or ""):
        if in_string:
            if escaped:
                output.append(char)
                escaped = False
            elif char == "\\":
                output.append(char)
                escaped = True
            elif char == '"':
                output.append(char)
                in_string = False
            elif char == "\n":
                output.append("\\n")
            elif char == "\r":
                output.append("\\r")
            elif char == "\t":
                output.append("\\t")
            elif ord(char) < 32:
                continue
            else:
                output.append(char)
        else:
            output.append(char)
            if char == '"':
                in_string = True
    return "".join(output)
