import json

from groq import Groq

from backend.config import get_groq_api_key, get_groq_model

SYSTEM_PROMPT = """\
You are a clinical support AI for longitudinal breast cancer monitoring.

Safety rules:
- Do not diagnose cancer, metastasis, recurrence, or treatment response.
- Do not recommend a specific treatment.
- Summarize breast imaging trends, CBC toxicity trends, treatment context, symptoms, and evidence that may need clinician review.
- State uncertainty clearly when findings come from NLP pattern matching.
- If SHAP/XAI values are present, explain them only as model behavior:
  positive SHAP pushes the model toward pCR/favorable complete response; negative SHAP pushes away from pCR.
  Do not describe SHAP features as clinically good or bad, and do not imply causality.
- Keep patient-facing language calm, simple, and non-alarming.
"""

USER_PROMPT_TEMPLATE = """\
Here is a structured patient state for longitudinal breast cancer monitoring:

{patient_state}

Return valid JSON with these keys:
- clinical_summary: doctor-facing summary of relevant longitudinal changes.
- patient_explanation: patient-friendly explanation in simple language.
- changes_since_baseline: concise list of major changes over time.
- review_reasons: concise list of reasons this case may need clinician review.
- limitations: concise list of safety limitations.

Do not include markdown fences. Do not make diagnoses.
Return strict JSON only. Do not put raw line breaks inside string values.
If SHAP/XAI is available, include the strongest toward-pCR and away-from-pCR contributors in plain language.
"""


def _fallback_summary(patient_state, reason):
    risks = patient_state.get("risk_flags", [])
    radiology = patient_state.get("radiology") or {}
    review_reasons = [risk.get("message") for risk in risks]

    for indicator in radiology.get("possible_metastatic_indicators", []):
        review_reasons.append(indicator.get("message"))

    changes = []
    if radiology:
        changes.append(
            f"CT report NLP shows tumor size status: {radiology.get('size_status', 'unknown')} "
            f"({radiology.get('percent_change')}% change when measurable)."
        )

    return {
        "clinical_summary": f"Automated LLM summary unavailable: {reason}",
        "patient_explanation": "The system can still show trends and safety flags, but the AI text summary is unavailable right now.",
        "changes_since_baseline": changes,
        "review_reasons": [item for item in review_reasons if item],
        "limitations": [
            "This system does not diagnose.",
            "Breast imaging findings are extracted from report text and require clinician interpretation.",
            "Risk flags are decision-support signals, not medical orders.",
        ],
        "summary_source": "fallback",
        "model": None,
    }


def generate_clinical_summary(patient_state):
    """Generate structured doctor and patient summaries from fused patient state."""

    api_key = get_groq_api_key()
    if not api_key:
        return _fallback_summary(patient_state, "GROQ_API_KEY environment variable is not set")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        patient_state=json.dumps(patient_state, indent=2, default=str),
    )

    try:
        model = get_groq_model()
        client = Groq(api_key=api_key, timeout=8)
        response = _request_summary_completion(client, model, user_prompt, json_mode=True)
    except Exception as exc:
        if exc.__class__.__name__ != "BadRequestError":
            return _fallback_summary(patient_state, f"LLM request failed: {exc.__class__.__name__}")
        try:
            response = _request_summary_completion(client, model, user_prompt, json_mode=False)
        except Exception as retry_exc:
            return _fallback_summary(patient_state, f"LLM request failed: {retry_exc.__class__.__name__}")

    content = response.choices[0].message.content

    parsed = _parse_summary_json(content)
    if not parsed:
        parsed = _fallback_summary(patient_state, "LLM returned non-JSON text")
        parsed["clinical_summary"] = content

    parsed.setdefault("summary_source", "groq")
    parsed.setdefault("model", get_groq_model())
    return parsed


def _request_summary_completion(client, model, user_prompt, json_mode):
    params = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**params)


def _parse_summary_json(content):
    text = str(content or "").strip()
    if not text:
        return None

    candidates = [text]
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


def _escape_control_chars_inside_json_strings(text):
    output = []
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
