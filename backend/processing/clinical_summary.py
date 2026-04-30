from dotenv import load_dotenv
import json
import os

from groq import Groq


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """\
You are a clinical support AI for longitudinal oncology monitoring.

Safety rules:
- Do not diagnose cancer, metastasis, recurrence, or treatment response.
- Do not recommend a specific treatment.
- Summarize trends, risk flags, and evidence that may need clinician review.
- State uncertainty clearly when findings come from NLP pattern matching.
- Keep patient-facing language calm, simple, and non-alarming.
"""

USER_PROMPT_TEMPLATE = """\
Here is a structured patient state for longitudinal oncology monitoring:

{patient_state}

Return valid JSON with these keys:
- clinical_summary: doctor-facing summary of relevant longitudinal changes.
- patient_explanation: patient-friendly explanation in simple language.
- changes_since_baseline: concise list of major changes over time.
- review_reasons: concise list of reasons this case may need clinician review.
- limitations: concise list of safety limitations.

Do not include markdown fences. Do not make diagnoses.
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
            "Radiology findings are extracted from report text and require clinician interpretation.",
            "Risk flags are decision-support signals, not medical orders.",
        ],
    }


def generate_clinical_summary(patient_state):
    """Generate structured doctor and patient summaries from fused patient state."""

    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _fallback_summary(patient_state, "GROQ_API_KEY environment variable is not set")

    client = Groq(api_key=api_key)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        patient_state=json.dumps(patient_state, indent=2, default=str),
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = _fallback_summary(patient_state, "LLM returned non-JSON text")
        parsed["clinical_summary"] = content

    return parsed
