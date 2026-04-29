from dotenv import load_dotenv
import os
import json
from groq import Groq


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """\
You are a clinical support AI. You do NOT diagnose. You summarize lab trends, \
detected risks, and treatment effects for medical professionals and patients.

Rules:
- Do NOT diagnose or recommend specific treatments.
- Focus on trends over time, not single data points.
- Mention chemotherapy context when relevant.
- Be concise but thorough.
"""

USER_PROMPT_TEMPLATE = """\
Here is a patient's longitudinal clinical data:

## Lab Trends
{trends}

## Detected Risks
{risks}

## Treatment Effects
{treatment_effects}

---

Please provide TWO sections:

### 1. Clinical Summary (Doctor-Level)
A professional-grade summary suitable for a physician reviewing this patient's \
chart. Use precise clinical language. Highlight concerning trends and their \
potential relationship to the treatment regimen.

### 2. Simple Explanation (Patient-Level)
A clear, reassuring, jargon-free explanation suitable for the patient or their \
family. Explain what the numbers mean in plain language and what to watch for.
"""


def generate_clinical_summary(
    trends,
    risks,
    treatment_effects,
    trend_risks=None,
    radiology_summary=None,
):
    """Generate a two-part clinical summary using Groq.

    Args:
        trends: dict of lab trend directions (e.g. {"wbc": "decreasing"})
        risks: list of risk strings from detect_risks
        treatment_effects: list of dicts from align_labs_with_treatment

    Returns:
        str: The generated clinical summary text.
    """

    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "[ERROR] GROQ_API_KEY environment variable is not set.\n"
            "Set it with:  set GROQ_API_KEY=your-key-here  (Windows)\n"
            "              export GROQ_API_KEY=your-key-here (Linux/Mac)"
        )

    client = Groq(api_key=api_key)

    # Format the data for the prompt
    merged_risks = risks
    if trend_risks:
        merged_risks = risks + trend_risks

    trends_str = json.dumps(trends, indent=2)
    risks_str = "\n".join(f"- {r}" for r in merged_risks) if merged_risks else "No risks detected."
    effects_str = json.dumps(treatment_effects, indent=2)
    radiology_str = (
        json.dumps(radiology_summary, indent=2)
        if radiology_summary is not None
        else "No radiology summary available."
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        trends=trends_str,
        risks=risks_str,
        treatment_effects=effects_str,
    )
    user_prompt += f"\n\n## Radiology Summary\n{radiology_str}\n"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content
