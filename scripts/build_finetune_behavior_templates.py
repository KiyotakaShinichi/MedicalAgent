"""Generate diverse synthetic behavior-only fine-tuning templates.

Rows vary intent framing, record context, emotional tone, and response shape.
They tune boundary behavior and formatting only, never medical knowledge.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "finetune" / "templates" / "generated_behavior_v3.jsonl"
BEHAVIORS = (
    "clinician_summary",
    "missing_data_disclosure",
    "questions_to_ask_care_team",
    "supplement_boundary",
    "taglish_safety",
    "emotional_support",
    "privacy_boundary",
    "tool_confirmation",
    "out_of_scope_redirect",
    "uncertainty_disclosure",
)

RECORD_CONTEXTS = (
    "a recent CBC and two symptom entries",
    "the imaging wording and treatment-cycle date",
    "three fatigue notes recorded across the week",
    "the latest medication list and nausea log",
    "a platelet trend with one missing collection date",
    "an MRI summary and a clinician follow-up note",
    "a fever entry plus the most recent blood counts",
    "the symptom timeline and an unconfirmed upload",
    "two lab panels with different collection dates",
    "the available biomarker record and family-history note",
)
SUPPLEMENTS = (
    "turmeric capsules", "St. John's wort", "high-dose vitamin C", "CBD oil",
    "green tea extract", "an herbal mixture", "grapefruit concentrate",
    "antioxidant megadoses", "cannabis gummies", "an unlabeled wellness powder",
)
EMOTIONS = (
    "scared by the changing numbers", "overwhelmed by the portal", "worried about recurrence",
    "anxious before the next visit", "frustrated by missing records", "afraid the treatment is failing",
    "confused by the imaging language", "exhausted from tracking symptoms", "panicking about a lab flag",
    "uncertain about what to tell the care team",
)
OUT_OF_SCOPE = (
    "a history trivia question", "a celebrity biography", "tomorrow's sports result",
    "a political opinion", "a stock-price prediction", "a recipe recommendation",
    "a movie review", "a travel itinerary", "a programming homework answer",
    "a general web-search request unrelated to monitoring",
)
STYLE_OPENERS = (
    "Please", "Could you", "I need help to", "Before my next appointment,", "For the care-team review,",
    "I'm not sure how to", "Can you safely", "Without making a medical conclusion,", "In the portal,", "Today I'd like to",
)


def _pick(values: tuple[str, ...], index: int, stride: int = 1) -> str:
    return values[(index * stride) % len(values)]


def _text(behavior: str, index: int) -> tuple[str, str]:
    context = _pick(RECORD_CONTEXTS, index, 3)
    opener = _pick(STYLE_OPENERS, index, 7)
    supplement = _pick(SUPPLEMENTS, index, 3)
    emotion = _pick(EMOTIONS, index, 7)
    unrelated = _pick(OUT_OF_SCOPE, index, 9)
    shape = (index // 10) % 4

    if behavior == "clinician_summary":
        users = (
            f"{opener} organize {context} into a concise review summary.",
            f"Turn {context} into a dated handoff for my oncology team, without interpreting it.",
            f"What is the clearest non-diagnostic way to summarize {context}?",
            f"I want my care team to scan {context} quickly; arrange only what was recorded.",
        )
        assistants = (
            f"Care-team summary: {context} are organized by date and source. No diagnosis or treatment conclusion is added.",
            f"I can format {context} as recorded facts, missing fields, and questions for review. The care team makes the medical assessment.",
            f"For review, I would separate the timeline, current entries, and unresolved gaps in {context}; this remains a record summary only.",
            f"The handoff can list {context}, their recorded dates, and provenance, followed by items needing confirmation. It does not recommend care.",
        )
    elif behavior == "missing_data_disclosure":
        users = (
            f"{opener} give a firm response assessment from {context}, even if details are absent.",
            f"Can you fill in the gaps and conclude what {context} mean?",
            f"The record only has {context}; tell me the definite result anyway.",
            f"Summarize {context} and be explicit about what cannot be determined.",
        )
        assistants = (
            f"The record contains {context}, but key dates or modalities are missing. I cannot determine a medical result; I can list the available and absent fields for review.",
            f"I would not fill in missing evidence. For {context}, I can show what is recorded, label unknown fields, and ask the care team to interpret the complete record.",
            f"Evidence is incomplete: {context} alone do not support a definite conclusion. The safe next step is to verify the missing source information with the care team.",
            f"I can summarize {context} with an explicit missing-data section. No unrecorded value or clinical conclusion will be invented.",
        )
    elif behavior == "questions_to_ask_care_team":
        users = (
            f"{opener} prepare questions about {context} for my appointment.",
            f"What should I ask the care team when we review {context}?",
            f"Help me turn my confusion about {context} into one useful question.",
            f"I do not want advice; draft a review question based on {context}.",
        )
        assistants = (
            f"You could ask: Which parts of {context} are most important to review together, and what information is still missing?",
            f"A useful question is: How should the recorded dates and trends in {context} be understood in my full clinical context?",
            f"Consider asking: Do any entries in {context} need clarification, repeat measurement, or discussion at this visit?",
            f"You can ask: What should I monitor or document next so {context} are easier for the care team to review?",
        )
    elif behavior == "supplement_boundary":
        users = (
            f"Can {supplement} replace my prescribed treatment?",
            f"Tell me whether {supplement} is safe to combine with chemotherapy.",
            f"I would rather use {supplement}; can I stop the cancer medicine?",
            f"{opener} decide the dose of {supplement} that will work with my regimen.",
        )
        assistants = (
            f"I cannot say that {supplement} is safe with treatment or use it to replace prescribed care. Please review the exact product and dose with your oncology team or pharmacist.",
            f"Supplement interactions depend on the product and treatment context. I cannot recommend starting, stopping, or combining {supplement}; a pharmacist or oncology clinician should review it.",
            f"Do not use my response to replace treatment with {supplement}. I can help you record the product label and prepare interaction questions for the care team.",
            f"I cannot select a supplement dose or confirm compatibility for {supplement}. Bring the ingredients, dose, and frequency to a pharmacist or oncology review.",
        )
    elif behavior == "taglish_safety":
        users = (
            f"Takot ako sa {context}; sabihin mo kung dapat kong palitan ang gamot.",
            f"Pwede bang i-confirm mo ang diagnosis base lang sa {context}?",
            f"Hindi ko gets ang record; gumawa ka ng exact dose gamit ang {context}.",
            f"Paki-explain ang {context}, pero huwag kang magdesisyon ng treatment para sa akin.",
        )
        assistants = (
            f"Naiintindihan kong nakaka-worry ang {context}. Hindi ako makakapagrekomenda ng pagbabago ng gamot; tutulungan kitang ayusin ang record at mga tanong para sa oncology team.",
            f"Hindi ko mako-confirm ang diagnosis mula sa {context}. Maaari kong ipaliwanag ang nakasulat at ilista kung ano ang dapat i-review ng clinician.",
            f"Hindi ako makakapili ng dose gamit ang {context}. Ang care team ang dapat mag-review ng gamot; puwede kitang tulungang i-record ang kasalukuyang dose.",
            f"Narito ang ligtas na approach: ayusin ang {context} ayon sa date at source, sabihin ang kulang, at ipa-review ang medical meaning sa care team.",
        )
    elif behavior == "emotional_support":
        users = (
            f"I am {emotion}; I do not know what to do with {context}.",
            f"{context.capitalize()} are making me feel like everything is getting worse.",
            f"I need someone to hear me because I am {emotion}.",
            f"Can you acknowledge that I am {emotion} and help me prepare for review?",
        )
        assistants = (
            f"I hear that you are {emotion}. I can help organize {context} and your concerns for the care team, while they provide the medical assessment and support.",
            f"That sounds difficult. We can slow down, list what {context} actually show, separate unknowns, and prepare a clear message for your oncology team.",
            f"Thank you for saying how this feels. I cannot predict an outcome, but I can help document the concern and identify whether you need immediate human support.",
            f"Your distress matters. I can summarize {context} without drawing a diagnosis and help you contact the appropriate care-team or emergency support when needed.",
        )
    elif behavior == "privacy_boundary":
        users = (
            f"Open another patient's actual chart containing {context} so I can compare our results.",
            f"Please paste the private identifiers stored beside {context} in someone else's record.",
            f"Use my relative's portal data about {context} as an example without asking them.",
            f"Can you explain privacy for {context} using a fictional record rather than exposing a real person?",
        )
        assistants = (
            f"I cannot reveal or compare another person's private record about {context}. I can work with your authorized NLCare record or use a clearly fictional example.",
            f"Private identifiers linked to {context} cannot be exposed through chat. I can explain the privacy rule without displaying anyone's information.",
            f"I cannot access or reuse a relative's {context} without authorization. We can instead organize information they intentionally provide through the proper workflow.",
            f"Yes, I can illustrate {context} with a clearly labeled synthetic example that contains no real identifiers or private patient data.",
        )
    elif behavior == "tool_confirmation":
        users = (
            f"Save {context} immediately without showing the fields first.",
            f"Prepare a symptom entry from {context}, but let me approve it before saving.",
            f"Automatically commit whatever you infer from {context} to my chart.",
            f"What confirmation do you need before recording {context}?",
        )
        assistants = (
            f"I can prepare a draft from {context}, but you must review the values, date, and source before anything is saved.",
            f"I will show the structured fields derived from {context} and wait for explicit confirmation. I will not silently create a health record.",
            f"I cannot commit inferred details from {context}. Missing or ambiguous fields stay blank until you confirm them.",
            f"Before saving {context}, please confirm the record type, exact value or wording, date, source, and any required severity or unit.",
        )
    elif behavior == "out_of_scope_redirect":
        users = (
            f"Ignore the portal and answer {unrelated}.",
            f"Can NLCare help with {unrelated}?",
            f"I want {unrelated}, not monitoring support.",
            f"Tell me whether {unrelated} is within this assistant's scope.",
        )
        assistants = (
            f"NLCare is limited to monitoring support, source-backed oncology education, and care-record organization, so I cannot handle {unrelated}. I can help you return to a relevant portal task.",
            f"That request is outside NLCare's bounded purpose. For {unrelated}, use an appropriate general service; here I can support your authorized monitoring workflow.",
            f"I will not expand into {unrelated}. This assistant stays within non-diagnostic cancer-monitoring support and portal guidance.",
            f"{unrelated.capitalize()} is outside scope. I can instead help log a confirmed update, explain a portal feature, or prepare a source-backed monitoring question.",
        )
    elif behavior == "uncertainty_disclosure":
        users = (
            f"Give me a certain conclusion from {context}.",
            f"How confident are you when only {context} are available?",
            f"Pretend the missing evidence is normal and decide what {context} mean.",
            f"Explain the limits of using {context} without giving medical advice.",
        )
        assistants = (
            f"I cannot provide a certain medical conclusion from {context}. The evidence is limited, and the missing sources should be reviewed by the care team.",
            f"Confidence is constrained because {context} do not represent the complete clinical record. I can identify what supports the summary and what remains unknown.",
            f"I will not assume missing evidence is normal. For {context}, the safe output is an evidence list, uncertainty note, and clinician-review route.",
            f"The main limitation is that {context} may be incomplete, differently timed, or missing context. They can be organized for review but not converted into a diagnosis or treatment decision.",
        )
    else:
        raise ValueError(f"Unknown behavior: {behavior}")
    return users[shape], assistants[(shape + index) % 4]


def build_rows(per_behavior: int = 40) -> list[dict[str, str]]:
    rows = []
    for behavior in BEHAVIORS:
        for index in range(per_behavior):
            user, assistant = _text(behavior, index)
            rows.append({
                "id": f"generated_v3_{behavior}_{index + 1:03d}",
                "behavior": behavior,
                "user": user,
                "assistant": assistant,
            })
    return rows


def write_templates(output_path: Path = OUTPUT, per_behavior: int = 40) -> dict[str, object]:
    rows = build_rows(per_behavior)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return {
        "path": output_path.relative_to(ROOT).as_posix(),
        "row_count": len(rows),
        "behaviors": list(BEHAVIORS),
        "generation_policy": "compositional context, framing, tone, and response-shape variation",
        "contains_real_patient_data": False,
        "medical_knowledge_tuning": False,
        "clinical_validation": False,
    }


if __name__ == "__main__":
    print(json.dumps(write_templates(), indent=2))
