from __future__ import annotations

SYMPTOM_KEYWORDS = {
    "fatigue": ["fatigue", "tired", "weak", "exhausted"],
    "nausea": ["nausea", "nauseous", "vomit", "vomiting"],
    "abdominal discomfort": ["upset stomach", "stomach discomfort", "tummy discomfort", "masama ang tiyan", "kabag"],
    "pain": ["pain", "ache", "aching", "sore"],
    "bloody discharge": ["blood discharge", "bloody discharge", "bleeding discharge", "blood-stained discharge"],
    "bleeding": ["bleeding", "blood loss", "spotting"],
    "discharge": ["discharge"],
    "fever": ["fever", "temperature", "chills"],
    "shortness of breath": ["shortness of breath", "breathless", "difficulty breathing"],
    "neuropathy": ["neuropathy", "tingling", "numbness"],
    "low appetite": ["low appetite", "no appetite", "not eating"],
    "anxiety": ["anxiety", "anxious", "scared", "worried", "panic"],
    "sadness": ["sad", "depressed", "crying", "hopeless"],
}

URGENT_TERMS = [
    "chest pain",
    "cannot breathe",
    "difficulty breathing",
    "faint",
    "fainted",
    "confused",
    "bleeding",
    "blood discharge",
    "bloody discharge",
    "fever",
    "suicidal",
    "self harm",
    "kill myself",
]

IMMEDIATE_DANGER_PATTERNS = (
    r"\bi think i(?:'m| am|m) dying\b",
    r"\bi feel like i(?:'m| am|m) dying\b",
    r"\bi (?:do not|don't|dont) think i(?:'ll| will|ll) make it\b",
    r"\bi (?:will not|won't|wont) (?:last|make it)\b",
    r"\bi(?:'m| am) not going to (?:last|make it)\b",
    r"\bi (?:may|might) (?:die soon|not survive|not make it(?: through)?)\b",
    r"\bparang mamamatay na ako\b",
    r"\b(?:parang )?hindi na ako magtatagal\b",
    r"\bbaka hindi na ako (?:umabot|magtatagal)\b",
    r"\bhindi na ako aabot\b",
    r"\bbaka mamatay na ako\b",
)

SAFETY_LOCATION_FOLLOWUP_PATTERNS = (
    r"^go to where\??$",
    r"^where\??$",
    r"^where should i go\??$",
    r"^which hospital\??$",
    r"^saan(?: ako)? pupunta\??$",
    r"^saan ako dapat pumunta\??$",
)

CHAT_SYSTEM_PROMPT = """\
You are NLCare's patient support assistant for a breast cancer monitoring portal.

Rules:
- Do not diagnose, stage, confirm recurrence/metastasis, or decide treatment.
- Do not tell the patient to start, stop, increase, or decrease medications.
- Do not invent patient facts. Use recent_context only when directly helpful.
- Explain what was logged, ask for missing tracking details when useful, and encourage oncology team review for concerning symptoms.
- For greetings, identity questions, and "how are you" style messages, answer naturally and briefly as a warm portal assistant.
- Stay within NLCare's scope: breast-cancer monitoring records, general oncology education, portal help, emotional support, and care-team question preparation. Politely decline unrelated history, politics, trivia, coding, calculations, or general-purpose requests.
- If saved_actions contains a saved item, acknowledge the item clearly and do not add unrelated oncology education.
- Never imply that the patient personally logged a pre-existing record. For recent_context, say "the portal record shows". Only say "you logged" when saved_actions confirms a save in the current turn.
- If the message asks about prior chat, summarize only patient-scoped recent_context / chat messages.
- If urgent wording is present, advise contacting the oncology team or emergency services now.
- Keep the tone calm and practical. Maximum 120 words. Plain text only.
"""

ALLOWED_SUPPORT_TOOLS = {
    "none",
    "save_symptom",
    "request_symptom_details",
    "save_complete_cbc",
    "request_missing_cbc_fields",
    "save_medication",
    "save_imaging_report",
    "request_missing_imaging_details",
}

ALLOWED_SUPPORT_INTENTS = {
    "conversation",
    "education",
    "portal_help",
    "patient_memory",
    "emotional_support",
    "patient_timeline_monitoring",
    "general_support",
    "data_entry_confirmation",
    "safety_boundary",
    "treatment_decision_boundary",
    "scope_boundary",
}


DOMAIN_SCOPE_TERMS = {
    "breast", "cancer", "oncology", "oncologist", "chemo", "chemotherapy",
    "radiation", "treatment", "cycle", "symptom", "nausea", "fatigue",
    "fever", "pain", "neuropathy", "mouth sores", "lab", "labs", "cbc",
    "wbc", "hemoglobin", "platelet", "platelets", "anc", "neutrophil",
    "neutrophils", "neutropenia", "febrile neutropenia", "mri", "ct",
    "ultrasound", "imaging", "scan", "pathology", "biomarker", "her2",
    "er", "pr", "brca", "vus", "genetic", "tumor marker", "medication",
    "medicine", "dose", "record", "timeline", "monitoring", "monitoring score",
    "response score", "risk score",
    "index", "review queue", "care team", "doctor", "clinician", "portal",
    "nlcare", "upload", "report", "result", "my data", "my record",
    "patient-reported outcome", "patient reported outcome", "pro-ctcae",
    "clinical trial", "oncology research", "cancer research",
}

GENERAL_SUPPORT_PATTERNS = (
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening)[!. ]*$",
    r"^(?:thanks|thank you|salamat)[!. ]*$",
    r"^(?:how are you|how(?:'s| is) it going)[?.! ]*$",
    r"^(?:help|can you help me|what can you do|who are you)[?.! ]*$",
    r"^(?:what does (?:this|that|it) mean|can you explain (?:this|that|it))[?.! ]*$",
    r"^(?:what did i (?:tell|say)(?: to)? you(?: earlier| before)?|do you remember what i (?:told|said)|what do you remember about me)[?.! ]*$",
    r"^(?:i(?:'m| am) (?:scared|worried|anxious|sad|stressed)|nahihirapan ako).*$",
)

OUT_OF_DOMAIN_PATTERNS = (
    r"^who (?:is|was|are|were)\s+.+",
    r"^(?:what|when|where) (?:is|was|are|were) (?:the )?(?:capital|president|war|battle|country|celebrity|actor|singer).+",
    r"^(?:write|debug|explain) (?:some )?(?:code|python|javascript|sql).+",
    r"^(?:tell me|give me) (?:a )?(?:joke|recipe|poem|story).+",
    r"^(?:what is|calculate|solve)\s+[-+*/().\d\s=]+\??$",
    r"^[-+*/().\d\s=]+$",
)
