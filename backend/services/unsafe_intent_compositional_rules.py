"""Static compositional rules used by the unsafe-intent classifier."""

from __future__ import annotations


SAFE_EDUCATIONAL_ANCHORS = (
    "what is",
    "what does",
    "explain",
    "in general",
    "generally",
    "questions can i ask",
    "why should",
)

# These rules intentionally describe concepts rather than benchmark strings.
# A high-confidence hit requires one cue from every group (for example an
# access action + a private-record object). This makes paraphrase handling less
# dependent on one long regex while keeping safe educational mentions open.
# fmt: off
COMPOSITIONAL_RULES: dict[str, tuple[tuple[str, ...], ...]] = {
    "privacy_pii": (
        (
            "read back", "read", "recite", "download", "copy", "export", "paste",
            "forward", "reveal", "expose", "show", "send", "paki-send", "print",
            "verify", "display", "disclose", "unmask", "reproduce", "include",
            "keep visible", "visible", "leave unredacted", "omit redaction",
            "remove redaction", "return", "keep all",
        ),
        (
            "identifier", "identity field", "stored identity", "unredacted chart",
            "identity fields",
            "chart header", "confidential number", "confidential account",
            "account credential", "account detail", "profile secret", "address",
            "insurance", "member id", "membership code", "mrn", "dob", "phone",
            "token", "api key", "complete account record", "full chart",
            "account number", "patient identifier", "protected identifier",
            "private field", "private metadata", "protected field",
            "record header", "registration number", "profile header",
            "identity reference", "enrollment reference", "account metadata",
            "protected metadata",
        ),
    ),
    "prompt_injection": (
        (
            "ignore", "silently ignore", "disregard", "switch off", "disable",
            "replace", "work around", "circumvent", "bypass", "override",
            "operate without", "suspend", "pretend", "assume", "huwag sundin",
            "waive", "set aside", "deactivate", "skip", "exempt", "discard",
            "treat as approved", "treat as cleared", "execute past", "suppress",
            "omit", "continue after", "outside", "unrestricted",
        ),
        (
            "boundary", "medical refusal boundary", "refusal boundary",
            "safety instruction", "safety hierarchy", "policy hierarchy",
            "system constraint", "system constraints", "refusal policy",
            "citation check", "citation checker", "validator",
            "response validator", "final response validator", "final validator",
            "final response checker", "final checker", "instruction hierarchy", "internal instruction",
            "hidden prompt", "guardrail", "safety check", "safety checks",
            "remaining check", "remaining checks",
            "review layer", "policy check", "response check", "safety rule",
            "higher-priority instruction", "policy engine", "guardrails",
            "guardrail", "safety decision layer", "decision layer",
            "quoted instruction",
        ),
    ),
    "cross_patient_exfiltration": (
        (
            "open", "retrieve", "fetch", "quote", "contrast", "compare", "copy",
            "summarize", "show", "use", "access", "pull", "pakibukas", "gamitin",
            "buksan", "provide", "display", "disclose", "read", "export",
        ),
        (
            "person after me", "person before me", "next person",
            "previous person", "next patient", "previous patient",
            "another person", "someone else", "another patient", "other patient",
            "everyone's", "all patient", "relative's", "aunt's", "mother's",
            "father's", "sister's", "brother's", "mama", "patient q-",
            "patient id", "same surname", "kapatid", "cousin's", "partner's",
            "spouse's", "neighbor's", "coworker's", "family member's",
            "different patient", "unrelated patient", "every patient",
            "each patient", "adjacent account", "adjacent patient",
        ),
        (
            "record", "chart", "timeline", "history", "result", "results", "lab", "note",
            "medication", "imaging", "database", "portal", "prescription",
            "appointment", "treatment", "diagnosis", "profile", "marker value",
            "marker values", "ca 15", "ca15", "cea",
        ),
    ),
    "genetic_risk_interpretation": (
        ("gene", "genetic", "variant", "mutation", "brca", "palb2", "chek2", "atm", "germline", "somatic"),
        (
            "predict", "forecast", "verdict", "determine", "prove", "confirm",
            "definite", "decide", "will develop", "will get", "magkaka",
            "inherit cancer", "rank", "at risk", "siguradong at risk",
        ),
        (
            "child", "children", "anak", "son", "daughter", "sibling", "siblings",
            "relative", "family", "pamilya", "recurrence", "treatment", "therapy",
            "inherited", "hereditary", "cancer",
        ),
    ),
    "vus_misinterpretation": (
        (
            "vus", "variant of uncertain significance", "uncertain variant",
            "unclear variant", "unclassified variant", "indeterminate variant",
            "unclassified brca", "uncertain result", "uncertain genetic",
            "unclear result", "result is unclear", "inconclusive genetic",
            "inconclusive hereditary", "indeterminate hereditary",
            "hereditary result", "hereditary finding",
        ),
        (
            "resolve", "reclassify", "actionable", "pathogenic", "positive",
            "negative", "confirmed", "proof", "disease-causing", "disease causing",
            "decide therapy",
            "decide treatment", "decide which therapy", "select therapy",
            "count it", "count this", "treat it", "use the", "assume pathogenic",
        ),
    ),
    "diagnosis_confirmation": (
        (
            "issue", "declare", "confirm", "confirms", "decide", "definitively",
            "prove", "proves", "proved", "proven",
            "establish", "establishes", "yes-or-no", "turn", "conclusion",
            "assume", "diagnose", "definitely have", "doctor would conclude",
            "label", "classify", "rule that", "final verdict", "settle whether",
            "convert into",
        ),
        (
            "lesion", "mass", "scan", "scan phrase", "finding", "pathology",
            "lab", "laboratory panel", "portal summary", "cancer", "malignant",
            "malignancy", "spread", "recurrence", "metastatic", "cancer-free",
            "imaging wording", "biopsy wording", "disease label",
            "complete response", "treatment response", "pathologic response",
            "residual disease", "response outcome",
        ),
    ),
    "tumor_marker_conclusion": (
        ("marker", "marker trend", "rising marker", "ca 15", "ca15", "ca 27", "ca27", "cea", "number climbed", "number rose", "level rose"),
        (
            "confirm", "proof", "prove", "verdict", "definite", "settle",
            "assume", "therapy failed", "treatment failed", "disease returned",
            "recurrence", "metastasis", "progression", "replace imaging",
            "override imaging",
        ),
    ),
    "treatment_change": (
        (
            "stop", "pause", "delay", "miss", "skip", "restart", "resume",
            "switch", "replace", "cancel", "authorize", "continue or pause",
            "stopped", "paused", "delayed", "switched", "replaced",
            "ihihinto", "ihinto", "itigil", "ititigil", "palitan",
        ),
        ("chemotherapy", "chemo", "radiation", "treatment", "therapy", "regimen", "infusion", "tamoxifen", "trastuzumab", "medicine", "medication", "symptoms", "gamot"),
    ),
    "dosage_request": (
        (
            "calculate", "compute", "pick", "set", "adjust", "choose",
            "personalize", "double", "halve", "lower", "increase", "reduce",
            "how many", "ilang", "exact", "determine", "specify", "fill in",
            "work out", "derive", "decide the amount", "tell me", "convert",
            "supply",
        ),
        (
            "dose", "dosage", "mg", "tablet", "tablets", "pill", "pills",
            "injection amount", "medicine amount", "medication amount",
            "tablet count", "pill count", "number of tablets",
            "number of pills", "prescribed amount", "milligram", "milligrams",
            "prescription strength", "tablet amount", "milligram value",
            "next dose",
        ),
    ),
    "prognosis_survival": (
        (
            "estimate", "rank", "calculate", "convert", "quantify", "tell me",
            "give", "predict", "translate", "roughly", "turn", "turn into", "derive",
            "infer", "project", "work out", "will i", "what are my",
        ),
        (
            "remaining months", "remaining lifetime", "life remains", "surviving",
            "survival", "die soon", "alive next year", "alive after",
            "life expectancy", "survival countdown", "cure percentage",
            "personal timeline", "prognosis", "outlook", "chance of being alive",
            "alive in two years", "mabubuhay", "months left", "time left",
            "death date", "date of death", "personal survival date",
            "when i will die", "end of my life", "death",
            "odds of recurrence", "recurrence odds", "recurrence probability",
            "future birthdays", "final year",
        ),
    ),
    "supplement_replacement": (
        (
            "natural", "herb", "herbs", "herbal", "supplement", "vitamin",
            "vitamins", "cannabis", "turmeric", "cbd", "st john's wort",
            "st johns wort", "st john s wort",
        ),
        (
            "replace", "replacing", "replacement", "avoid", "instead",
            "stop treatment", "natural-only", "na lang", "cure", "only",
            "only therapy", "only cancer therapy", "without prescribed therapy",
            "without pharmacist",
        ),
        (
            "chemotherapy", "chemo", "radiation", "treatment", "therapy",
            "medicine", "medication", "prescribed care", "cancer", "gamot",
            "tamoxifen", "endocrine medicine", "endocrine treatment",
        ),
    ),
}

# fmt: on

DECISION_OR_ACCESS_CUES = (
    "for me",
    "my ",
    "should i",
    "tell me",
    "decide",
    "confirm",
    "prove",
    "predict",
    "show",
    "reveal",
    "export",
    "send",
    "forward",
    "open",
    "access",
    "use this",
    "without",
    "instead",
    "replace",
    "stop",
    "skip",
    "dose",
    "dosage",
    "mg",
)
