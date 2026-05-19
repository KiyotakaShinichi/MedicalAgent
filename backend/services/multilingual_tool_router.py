"""Multilingual + typo-tolerant tool routing for the chat agent.

Goal
~~~~
The patient-facing chat layer accepts messages in **English**, **Filipino /
Taglish**, **Spanish**, and frequent typos / slang / mixed-language
sentences.  The deterministic extractors in ``support_chat_agent`` were
written assuming English, so a message like ``"may lagnat ako, severity 7
- nakakasuka rin"`` would miss the ``fever`` and ``nausea`` extraction
even though both are obvious to a human reader.

This module is the multilingual front-end for tool extraction:

  - :func:`normalize_user_text`   — diacritic-strip, repeated-char
    collapse, common-typo + slang substitution.  Cheap, deterministic.
  - :data:`MULTILINGUAL_SYMPTOM_TERMS` — symptom → list of terms across
    English, Filipino, Spanish, common variants, and frequent typos.
  - :data:`MULTILINGUAL_SEVERITY_HINTS` — qualitative phrases that map
    to a 0-10 severity bucket (``matindi`` → 8, ``konti lang`` → 3).
  - :func:`extract_symptom_multilingual` — drop-in shape-compatible
    replacement for ``support_chat_agent._extract_symptom``.
  - :func:`tool_intent_hints_from_text` — language-agnostic hints
    (``"save_labs"``, ``"save_imaging_report"``, ...) used by the tool
    planner when the regex extractors are inconclusive.

Claim boundary
~~~~~~~~~~~~~~
This module ROUTES queries to tools; it does NOT interpret medical
content.  Every blocked claim that the medical claim boundary checker
or the post-generation validator catches still applies.  The router is
just a multilingual + fuzzy front-end for *intent* + *what data the
user is trying to log*.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping


# ─── Normalization ───────────────────────────────────────────────────────────


# Common typo / slang substitutions, applied AFTER diacritic strip and
# lowercase.  Keep this table tight — every entry is a substring rewrite,
# so over-eager rules will mangle legitimate text.
_TYPO_AND_SLANG_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    # repeated-char shortenings (further repeats normalized by the regex below)
    ("fevr", "fever"),
    ("feverr", "fever"),
    ("feever", "fever"),
    ("nausious", "nauseous"),
    ("nautious", "nauseous"),
    ("vomitting", "vomiting"),
    ("vomitin", "vomiting"),
    ("vomits", "vomiting"),
    ("diarhea", "diarrhea"),
    ("diarrohea", "diarrhea"),
    ("daiarrhea", "diarrhea"),
    ("diahrea", "diarrhea"),
    ("hadache", "headache"),
    ("headach", "headache"),
    ("breathng", "breathing"),
    ("breething", "breathing"),
    ("shortnes", "shortness"),
    ("nuropathy", "neuropathy"),
    ("neuropathi", "neuropathy"),
    ("hemoglobon", "hemoglobin"),
    ("hemglobin", "hemoglobin"),
    ("hgb", "hemoglobin"),
    ("hb", "hemoglobin"),
    ("plts", "platelets"),
    ("plt", "platelets"),
    ("platelet ", "platelets "),
    ("wbcs", "wbc"),
    ("white blood cell", "wbc"),
    ("white blood cells", "wbc"),
    ("anc count", "anc"),
    # slang / informal
    ("temp ", "temperature "),
    ("mri report", "mri report"),  # no-op anchor, kept so the table is exhaustive
    ("uts", "ultrasound"),
    # Filipino / Taglish common spellings
    ("nlalagnat", "nilalagnat"),
    ("nalalagnat", "nilalagnat"),
    ("lagnat ako", "i have fever"),
    ("may lagnat", "i have fever"),
    ("nilalagnat", "fever"),
    ("nagsusuka", "vomiting"),
    ("nasusuka", "nausea"),
    ("hinihilo", "dizziness"),
    ("nahihilo", "dizziness"),
    ("nagdurugo", "bleeding"),
    ("masakit ang tiyan", "stomach pain"),
    ("masakit ang ulo", "headache"),
    ("sumasakit ang ulo", "headache"),
    ("sumasakit ang katawan", "body pain"),
    ("masakit ang katawan", "body pain"),
    ("sumasakit", "pain"),
    ("masakit", "pain"),
    ("hirap huminga", "shortness of breath"),
    ("hirap ako huminga", "shortness of breath"),
    ("nahihirapan huminga", "shortness of breath"),
    ("nahihirapan ako huminga", "shortness of breath"),
    ("pagod na pagod", "fatigue"),
    ("napapagod", "fatigue"),
    ("walang gana kumain", "low appetite"),
    ("walang gana", "low appetite"),
    ("ngalay", "fatigue"),
    ("kabag", "abdominal discomfort"),
    ("pawis na pawis", "sweating"),
    ("nilalamig", "chills"),
    # Filipino / Taglish medication phrasing — rewrite so the
    # English-pattern medication extractor in support_chat_agent picks
    # them up.
    ("umiinom ako ng", "i am taking"),
    ("umiinom ako", "i am taking"),
    ("ininum ko", "i took"),
    ("iniinom ko", "i am taking"),
    ("ginagamit ko", "i am taking"),
    ("gumagamit ng", "taking"),
    ("gamot ko ay", "medication is"),
    ("gamot ko:", "medication is"),
    ("gamot ko", "medication"),
    # Filipino imaging-report phrasing — rewrite to surface the
    # imaging-term triggers the extractor already looks for.
    ("ulat ng mri", "mri report"),
    ("ulat ng ct", "ct report"),
    ("ulat ng ultrasound", "ultrasound report"),
    ("ulat ng mammogram", "mammogram report"),
    ("mri ko", "mri report"),
    ("ct scan ko", "ct report"),
    ("ultrasound ko", "ultrasound report"),
    ("imaging ko", "imaging report"),
    ("nakapag mri ako", "i had an mri"),
    ("nakapag ct ako", "i had a ct"),
    # Filipino lab-result phrasing — pre-normalize to expose
    # English keywords for _extract_complete_labs.
    ("resulta ng cbc", "cbc result"),
    ("cbc ko ay", "cbc result is"),
    ("cbc result ko", "cbc result"),
    ("lab result ko", "lab result"),
    ("globulos blancos", "wbc"),
    ("conteo de plaquetas", "platelets"),
    # Spanish common forms (light coverage)
    ("fiebre", "fever"),
    ("nauseas", "nausea"),
    ("nausea ", "nausea "),
    ("vomito", "vomiting"),
    ("vomitos", "vomiting"),
    ("dolor de cabeza", "headache"),
    ("dolor de pecho", "chest pain"),
    ("falta de aire", "shortness of breath"),
    ("sangrado", "bleeding"),
    ("cansancio", "fatigue"),
    ("debilidad", "weakness"),
    # Spanish medication phrasing — rewrite to English shape.
    ("estoy tomando", "i am taking"),
    ("estoy ingiriendo", "i am taking"),
    ("tome ", "i took "),
    ("tomo ", "taking "),
    ("medicamento es", "medication is"),
)


_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_MULTI_WS_RE = re.compile(r"\s+")


def normalize_user_text(text: str) -> str:
    """Return a normalized form of ``text`` suitable for keyword matching.

    Steps (in order):
      1. NFKD + strip diacritics ("kamustá" → "kamusta")
      2. Lowercase
      3. Replace common punctuation noise with spaces
      4. Collapse 3+ repeated characters down to 2 ("feeeever" → "feever")
      5. Apply :data:`_TYPO_AND_SLANG_SUBSTITUTIONS` (substring rewrites)
      6. Collapse whitespace runs

    Pure function.  No state, no I/O.
    """
    if not text:
        return ""

    stripped = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    lower = stripped.lower()
    # Standardize quotes and dashes; preserve numbers and slashes (lab values use them).
    lower = lower.replace("’", "'").replace("‘", "'").replace("`", "'")
    lower = lower.replace("—", " ").replace("–", " ")
    # Collapse 3+ repeated chars to 2 ("feeeever" → "feever").
    lower = _REPEATED_CHAR_RE.sub(r"\1\1", lower)
    # Apply rewrite table.
    for needle, replacement in _TYPO_AND_SLANG_SUBSTITUTIONS:
        if needle in lower:
            lower = lower.replace(needle, replacement)
    return _MULTI_WS_RE.sub(" ", lower).strip()


# ─── Multilingual vocabulary tables ─────────────────────────────────────────


# Symptom name → list of terms.  English + Filipino + Spanish + common typos.
# Order within each list is irrelevant; matching is "any term in normalized text".
MULTILINGUAL_SYMPTOM_TERMS: dict[str, tuple[str, ...]] = {
    "fever": (
        "fever", "febrile", "high temperature", "temperature",
        # Filipino
        "lagnat", "mainit ang katawan", "nilalagnat",
        # Spanish
        "fiebre",
    ),
    "fatigue": (
        "fatigue", "tired", "weak", "exhausted", "weakness",
        # Filipino
        "pagod", "pagod na pagod", "ngalay", "panghihina", "nanghihina",
        # Spanish
        "cansancio", "debilidad",
    ),
    "nausea": (
        "nausea", "nauseous", "queasy", "want to vomit",
        # Filipino
        "nasusuka", "naduduwal", "duwal", "nahihilo at nasusuka",
        # Spanish
        "nauseas",
    ),
    "vomiting": (
        "vomit", "vomiting", "throw up", "threw up", "throwing up",
        # Filipino
        "nagsusuka", "sumuka", "sumusuka",
        # Spanish
        "vomito", "vomitos",
    ),
    "pain": (
        "pain", "ache", "aching", "sore", "hurts",
        # Filipino
        "sumasakit", "masakit", "kirot", "kumikirot",
        # Spanish
        "dolor",
    ),
    "fever_chills": (
        "chills", "shivering", "shaking with cold",
        # Filipino
        "nilalamig", "ginawan",
    ),
    "bleeding": (
        "bleeding", "blood loss", "spotting", "hemorrhage",
        # Filipino
        "nagdurugo", "may dugo", "dumudugo",
        # Spanish
        "sangrado", "sangrando",
    ),
    "bloody discharge": (
        "blood discharge", "bloody discharge", "bleeding discharge",
        "blood-stained discharge", "blood stained discharge",
    ),
    "discharge": (
        "discharge",
        # Filipino
        "may lumalabas", "may umaagos",
    ),
    "shortness of breath": (
        "shortness of breath", "breathless", "difficulty breathing",
        "trouble breathing", "can't breathe", "cannot breathe",
        # Filipino
        "hirap huminga", "nahihirapan huminga", "nahihirapan magpalipas ng hininga",
        # Spanish
        "falta de aire", "dificultad para respirar",
    ),
    "neuropathy": (
        "neuropathy", "tingling", "numbness", "pins and needles",
        "burning in hands", "burning in feet",
        # Filipino
        "namamanhid", "manhid", "pangangati ng kamay", "pangangati ng paa",
    ),
    "low appetite": (
        "low appetite", "no appetite", "not eating", "loss of appetite",
        # Filipino
        "walang gana", "walang gana kumain", "ayaw kumain",
    ),
    "anxiety": (
        "anxiety", "anxious", "scared", "worried", "panic",
        # Filipino
        "kabado", "nag-aalala", "natatakot",
    ),
    "sadness": (
        "sad", "depressed", "crying", "hopeless", "down",
        # Filipino
        "malungkot", "nalulungkot", "umiiyak",
    ),
    "mouth sores": (
        "mouth sores", "mouth ulcer", "mouth pain", "sore mouth", "mucositis",
        # Filipino
        "singaw", "may singaw", "masakit ang bibig",
    ),
    "diarrhea": (
        "diarrhea", "loose stool", "loose stools", "watery stool", "watery stools",
        # Filipino
        "pagtatae", "nagtatae", "matubig ang dumi",
    ),
    "dizziness": (
        "dizzy", "dizziness", "lightheaded",
        # Filipino
        "nahihilo", "hinihilo",
    ),
}


# Qualitative severity hints → integer 0-10 bucket.
MULTILINGUAL_SEVERITY_HINTS: dict[str, int] = {
    # English
    "mild": 3, "a little": 3, "slight": 3, "slightly": 3, "minor": 2,
    "moderate": 5, "medium": 5, "okay-ish": 4, "okay ish": 4,
    "bad": 7, "really bad": 8, "very bad": 8, "severe": 8, "intense": 8,
    "really severe": 9, "very severe": 9, "extreme": 9, "worst": 10,
    # Filipino
    "konti lang": 3, "konti": 3, "kaunti lang": 3, "kaunti": 3, "minsan lang": 3,
    "katamtaman": 5, "medyo": 4,
    "matindi": 8, "sobra": 8, "sobrang sakit": 9, "sobrang lala": 9,
    "lala": 8, "grabe": 9, "grabeng grabe": 10,
    # Spanish
    "leve": 3, "moderado": 5, "fuerte": 7, "muy fuerte": 9, "intenso": 8,
}


# Lab value normalizers — covers "wbc 2 . 1", "wbc:2,1", "wbc=2.1", etc.
_LAB_VALUE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?<=\b)(wbc|anc|hemoglobin|hgb|hb|platelets|plt)\s*[:=]\s*", r"\1 "),
    (r"(\d+)\s*,\s*(\d+)", r"\1.\2"),  # European decimal comma
    (r"(\d+)\s*\.\s*(\d+)", r"\1.\2"),  # spaced decimal "2 . 1" → "2.1"
)


def normalize_lab_value_string(text: str) -> str:
    """Normalize a lab-value-bearing string so the existing
    ``_extract_complete_labs`` regex still matches messages from
    careless typists ("WBC : 2 , 1" → "WBC 2.1")."""
    normalized = text
    for pattern, replacement in _LAB_VALUE_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


# ─── Symptom extraction ──────────────────────────────────────────────────────


# Severity numeric patterns — same family as support_chat_agent._extract_severity,
# but extended for Filipino phrasing ("severity 7", "lala 8/10", "lebel 6 sa 10").
_SEVERITY_NUMERIC_PATTERNS: tuple[str, ...] = (
    r"(\d{1,2})\s*/\s*10",
    r"(\d{1,2})\s*out of\s*10",
    r"severity\s*(?:is|:)?\s*(\d{1,2})",
    r"pain\s*(?:is|:)?\s*(\d{1,2})",
    r"level\s*(?:is|:)?\s*(\d{1,2})",
    r"lebel\s*(?:is|:)?\s*(\d{1,2})",  # Taglish phonetic
    r"lala\s*(?:is|:)?\s*(\d{1,2})",
    r"(\d{1,2})\s*sa\s*10",
)


def _extract_numeric_severity(text: str) -> int | None:
    """Return an integer 0-10 severity from numeric patterns, else None."""
    for pattern in _SEVERITY_NUMERIC_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            if 0 <= value <= 10:
                return value
    return None


def _extract_qualitative_severity(text: str) -> int | None:
    """Return an integer severity inferred from qualitative hints
    (``"matindi"``, ``"mild"``, ``"sobra"``)."""
    # Longest-prefix-first so "really severe" beats "severe".
    for phrase in sorted(MULTILINGUAL_SEVERITY_HINTS, key=len, reverse=True):
        if phrase in text:
            return MULTILINGUAL_SEVERITY_HINTS[phrase]
    return None


def extract_symptom_multilingual(message: str) -> dict[str, Any] | None:
    """Drop-in shape-compatible replacement for
    ``support_chat_agent._extract_symptom`` that understands
    Filipino/Taglish/Spanish wording, common typos, and qualitative
    severity hints.

    Returns ``None`` when no symptom matched; otherwise returns:

        {
            "symptom":           <canonical symptom name>,
            "severity":          <int 0-10 or None>,
            "severity_provided": <bool>,
            "severity_source":   "numeric" | "qualitative" | "missing",
            "matched_terms":     [<term1>, <term2>, ...],
            "language_hint":     "en" | "tl" | "es" | "mixed",
        }
    """
    if not message:
        return None
    normalized = normalize_user_text(message)
    if not normalized:
        return None

    matched_symptom: str | None = None
    matched_terms: list[str] = []
    for canonical, terms in MULTILINGUAL_SYMPTOM_TERMS.items():
        for term in terms:
            if term in normalized:
                matched_symptom = canonical
                matched_terms.append(term)
                break
        if matched_symptom:
            break
    if matched_symptom is None:
        return None

    severity = _extract_numeric_severity(normalized)
    severity_source = "numeric" if severity is not None else "missing"
    if severity is None:
        severity = _extract_qualitative_severity(normalized)
        severity_source = "qualitative" if severity is not None else "missing"

    return {
        "symptom":           matched_symptom,
        "severity":          severity,
        "severity_provided": severity is not None,
        "severity_source":   severity_source,
        "matched_terms":     matched_terms,
        "language_hint":     guess_language(normalized),
    }


# ─── Language detection (best-effort hint) ──────────────────────────────────


_FILIPINO_MARKERS: tuple[str, ...] = (
    "ako", "ko", "naman", "po", "talaga", "yung", "ng", "sa ", "may ",
    "masakit", "pagod", "lagnat", "nasusuka", "kanina", "bukas", "kasi",
    "kamusta",
)
_SPANISH_MARKERS: tuple[str, ...] = (
    "tengo", "estoy", "muy", "siento", "dolor", "fiebre", "nauseas",
)
_ENGLISH_MARKERS: tuple[str, ...] = (
    "i have", "i am", "feel", "feeling", "today", "doctor", "since",
)


def guess_language(normalized_text: str) -> str:
    """Return ``"en"``, ``"tl"``, ``"es"``, or ``"mixed"`` for ``normalized_text``.
    Used as a routing hint only — the actual safety contract does not
    branch on language."""
    score_en = sum(1 for m in _ENGLISH_MARKERS if m in normalized_text)
    score_tl = sum(1 for m in _FILIPINO_MARKERS if m in normalized_text)
    score_es = sum(1 for m in _SPANISH_MARKERS if m in normalized_text)
    nonzero = [s for s in (score_en, score_tl, score_es) if s > 0]
    if len(nonzero) > 1:
        return "mixed"
    if score_tl > 0:
        return "tl"
    if score_es > 0:
        return "es"
    return "en"


# ─── Tool intent hints ──────────────────────────────────────────────────────


_TOOL_HINT_TERMS: dict[str, tuple[str, ...]] = {
    "save_symptom": (
        # English
        "log my", "log this", "save this", "save my", "record my", "track my",
        "i have", "i feel", "i am feeling", "i'm feeling",
        # Filipino
        "ako", "may ako", "masakit", "sumasakit",
        # Spanish
        "tengo", "siento",
    ),
    "save_complete_cbc": (
        "cbc", "wbc", "anc", "hemoglobin", "platelets",
        "blood count", "lab values", "blood test result",
        # Filipino
        "resulta ng dugo", "lab ko", "cbc ko",
    ),
    "save_imaging_report": (
        "mri", "ct", "ultrasound", "mammogram", "imaging report",
        "report says", "findings", "impression",
        # Filipino
        "ulat ng imaging", "ulat ng mri",
    ),
    "save_medication": (
        "i take", "i'm taking", "i am taking", "medication",
        "medicine", "supplement",
        # Filipino
        "umiinom ako", "gamot ko", "supplement ko",
        # Spanish
        "tomo", "estoy tomando",
    ),
}


def tool_intent_hints_from_text(message: str) -> list[str]:
    """Return zero or more **language-agnostic** tool intent hints
    based on ``normalize_user_text(message)``.  Hints are advisory —
    the tool planner is allowed to ignore them when the regex
    extractors already produced concrete data."""
    normalized = normalize_user_text(message)
    if not normalized:
        return []
    hits: list[str] = []
    for tool, terms in _TOOL_HINT_TERMS.items():
        if any(term in normalized for term in terms):
            hits.append(tool)
    return hits


__all__ = [
    "MULTILINGUAL_SYMPTOM_TERMS",
    "MULTILINGUAL_SEVERITY_HINTS",
    "normalize_user_text",
    "normalize_lab_value_string",
    "extract_symptom_multilingual",
    "guess_language",
    "tool_intent_hints_from_text",
]
