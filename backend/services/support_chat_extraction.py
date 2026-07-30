"""Deterministic extraction helpers for patient support chat.

These helpers parse candidate record inputs only. Persistence and confirmation
remain in ``support_chat_agent`` so ambiguous mentions cannot silently become
patient records.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from backend.services.support_chat_policy import SYMPTOM_KEYWORDS


def _extract_symptom(message):
    # First try the multilingual / typo-tolerant extractor (handles
    # Taglish, Spanish, common typos, and qualitative severity hints).
    # If it returns None we fall back to the legacy English-only path.
    try:
        from backend.services.multilingual_tool_router import extract_symptom_multilingual
        multi = extract_symptom_multilingual(message)
    except Exception:  # noqa: BLE001 — never break chat on router error
        multi = None
    if multi is not None:
        # Shape-compatible with the legacy return value: the extra
        # keys (matched_terms, language_hint, severity_source) ride
        # through downstream consumers untouched.
        return {
            "symptom": multi["symptom"],
            "severity": multi.get("severity"),
            "severity_provided": multi.get("severity_provided", False),
            "severity_source": multi.get("severity_source"),
            "matched_terms": multi.get("matched_terms"),
            "language_hint": multi.get("language_hint"),
        }

    lower = message.lower()
    symptom_name = None
    for canonical, terms in SYMPTOM_KEYWORDS.items():
        if any(term in lower for term in terms):
            symptom_name = canonical
            break

    if not symptom_name:
        return None

    severity = _extract_severity(lower)
    return {
        "symptom": symptom_name,
        "severity": severity,
        "severity_provided": severity is not None,
    }


def _should_save_symptom(message, symptom):
    lower = message.lower()
    if _is_general_symptom_question(lower):
        return False
    if symptom.get("severity_provided"):
        return True
    if _explicit_tracking_request(lower) and _has_self_scoped_symptom_wording(lower):
        return True
    if symptom.get("symptom") == "fever" and _has_self_scoped_symptom_wording(lower):
        return True
    return False


def _should_request_symptom_details(message, symptom):
    lower = message.lower()
    if not symptom or _should_save_symptom(message, symptom) or _is_general_symptom_question(lower):
        return False
    if symptom.get("symptom") in {"anxiety", "sadness"} and not _explicit_tracking_request(lower):
        return False
    return _has_self_scoped_symptom_wording(lower) or _explicit_tracking_request(lower)


def _explicit_tracking_request(lower_message):
    return any(term in lower_message for term in ["log ", "save ", "record ", "report ", "track ", "add "])


def _has_self_scoped_symptom_wording(lower_message):
    patterns = [
        "i have",
        "i feel",
        "i am feeling",
        "i'm feeling",
        "i am having",
        "i'm having",
        "my ",
        "ako",
        "may ",
        "masakit",
        "nakakaramdam",
    ]
    return any(pattern in lower_message for pattern in patterns)


def _is_general_symptom_question(lower_message):
    stripped = lower_message.strip()
    if "my " in stripped or " i " in f" {stripped} ":
        return False
    question_starts = ("what is", "what are", "what does", "how does", "why", "can you explain", "explain", "tell me about")
    return stripped.endswith("?") or stripped.startswith(question_starts)


def _extract_severity(lower_message):
    patterns = [
        r"(\d{1,2})\s*/\s*10",
        r"severity\s*(?:is|:)?\s*(\d{1,2})",
        r"pain\s*(?:is|:)?\s*(\d{1,2})",
        r"level\s*(?:is|:)?\s*(\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower_message)
        if match:
            return max(0, min(10, int(match.group(1))))
    return None


def _extract_complete_labs(message):
    lower = message.lower()
    wbc = _extract_number_after(lower, ["wbc", "white blood"])
    hemoglobin = _extract_number_after(lower, ["hemoglobin", "hgb", "hb"])
    platelets = _extract_number_after(lower, ["platelets", "platelet", "plt"])
    if wbc is None or hemoglobin is None or platelets is None:
        return None
    return {
        "wbc": wbc,
        "hemoglobin": hemoglobin,
        "platelets": platelets,
    }


def _looks_like_partial_labs(message):
    lower = message.lower()
    return any(term in lower for term in ["wbc", "white blood", "hemoglobin", "hgb", "platelet", "platelets", "plt"])


def _is_general_lab_question(message):
    lower = message.lower().strip()
    if any(term in lower for term in ["my wbc", "my hemoglobin", "my hgb", "my platelet", "my cbc", "i got", "i have"]):
        return False
    question_starts = ("what is", "what are", "what does", "how does", "why", "can you explain", "explain", "tell me about")
    return lower.endswith("?") or lower.startswith(question_starts)


def _is_short_record_hint(message):
    return len(re.findall(r"[a-zA-Z0-9]+", message)) <= 4


def _extract_imaging_report(message):
    lower = message.lower()
    if not _has_imaging_term(lower):
        return None
    report_terms = [
        "findings",
        "impression",
        "mass",
        "tumor",
        "lesion",
        "bi-rads",
        "birads",
        "cm",
        "decreased",
        "increased",
        "stable",
        "interval",
        "response",
        "ct",
        "pet/ct",
        "pet ct",
        "hepatic",
        "liver",
        "lung",
        "pleural",
        "pericardial",
        "ascites",
        "free fluid",
        "peritoneal",
        "omental",
        "carcinomatosis",
        "lymph node",
        "adenopathy",
        "metastatic",
        "metastasis",
        "metastases",
    ]
    if not any(term in lower for term in report_terms):
        return None
    if _is_general_imaging_question(lower):
        return None

    modality, body_site = _infer_imaging_modality_and_body_site(lower)

    report_type = "Patient-entered imaging note"
    if any(term in lower for term in ["follow-up", "follow up", "interval", "decreased", "increased", "stable"]):
        report_type = "Follow-up"
    elif "baseline" in lower:
        report_type = "Baseline"
    elif any(term in lower for term in ["staging", "restaging", "metastatic", "metastasis", "metastases", "ascites", "peritoneal"]):
        report_type = "Staging/follow-up"

    impression = _extract_report_section(message, "impression") or _best_imaging_sentence(message) or message
    findings = _extract_report_section(message, "findings") or message
    return {
        "date": _extract_date(message),
        "modality": modality,
        "report_type": report_type,
        "body_site": body_site,
        "findings": findings.strip()[:12000],
        "impression": impression.strip()[:4000],
    }


def _looks_like_partial_imaging(message):
    lower = message.lower()
    if _is_general_imaging_question(lower):
        return False
    if not _has_imaging_term(lower):
        return False
    self_scoped_terms = ["my", "report", "scan", "result", "uploaded", "impression", "findings", "ct", "pet", "ultrasound"]
    return len(lower.split()) <= 6 or any(term in lower for term in self_scoped_terms)


def _has_imaging_term(lower_message):
    return any(term in lower_message for term in [
        "mri",
        "imaging",
        "scan",
        "bi-rads",
        "birads",
        "mammogram",
        "ultrasound",
        "sonogram",
        "ct ",
        " ct",
        "ct/",
        "pet/ct",
        "pet ct",
        "cat scan",
    ])


def _is_general_imaging_question(lower_message):
    stripped = lower_message.strip()
    question_starts = ("what is", "what does", "how does", "why", "can you explain", "explain")
    return stripped.endswith("?") and stripped.startswith(question_starts) and "my " not in stripped


def _extract_report_section(message, label):
    pattern = rf"{label}\s*(?:is|:|-)?\s*(.+?)(?:\b(?:findings|impression)\s*(?:is|:|-)|$)"
    match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    section = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")
    return section if section else None


def _infer_imaging_modality_and_body_site(lower):
    abdominal_terms = [
        "abdomen",
        "abdominal",
        "pelvis",
        "pelvic",
        "liver",
        "hepatic",
        "ascites",
        "free fluid",
        "peritoneal",
        "omental",
        "carcinomatosis",
        "retroperitoneal",
    ]
    chest_terms = ["chest", "lung", "pulmonary", "pleural", "mediastinal", "pericardial"]

    if "pet/ct" in lower or "pet ct" in lower:
        return "FDG PET/CT", "Whole body"
    if re.search(r"\bct\b", lower) or "cat scan" in lower:
        if any(term in lower for term in abdominal_terms) and any(term in lower for term in chest_terms):
            return "CT chest/abdomen/pelvis", "Chest/abdomen/pelvis"
        if any(term in lower for term in abdominal_terms):
            return "CT abdomen/pelvis", "Abdomen/pelvis"
        if any(term in lower for term in chest_terms):
            return "CT chest", "Chest"
        return "CT scan", "Unspecified"
    if "mammogram" in lower or "mammography" in lower:
        return "Mammogram", "Breast"
    if "ultrasound" in lower or "sonogram" in lower:
        if any(term in lower for term in abdominal_terms):
            return "Abdominal ultrasound", "Abdomen"
        return "Breast ultrasound", "Breast"
    return "Breast MRI", "Breast"


def _best_imaging_sentence(message):
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[;\n]", message) if part.strip()]
    terms = [
        "mass",
        "tumor",
        "lesion",
        "bi-rads",
        "birads",
        "cm",
        "decreased",
        "increased",
        "stable",
        "interval",
        "ascites",
        "free fluid",
        "peritoneal",
        "omental",
        "pleural",
        "hepatic",
        "liver",
        "metastatic",
        "metastasis",
        "metastases",
        "adenopathy",
    ]
    for sentence in sentences:
        if any(term in sentence.lower() for term in terms):
            return sentence
    return None


def _extract_number_after(lower_message, labels):
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", lower_message)
        if match:
            return float(match.group(1))
    return None


def _clinical_lab_alerts(labs):
    checks = [
        {
            "rule": "very_low_wbc",
            "label": "WBC",
            "value": labs["wbc"],
            "threshold": "<2.0",
            "severity": "high",
            "triggered": labs["wbc"] < 2.0,
        },
        {
            "rule": "very_low_hemoglobin",
            "label": "hemoglobin",
            "value": labs["hemoglobin"],
            "threshold": "<8.0",
            "severity": "high",
            "triggered": labs["hemoglobin"] < 8.0,
        },
        {
            "rule": "very_low_platelets",
            "label": "platelets",
            "value": labs["platelets"],
            "threshold": "<50",
            "severity": "high",
            "triggered": labs["platelets"] < 50,
        },
    ]
    return [
        {key: value for key, value in item.items() if key != "triggered"}
        for item in checks
        if item["triggered"]
    ]


def _extract_medication(message):
    lower = message.lower()
    patterns = [
        r"(?:i am taking|i'm taking|taking|started|start taking|took)\s+([a-zA-Z0-9\- ]{3,40})",
        r"(?:medication|medicine|drug)\s*(?:is|:)?\s*([a-zA-Z0-9\- ]{3,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        medication = _clean_medication_name(match.group(1))
        if medication:
            return {
                "medication": medication,
                "dose": _extract_dose(message),
                "frequency": _extract_frequency(lower),
            }
    return None


def _clean_medication_name(raw):
    cleaned = re.split(r"\b(?:and|but|because|for|since|with|today|yesterday|dose|mg|mcg|ml)\b|[,.]", raw, maxsplit=1)[0]
    cleaned = re.sub(r"\s+\d+(?:\.\d+)?\s*$", "", cleaned)
    cleaned = cleaned.strip(" .,:;-")
    if cleaned in {"my medication", "medicine", "medication"}:
        return None
    return cleaned[:80] if cleaned else None


def _extract_dose(message):
    match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)", message, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _extract_frequency(lower_message):
    for phrase in ["twice a day", "once a day", "daily", "weekly", "every week", "every night", "every morning"]:
        if phrase in lower_message:
            return phrase
    return None


def _extract_date(message):
    lower = message.lower()
    iso_match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", lower)
    if iso_match:
        year, month, day = [int(part) for part in iso_match.groups()]
        return date(year, month, day)
    slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", lower)
    if slash_match:
        month, day, year = [int(part) for part in slash_match.groups()]
        return date(year, month, day)
    if "yesterday" in lower:
        return date.today() - timedelta(days=1)

    month_match = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*"
        r"(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d{2}))?\b",
        lower,
    )
    if month_match:
        month_name, day_text, year_text = month_match.groups()
        month_lookup = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        year = int(year_text) if year_text else date.today().year
        return date(year, month_lookup[month_name], int(day_text))
    return date.today()


__all__ = [
    "_clinical_lab_alerts",
    "_extract_complete_labs",
    "_extract_date",
    "_extract_imaging_report",
    "_extract_medication",
    "_extract_severity",
    "_extract_symptom",
    "_is_general_lab_question",
    "_is_short_record_hint",
    "_looks_like_partial_imaging",
    "_looks_like_partial_labs",
    "_should_request_symptom_details",
    "_should_save_symptom",
]
