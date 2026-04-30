import re


SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
BI_RADS_PATTERN = re.compile(r"bi[- ]?rads\s*(\d)", re.IGNORECASE)

BREAST_SIDE_PATTERNS = {
    "left": r"\bleft breast\b|\bleft\b",
    "right": r"\bright breast\b|\bright\b",
    "bilateral": r"\bbilateral\b|both breasts",
}

BREAST_LOCATION_PATTERNS = {
    "upper_outer": r"upper outer|upper-outer|uoq",
    "upper_inner": r"upper inner|upper-inner|uiq",
    "lower_outer": r"lower outer|lower-outer|loq",
    "lower_inner": r"lower inner|lower-inner|liq",
    "retroareolar": r"retroareolar|subareolar",
    "axilla": r"axilla|axillary",
}

RESPONSE_PATTERNS = {
    "improved": [
        "decrease",
        "decreased",
        "smaller",
        "reduced enhancement",
        "partial response",
        "near complete response",
        "response to treatment",
    ],
    "worsened": [
        "increase",
        "increased",
        "larger",
        "progression",
        "progressive",
        "new lesion",
        "new enhancing mass",
        "new metast",
    ],
    "stable": [
        "stable",
        "unchanged",
        "no significant change",
    ],
}

METASTATIC_INDICATOR_PATTERNS = [
    ("bone", r"\b(bone|osseous|spine|vertebral|rib)\b.*\b(lesion|metasta|deposit)\b"),
    ("liver", r"\b(liver|hepatic)\b.*\b(lesion|metasta|deposit)\b"),
    ("lung", r"\b(lung|pulmonary)\b.*\b(nodule|lesion|metasta)\b"),
    ("brain", r"\b(brain|intracranial)\b.*\b(lesion|metasta|deposit)\b"),
    ("adrenal", r"\b(adrenal)\b.*\b(nodule|mass|lesion|metasta)\b"),
    ("pleura", r"\b(pleural)\b.*\b(nodularity|nodule|metasta|effusion)\b"),
    ("distant_lymph_nodes", r"\b(supraclavicular|internal mammary|mediastinal|abdominal|retroperitoneal)\b.*\b(lymph node|node|adenopathy)\b"),
]


def extract_tumor_sizes(text):
    return [float(match.group(1)) for match in SIZE_PATTERN.finditer(text or "")]


def extract_largest_tumor_size(text):
    sizes = extract_tumor_sizes(text)
    return max(sizes) if sizes else None


def extract_bi_rads(text):
    match = BI_RADS_PATTERN.search(text or "")
    return int(match.group(1)) if match else None


def extract_breast_side(text):
    normalized = (text or "").lower()
    for side, pattern in BREAST_SIDE_PATTERNS.items():
        if re.search(pattern, normalized):
            return side
    return "unknown"


def extract_breast_locations(text):
    normalized = (text or "").lower()
    return [
        location
        for location, pattern in BREAST_LOCATION_PATTERNS.items()
        if re.search(pattern, normalized)
    ]


def classify_response_language(text):
    normalized = (text or "").lower()
    matched = []

    for status, phrases in RESPONSE_PATTERNS.items():
        for phrase in phrases:
            if phrase in normalized:
                matched.append({"status": status, "phrase": phrase})

    if not matched:
        return {"status": "unknown", "matched_phrases": []}

    priority = {"worsened": 3, "improved": 2, "stable": 1}
    status = max(matched, key=lambda item: priority[item["status"]])["status"]
    return {"status": status, "matched_phrases": matched}


def detect_lymph_node_mentions(text):
    normalized = (text or "").lower()
    return {
        "mentioned": any(term in normalized for term in ["lymph node", "lymph nodes", "adenopathy", "axillary node"]),
        "axillary_mentioned": "axillary" in normalized,
        "improved": any(term in normalized for term in ["lymph nodes decreased", "adenopathy decreased", "nodes decreased", "axillary node decreased"]),
        "worsened": any(term in normalized for term in ["lymph nodes increased", "adenopathy increased", "nodes increased", "new lymph", "new axillary"]),
    }


def _is_negated(sentence):
    negations = [
        "no new",
        "no evidence of",
        "without evidence of",
        "negative for",
        "no definite",
    ]
    return any(phrase in sentence for phrase in negations)


def detect_possible_metastatic_indicators(text):
    indicators = []
    normalized = (text or "").lower()
    sentences = [sentence.strip() for sentence in re.split(r"[.;]", normalized) if sentence.strip()]

    for site, pattern in METASTATIC_INDICATOR_PATTERNS:
        for sentence in sentences:
            if re.search(pattern, sentence) and not _is_negated(sentence):
                indicators.append({
                    "site": site,
                    "message": f"Report text mentions a possible distant disease indicator involving {site.replace('_', ' ')}. This is not a diagnosis and should be reviewed by a clinician.",
                })
                break

    for sentence in sentences:
        has_metastasis_wording = "metastatic" in sentence or "metastasis" in sentence or "metastases" in sentence
        if has_metastasis_wording and not _is_negated(sentence):
            indicators.append({
                "site": "unspecified",
                "message": "Report text contains metastasis-related wording. This should be interpreted by a licensed clinician in context.",
            })
            break

    return indicators


def summarize_report(row):
    text = f"{row.get('findings', '')} {row.get('impression', '')}"
    modality = row.get("modality", "unknown")
    return {
        "date": str(row["date"]),
        "modality": modality,
        "report_type": row.get("report_type"),
        "body_site": row.get("body_site"),
        "breast_side": extract_breast_side(text),
        "breast_locations": extract_breast_locations(text),
        "bi_rads": extract_bi_rads(text),
        "tumor_sizes_cm": extract_tumor_sizes(text),
        "largest_tumor_size_cm": extract_largest_tumor_size(text),
        "response_language": classify_response_language(text),
        "lymph_nodes": detect_lymph_node_mentions(text),
        "possible_metastatic_indicators": detect_possible_metastatic_indicators(text),
        "impression": row.get("impression"),
    }


def analyze_breast_imaging_reports(imaging_df):
    imaging_df = imaging_df.sort_values("date")

    reports = [summarize_report(row) for _, row in imaging_df.iterrows()]
    measurable_reports = [report for report in reports if report["largest_tumor_size_cm"] is not None]
    baseline = measurable_reports[0] if measurable_reports else reports[0]
    latest = measurable_reports[-1] if measurable_reports else reports[-1]
    latest_report = reports[-1]

    baseline_size = baseline["largest_tumor_size_cm"]
    latest_size = latest["largest_tumor_size_cm"]

    if baseline_size and latest_size:
        change = latest_size - baseline_size
        percent_change = (change / baseline_size) * 100

        if change < 0:
            size_status = "decreased"
        elif change > 0:
            size_status = "increased"
        else:
            size_status = "stable"
    else:
        change = None
        percent_change = None
        size_status = "unknown"

    indicators = []
    for report in reports:
        for indicator in report["possible_metastatic_indicators"]:
            indicators.append({
                "date": report["date"],
                "modality": report["modality"],
                **indicator,
            })

    return {
        "baseline_date": baseline["date"],
        "latest_date": latest["date"],
        "latest_report_date": latest_report["date"],
        "baseline_tumor_size_cm": baseline_size,
        "latest_tumor_size_cm": latest_size,
        "change_cm": round(change, 2) if change is not None else None,
        "percent_change": round(percent_change, 1) if percent_change is not None else None,
        "size_status": size_status,
        "response_language_status": latest["response_language"]["status"],
        "latest_modality": latest["modality"],
        "latest_report_modality": latest_report["modality"],
        "latest_breast_side": latest["breast_side"],
        "latest_breast_locations": latest["breast_locations"],
        "latest_bi_rads": latest["bi_rads"],
        "lymph_node_summary": latest["lymph_nodes"],
        "possible_metastatic_indicators": indicators,
        "reports": reports,
        "latest_impression": latest["impression"],
        "latest_report_impression": latest_report["impression"],
        "safety_note": "Breast imaging NLP flags report text patterns only. It does not diagnose metastasis, recurrence, or treatment response.",
    }


def analyze_radiology_reports(imaging_df):
    return analyze_breast_imaging_reports(imaging_df)
