"""Safe report-type parsers for patient-provided medical text.

These parsers extract only explicit fields. They never infer diagnosis, stage,
genetic risk, treatment response, recurrence, or treatment plans.
"""

from __future__ import annotations

import re
from typing import Any


REPORT_PARSER_VERSION = "medical_report_parser_v1_2026_05"


def classify_report_type(text: str, filename: str | None = None) -> str:
    haystack = f"{filename or ''}\n{text or ''}".lower()
    if any(term in haystack for term in ("wbc", "hemoglobin", "platelet", "anc", "cbc")):
        return "cbc_report"
    if any(term in haystack for term in ("mri", "ct", "ultrasound", "pet/ct", "impression", "radiology")):
        return "imaging_report"
    if any(term in haystack for term in ("er:", "pr:", "her2", "ki-67", "pathology", "ihc", "fish")):
        return "pathology_biomarker_report"
    if any(term in haystack for term in ("brca", "palb2", "vus", "pathogenic", "germline", "somatic")):
        return "genetic_test_report"
    if any(term in haystack for term in ("ca 15-3", "ca15-3", "ca 27.29", "cea")):
        return "tumor_marker_report"
    return "unknown_report"


def parse_report(text: str, filename: str | None = None) -> dict[str, Any]:
    report_type = classify_report_type(text, filename)
    parsers = {
        "cbc_report": parse_cbc_report,
        "imaging_report": parse_imaging_report,
        "pathology_biomarker_report": parse_biomarker_report,
        "genetic_test_report": parse_genetic_report,
        "tumor_marker_report": parse_tumor_marker_report,
    }
    extracted = parsers.get(report_type, lambda _: {})(text or "")
    return {
        "parser_version": REPORT_PARSER_VERSION,
        "report_type": report_type,
        "extracted": extracted,
        "needs_clinician_review": report_type != "unknown_report",
        "not_inferred": [
            "diagnosis",
            "stage",
            "treatment_plan",
            "recurrence",
            "genetic_risk",
            "prognosis",
        ],
        "claim_boundary": (
            "Parser extracts explicit fields only. It does not interpret reports "
            "or recommend diagnosis/treatment."
        ),
    }


def parse_cbc_report(text: str) -> dict[str, Any]:
    return {
        "wbc": _number_after(text, r"\bWBC\b"),
        "anc": _number_after(text, r"\bANC\b"),
        "hemoglobin": _number_after(text, r"\b(Hgb|Hemoglobin)\b"),
        "platelets": _number_after(text, r"\b(Platelets?|Plt)\b"),
    }


def parse_imaging_report(text: str) -> dict[str, Any]:
    lowered = text.lower()
    modality = "unknown"
    for candidate in ("mri", "ct", "ultrasound", "pet/ct", "pet-ct"):
        if candidate in lowered:
            modality = "pet_ct" if candidate in {"pet/ct", "pet-ct"} else candidate
            break
    return {
        "modality": modality,
        "impression": _section(text, "impression"),
        "findings": _section(text, "findings"),
        "mentions_ascites": "ascites" in lowered,
        "mentions_metastasis_wording": any(term in lowered for term in ("metastasis", "metastatic", "lesion")),
    }


def parse_biomarker_report(text: str) -> dict[str, Any]:
    return {
        "er_status": _status_after(text, "ER"),
        "pr_status": _status_after(text, "PR"),
        "her2_status": _status_after(text, "HER2"),
        "ki67_percent": _number_after(text, r"\b(Ki-?67)\b"),
        "fish_status": _fish_status(text),
    }


def parse_genetic_report(text: str) -> dict[str, Any]:
    genes = re.findall(r"\b(BRCA1|BRCA2|PALB2|TP53|PTEN|CHEK2|ATM)\b", text, flags=re.I)
    classification = None
    lowered = text.lower()
    for candidate in ("pathogenic", "likely pathogenic", "vus", "likely benign", "benign"):
        if candidate in lowered:
            classification = candidate
            break
    return {
        "genes_mentioned": sorted({gene.upper() for gene in genes}),
        "classification_text": classification,
        "mentions_vus": "vus" in lowered or "uncertain significance" in lowered,
    }


def parse_tumor_marker_report(text: str) -> dict[str, Any]:
    markers = {}
    for marker, pattern in {
        "CA 15-3": r"CA\s*15-?3",
        "CA 27.29": r"CA\s*27\.?29",
        "CEA": r"\bCEA\b",
    }.items():
        markers[marker] = _number_after(text, pattern)
    return {k: v for k, v in markers.items() if v is not None}


def _number_after(text: str, label_pattern: str) -> float | None:
    match = re.search(label_pattern + r"[^0-9\-]{0,20}([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if not match:
        return None
    try:
        return float(match.groups()[-1])
    except (TypeError, ValueError):
        return None


def _status_after(text: str, label: str) -> str | None:
    match = re.search(rf"\b{label}\b[^A-Za-z0-9]{{0,10}}(positive|negative|equivocal|low positive|unknown)", text, flags=re.I)
    return match.group(1).lower().replace(" ", "_") if match else None


def _fish_status(text: str) -> str | None:
    lowered = text.lower()
    if "fish amplified" in lowered:
        return "amplified"
    if "fish not amplified" in lowered or "fish non-amplified" in lowered:
        return "not_amplified"
    return None


def _section(text: str, heading: str) -> str | None:
    match = re.search(rf"{heading}\s*:\s*(.+?)(?:\n[A-Z][A-Za-z ]{{2,}}:|$)", text, flags=re.I | re.S)
    if not match:
        return None
    return " ".join(match.group(1).split())[:700]


__all__ = [
    "REPORT_PARSER_VERSION",
    "classify_report_type",
    "parse_report",
]
