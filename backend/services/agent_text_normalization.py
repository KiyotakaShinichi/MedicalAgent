"""Shared normalization for deterministic patient-agent routing.

The normalizer is intentionally conservative: it preserves decimal values and
severity fractions while removing formatting characters and punctuation noise
that should not change a safety or structured-record decision.
"""

from __future__ import annotations

import re
import unicodedata


TYPO_CANONICALIZATION = {
    "cancr": "cancer",
    "chemotherpy": "chemotherapy",
    "dosge": "dosage",
    "genetc": "genetic",
    "patien": "patient",
    "prognsis": "prognosis",
    "recods": "records",
    "severty": "severity",
    "shud": "should",
    "treatmnt": "treatment",
    "wat": "what",
}


def normalize_agent_text(text: str) -> str:
    """Return stable route-matching text without changing numeric values."""

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = "".join(char for char in value if unicodedata.category(char) not in {"Cf", "Cc"})
    value = _strip_diacritics(value).lower().replace("’", "'")
    value = re.sub(r"\bi\s*'?m\b", "i am", value)
    value = re.sub(r"\bcan't\b", "cannot", value)
    value = re.sub(r"\bwon't\b", "will not", value)
    value = re.sub(r"\bdon't\b", "do not", value)
    value = re.sub(r"\bdoesn't\b", "does not", value)
    value = re.sub(r"(?<!\d)[./]|[./](?!\d)", " ", value)
    value = re.sub(r"[_\-\\|*~`\"()\[\]{}:;,+?<>]+", " ", value)
    value = re.sub(r"[^a-z0-9.'/%\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    for typo, canonical in TYPO_CANONICALIZATION.items():
        value = re.sub(rf"\b{re.escape(typo)}\b", canonical, value)
    return value


def _strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


__all__ = ["TYPO_CANONICALIZATION", "normalize_agent_text"]
