from __future__ import annotations

from typing import Any, Iterable

from backend.services.medication_interactions import check_medication_interactions


SUPPLEMENT_SAFETY_TERMS = {
    "st_johns_wort": ("st johns wort", "st. johns wort", "st john's wort", "hypericum"),
    "high_dose_antioxidants": ("high dose antioxidant", "antioxidant supplement", "vitamin e supplement", "high dose vitamin c"),
    "grapefruit": ("grapefruit", "grapefruit juice"),
    "cbd_cannabis": ("cbd", "cannabis", "cannabidiol", "marijuana"),
    "turmeric_curcumin": ("turmeric", "curcumin"),
    "green_tea_extract": ("green tea extract",),
    "herbal_mixtures": ("herbal mixture", "herbal blend", "traditional herbal"),
    "vitamin_megadoses": ("megadose", "mega dose", "high dose vitamin"),
    "bleeding_related": ("ginkgo", "garlic supplement", "fish oil", "omega-3", "omega 3"),
}


def flag_supplement_safety(
    supplement_text: str,
    *,
    current_medications: Iterable[str] = (),
    notes: str | None = None,
) -> dict[str, Any]:
    text = f"{supplement_text or ''} {notes or ''}".lower()
    matched_categories = [
        category for category, terms in SUPPLEMENT_SAFETY_TERMS.items()
        if any(term in text for term in terms)
    ]
    interaction = check_medication_interactions(supplement_text, current_medications=current_medications, notes=notes)
    review_needed = bool(matched_categories or interaction.get("flags"))
    return {
        "schema_version": "supplement_safety_flags_v1",
        "status": "review_needed" if review_needed else "no_specific_rule_hit",
        "matched_categories": matched_categories,
        "interaction_check": interaction,
        "patient_message": (
            "Ask your oncology team or pharmacist before using this with cancer treatment."
            if review_needed
            else "No specific rule matched, but supplements should still be reviewed with the care team during treatment."
        ),
        "claim_boundary": (
            "This does not determine whether a supplement is safe or unsafe. "
            "It flags common concerns for oncology-team/pharmacist review."
        ),
    }


__all__ = ["SUPPLEMENT_SAFETY_TERMS", "flag_supplement_safety"]
