"""Tests for ``backend.services.special_population_boundary``.

Locks in:

  - Precedence: end_of_life_distress beats every other category.
  - Pediatric / pregnancy / breastfeeding / fertility / survivorship /
    recurrence anxiety / palliative all classify correctly with at
    least one English and (where relevant) one Taglish trigger.
  - ``urgent_escalation`` is True only for end_of_life_distress.
  - Empty / unrelated queries return ``category=None``.
  - Each safe-wording template includes a clinician routing phrase.
"""
from __future__ import annotations

import unittest

from backend.services.special_population_boundary import (
    SAFE_WORDING,
    classify_special_population,
    safe_wording_for,
)


class CategoryDetection(unittest.TestCase):
    def test_low_risk_default_is_no_category(self) -> None:
        verdict = classify_special_population("What is pCR?")
        self.assertIsNone(verdict.category)
        self.assertFalse(verdict.urgent_escalation)

    def test_empty_query_is_no_category(self) -> None:
        self.assertIsNone(classify_special_population("").category)
        self.assertIsNone(classify_special_population("   ").category)

    def test_pregnancy_is_detected(self) -> None:
        v = classify_special_population("I am pregnant — what about my chemo?")
        self.assertEqual(v.category, "pregnancy")
        self.assertFalse(v.urgent_escalation)
        self.assertIn("pregnant", v.matched_terms)

    def test_taglish_pregnancy_is_detected(self) -> None:
        v = classify_special_population("Buntis ako, ano ang dapat gawin?")
        self.assertEqual(v.category, "pregnancy")

    def test_breastfeeding_is_detected(self) -> None:
        v = classify_special_population("I am breastfeeding, is treatment safe?")
        self.assertEqual(v.category, "breastfeeding")

    def test_pediatric_is_detected(self) -> None:
        v = classify_special_population("My daughter is 14 year old and has a lump.")
        self.assertEqual(v.category, "pediatric")

    def test_fertility_is_detected(self) -> None:
        v = classify_special_population("What about fertility after chemo?")
        self.assertEqual(v.category, "fertility")

    def test_survivorship_is_detected(self) -> None:
        v = classify_special_population("I finished chemo last month — what now?")
        self.assertEqual(v.category, "survivorship")

    def test_recurrence_anxiety_is_detected(self) -> None:
        v = classify_special_population("I am scared the cancer came back.")
        self.assertEqual(v.category, "recurrence_anxiety")

    def test_palliative_is_detected(self) -> None:
        v = classify_special_population("I want to focus on palliative care.")
        self.assertEqual(v.category, "palliative_or_supportive")


class UrgentEscalationOnlyForEndOfLife(unittest.TestCase):
    def test_end_of_life_distress_forces_urgent(self) -> None:
        v = classify_special_population("I don't want to live through this.")
        self.assertEqual(v.category, "end_of_life_distress")
        self.assertTrue(v.urgent_escalation)
        self.assertIn("crisis", v.routing.lower())

    def test_taglish_end_of_life(self) -> None:
        v = classify_special_population("Gusto ko nang mamatay.")
        self.assertEqual(v.category, "end_of_life_distress")
        self.assertTrue(v.urgent_escalation)


class PrecedenceOrder(unittest.TestCase):
    """When two categories trigger, end_of_life_distress wins."""

    def test_end_of_life_beats_pregnancy(self) -> None:
        v = classify_special_population("I am pregnant and want to end it.")
        self.assertEqual(v.category, "end_of_life_distress")

    def test_pediatric_beats_recurrence(self) -> None:
        v = classify_special_population("My son is afraid the cancer came back.")
        # Pediatric is the bigger boundary (different specialists),
        # so it wins per the table order.
        self.assertEqual(v.category, "pediatric")


class SafeWordingTemplates(unittest.TestCase):
    """Every safe-wording template must route to a real clinician
    contact (oncology / obstetrics / pediatrics / pharmacist / genetic
    counselor / emergency)."""

    CLINICIAN_PHRASES = (
        "oncology", "obstetrician", "pediatric", "lactation",
        "genetic counselor", "palliative", "reproductive", "emergency",
        "crisis", "care team",
    )

    def test_every_category_has_safe_wording(self) -> None:
        for category in SAFE_WORDING:
            self.assertTrue(safe_wording_for(category))

    def test_every_template_mentions_a_clinician_or_crisis(self) -> None:
        for category, text in SAFE_WORDING.items():
            lower = text.lower()
            self.assertTrue(
                any(phrase in lower for phrase in self.CLINICIAN_PHRASES),
                f"{category} template missing clinician/crisis routing",
            )


class VerdictSerialization(unittest.TestCase):
    def test_to_dict_has_all_fields(self) -> None:
        v = classify_special_population("I am pregnant.")
        as_dict = v.to_dict()
        for key in ("category", "urgent_escalation", "matched_terms", "safe_wording", "routing"):
            self.assertIn(key, as_dict)


if __name__ == "__main__":
    unittest.main()
