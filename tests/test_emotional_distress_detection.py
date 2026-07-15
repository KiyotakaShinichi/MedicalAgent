"""Tests for emotional distress detection.

Lock-ins:

* All 5 response modes are reachable.
* Crisis wording dominates despair / fear / anxiety / denial.
* Taglish coverage exists in every category (so the parity is not
  English-only).
* Negative controls (educational queries) return ``none`` /
  ``normal_education``.
"""
from __future__ import annotations

import unittest

from backend.services.emotional_distress_detection import (
    ANXIETY_TERMS,
    CRISIS_TERMS,
    DENIAL_TERMS,
    DESPAIR_TERMS,
    FEAR_TERMS,
    MORTALITY_DISTRESS_TERMS,
    RESPONSE_MODE_VALUES,
    detect_emotional_distress,
    vocabulary_manifest,
)


def _has_taglish(terms: tuple[str, ...]) -> bool:
    # Heuristic: at least one term contains a Taglish marker.
    markers = ("ako", "sila", "ang ", "wala", "gusto", "ayoko", "takot",
               "kabado", "natatakot", "hindi", "ito", "balisa", "kakaisip",
               "lakas", "iniisip", "inaalala")
    return any(any(m in t for m in markers) for t in terms)


class VocabularyParity(unittest.TestCase):
    def test_taglish_coverage_across_categories(self) -> None:
        for name, terms in (
            ("crisis", CRISIS_TERMS),
            ("despair", DESPAIR_TERMS),
            ("fear", FEAR_TERMS),
            ("anxiety", ANXIETY_TERMS),
            ("denial", DENIAL_TERMS),
            ("mortality_distress", MORTALITY_DISTRESS_TERMS),
        ):
            self.assertTrue(_has_taglish(terms), name)

    def test_response_modes_constant(self) -> None:
        self.assertEqual(set(RESPONSE_MODE_VALUES), {
            "normal_education",
            "empathetic_support_plus_education",
            "urgent_clinician_review",
            "crisis_support",
            "clinician_review_with_warm_handoff",
        })

    def test_manifest_shape(self) -> None:
        m = vocabulary_manifest()
        self.assertEqual(set(m["categories"].keys()), {
            "crisis", "despair", "fear", "anxiety", "denial", "mortality_distress",
        })


class Detection(unittest.TestCase):
    def test_normal_education_for_safe_query(self) -> None:
        v = detect_emotional_distress("What does pCR mean?")
        self.assertEqual(v.category, "none")
        self.assertEqual(v.response_mode, "normal_education")
        self.assertFalse(v.detected)

    def test_crisis_dominates(self) -> None:
        v = detect_emotional_distress("I'm so scared and I want to die.")
        self.assertEqual(v.category, "crisis")
        self.assertEqual(v.response_mode, "crisis_support")

    def test_crisis_taglish(self) -> None:
        v = detect_emotional_distress("Gusto ko nang mamatay, sobrang pagod na ako.")
        self.assertEqual(v.category, "crisis")

    def test_despair_with_safety_high_risk(self) -> None:
        v = detect_emotional_distress(
            "I give up. There is no hope.",
            safety={"level": "high_risk", "scope": "urgent_or_safety_related"},
        )
        self.assertEqual(v.category, "despair")
        self.assertEqual(v.response_mode, "urgent_clinician_review")

    def test_despair_without_safety_warm_handoff(self) -> None:
        v = detect_emotional_distress("I give up. There is no hope.")
        self.assertEqual(v.category, "despair")
        self.assertEqual(v.response_mode, "clinician_review_with_warm_handoff")

    def test_mortality_distress_routes_to_urgent_review_without_inferring_self_harm(self) -> None:
        for query in (
            "I do not think I will make it.",
            "I might not make it through this.",
            "Parang hindi na ako magtatagal.",
        ):
            v = detect_emotional_distress(query)
            self.assertEqual(v.category, "mortality_distress", query)
            self.assertEqual(v.response_mode, "urgent_clinician_review", query)

    def test_fear_taglish(self) -> None:
        v = detect_emotional_distress("Natatakot ako bumalik yung cancer ko.")
        self.assertEqual(v.category, "fear")
        self.assertEqual(v.response_mode, "empathetic_support_plus_education")

    def test_anxiety_english(self) -> None:
        v = detect_emotional_distress("I'm so anxious about my next scan, I can't sleep.")
        self.assertEqual(v.category, "anxiety")

    def test_denial_returns_empathetic(self) -> None:
        v = detect_emotional_distress("Hindi ako naniniwala, siguro nagkamali sila.")
        self.assertEqual(v.category, "denial")
        self.assertEqual(v.response_mode, "empathetic_support_plus_education")

    def test_to_dict_has_required_keys(self) -> None:
        v = detect_emotional_distress("I'm scared.").to_dict()
        for k in ("detected", "category", "response_mode", "matched_terms", "notes"):
            self.assertIn(k, v)


if __name__ == "__main__":
    unittest.main()
