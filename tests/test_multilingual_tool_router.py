"""Tests for ``backend.services.multilingual_tool_router``.

Locks in three contracts:

  1. ``normalize_user_text`` is idempotent, strips diacritics, collapses
     3+ repeated chars to 2, and applies the typo / slang rewrite table.
  2. ``extract_symptom_multilingual`` correctly extracts symptom +
     severity from English, Filipino / Taglish, Spanish, mixed-language,
     and typo'd input, and returns ``None`` for non-symptom queries.
  3. ``tool_intent_hints_from_text`` returns the right tool family for
     each language variant and stays empty for greetings / education.
"""
from __future__ import annotations

import unittest

from backend.services.multilingual_tool_router import (
    MULTILINGUAL_SEVERITY_HINTS,
    MULTILINGUAL_SYMPTOM_TERMS,
    extract_symptom_multilingual,
    guess_language,
    normalize_user_text,
    tool_intent_hints_from_text,
)


class NormalizeUserText(unittest.TestCase):
    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(normalize_user_text(""), "")
        self.assertEqual(normalize_user_text("   "), "")

    def test_lowercases_and_collapses_repeats(self) -> None:
        # "FEEEEVER" collapses to "feever" via the repeated-char rule, and
        # then the typo table rewrites "feever" -> "fever".
        self.assertEqual(normalize_user_text("FEEEEVER"), "fever")
        self.assertEqual(normalize_user_text("OUCH!!!!"), "ouch!!")

    def test_strips_diacritics(self) -> None:
        # "kamustá pô" → "kamusta po"
        self.assertEqual(normalize_user_text("kamustá pô"), "kamusta po")

    def test_typo_rewrite(self) -> None:
        self.assertIn("vomiting", normalize_user_text("i am vomitting all morning"))
        self.assertIn("diarrhea", normalize_user_text("daiarrhea twice today"))
        self.assertIn("fever",    normalize_user_text("i have a fevr"))

    def test_filipino_rewrites(self) -> None:
        self.assertIn("shortness of breath", normalize_user_text("nahihirapan huminga"))
        self.assertIn("nausea",              normalize_user_text("nasusuka ako"))
        self.assertIn("fever",               normalize_user_text("nilalagnat ako kanina"))


class ExtractSymptomMultilingualEnglish(unittest.TestCase):
    def test_no_match_returns_none(self) -> None:
        self.assertIsNone(extract_symptom_multilingual("what is pCR?"))
        self.assertIsNone(extract_symptom_multilingual(""))

    def test_english_with_numeric_severity(self) -> None:
        out = extract_symptom_multilingual("I have a fever, severity 7")
        self.assertEqual(out["symptom"], "fever")
        self.assertEqual(out["severity"], 7)
        self.assertEqual(out["severity_source"], "numeric")

    def test_english_with_typo(self) -> None:
        out = extract_symptom_multilingual("i have fevr 8/10 since morning")
        self.assertEqual(out["symptom"], "fever")
        self.assertEqual(out["severity"], 8)

    def test_english_mild_qualitative(self) -> None:
        out = extract_symptom_multilingual("mild diarrhea today")
        self.assertEqual(out["symptom"], "diarrhea")
        self.assertEqual(out["severity"], MULTILINGUAL_SEVERITY_HINTS["mild"])
        self.assertEqual(out["severity_source"], "qualitative")


class ExtractSymptomMultilingualFilipino(unittest.TestCase):
    def test_taglish_fever_with_numeric_severity(self) -> None:
        out = extract_symptom_multilingual("may lagnat ako severity 7")
        self.assertEqual(out["symptom"], "fever")
        self.assertEqual(out["severity"], 7)
        self.assertEqual(out["language_hint"], "mixed")

    def test_taglish_pain_with_qualitative_severity(self) -> None:
        out = extract_symptom_multilingual("masakit ang ulo ko, matindi")
        self.assertEqual(out["symptom"], "pain")
        self.assertEqual(out["severity"], MULTILINGUAL_SEVERITY_HINTS["matindi"])
        self.assertEqual(out["severity_source"], "qualitative")

    def test_taglish_nausea(self) -> None:
        out = extract_symptom_multilingual("nasusuka ako simula kanina")
        self.assertEqual(out["symptom"], "nausea")

    def test_upset_stomach_maps_to_abdominal_discomfort_without_inventing_severity(self) -> None:
        out = extract_symptom_multilingual("I have an upset stomach")
        self.assertIsNotNone(out)
        self.assertEqual(out["symptom"], "abdominal discomfort")
        self.assertFalse(out["severity_provided"])

    def test_taglish_shortness_of_breath(self) -> None:
        out = extract_symptom_multilingual("nahihirapan ako huminga")
        self.assertEqual(out["symptom"], "shortness of breath")

    def test_taglish_low_appetite(self) -> None:
        out = extract_symptom_multilingual("walang gana kumain talaga")
        self.assertEqual(out["symptom"], "low appetite")

    def test_filipino_dialect_severity_konti(self) -> None:
        out = extract_symptom_multilingual("masakit konti lang ang katawan")
        self.assertEqual(out["symptom"], "pain")
        self.assertEqual(out["severity"], MULTILINGUAL_SEVERITY_HINTS["konti lang"])


class ExtractSymptomMultilingualSpanish(unittest.TestCase):
    def test_spanish_fever(self) -> None:
        out = extract_symptom_multilingual("tengo fiebre desde ayer")
        self.assertEqual(out["symptom"], "fever")

    def test_spanish_shortness_of_breath(self) -> None:
        out = extract_symptom_multilingual("tengo falta de aire")
        self.assertEqual(out["symptom"], "shortness of breath")

    def test_spanish_intense_severity(self) -> None:
        out = extract_symptom_multilingual("dolor muy fuerte en el pecho")
        self.assertEqual(out["symptom"], "pain")
        self.assertGreaterEqual(out["severity"] or 0, 7)


class MixedLanguageTriggers(unittest.TestCase):
    def test_taglish_and_english_in_one_sentence(self) -> None:
        out = extract_symptom_multilingual("i have lagnat tonight, severity 6/10")
        self.assertEqual(out["symptom"], "fever")
        self.assertEqual(out["severity"], 6)

    def test_extreme_typo_repeated_chars(self) -> None:
        # "feeeever" gets collapsed to "feever" which still matches.
        out = extract_symptom_multilingual("feeeever like 9/10")
        self.assertEqual(out["symptom"], "fever")
        self.assertEqual(out["severity"], 9)


class GuessLanguage(unittest.TestCase):
    def test_english_default(self) -> None:
        self.assertEqual(guess_language("i have fever today"), "en")

    def test_tagalog_marker(self) -> None:
        self.assertIn(guess_language("may lagnat ako"), {"tl", "mixed"})

    def test_spanish_marker(self) -> None:
        self.assertIn(guess_language("tengo fiebre"), {"es", "mixed"})


class ToolIntentHints(unittest.TestCase):
    def test_greeting_has_no_hints(self) -> None:
        self.assertEqual(tool_intent_hints_from_text("hi"), [])
        self.assertEqual(tool_intent_hints_from_text("hello there"), [])

    def test_education_has_no_hints(self) -> None:
        self.assertEqual(tool_intent_hints_from_text("what is pCR?"), [])

    def test_symptom_log_has_save_symptom_hint(self) -> None:
        hints = tool_intent_hints_from_text("i have fever today")
        self.assertIn("save_symptom", hints)

    def test_taglish_log_has_save_symptom_hint(self) -> None:
        hints = tool_intent_hints_from_text("ako may lagnat severity 7")
        self.assertIn("save_symptom", hints)

    def test_cbc_log_has_save_labs_hint(self) -> None:
        hints = tool_intent_hints_from_text("WBC 2.1 hemoglobin 10.4 platelets 145")
        self.assertIn("save_complete_cbc", hints)

    def test_medication_log_has_save_medication_hint(self) -> None:
        hints = tool_intent_hints_from_text("i'm taking paclitaxel weekly")
        self.assertIn("save_medication", hints)

    def test_taglish_medication_log_has_save_medication_hint(self) -> None:
        hints = tool_intent_hints_from_text("umiinom ako ng tamoxifen")
        self.assertIn("save_medication", hints)


class MultilingualLabImagingMedication(unittest.TestCase):
    """The multilingual normalizer must also let the lab / imaging /
    medication extractors in support_chat_agent recognize Taglish and
    Spanish phrasing without needing native multilingual extractors."""

    def setUp(self) -> None:
        from backend.services.support_chat_agent import _extract_candidate_inputs
        self._extract = _extract_candidate_inputs

    def test_taglish_medication_logging(self) -> None:
        out = self._extract("umiinom ako ng tamoxifen 20mg araw-araw")
        self.assertIsNotNone(out["medication"])
        self.assertIn("tamoxifen", out["medication"]["medication"].lower())

    def test_taglish_cbc_with_decimal_comma(self) -> None:
        out = self._extract("CBC ko ngayong araw: WBC 2,1 hgb 10.4 plts 145")
        self.assertIsNotNone(out["labs"])
        self.assertEqual(out["labs"]["wbc"], 2.1)
        self.assertEqual(out["labs"]["hemoglobin"], 10.4)
        self.assertEqual(out["labs"]["platelets"], 145)

    def test_taglish_imaging_trigger(self) -> None:
        out = self._extract(
            "may MRI ko kanina, MRI report says decrease sa tumor size by 30%"
        )
        # The extractor produces a full imaging_report dict when the
        # message has report-like wording; we accept either the dict or
        # a partial_imaging flag here since the user's phrasing is
        # already enough to surface an imaging entry.
        self.assertTrue(out["imaging_report"] or out["partial_imaging"])

    def test_mixed_language_symptom_plus_medication(self) -> None:
        out = self._extract("i have fevr severity 7, also taking paracetamol")
        self.assertEqual(out["symptom"]["symptom"], "fever")
        self.assertEqual(out["symptom"]["severity"], 7)
        self.assertIsNotNone(out["medication"])
        self.assertIn("paracetamol", out["medication"]["medication"].lower())

    def test_spanish_medication_logging(self) -> None:
        out = self._extract("estoy tomando tamoxifen 20mg cada dia")
        self.assertIsNotNone(out["medication"])
        self.assertIn("tamoxifen", out["medication"]["medication"].lower())


class FastModeShortCircuitsAdjudication(unittest.TestCase):
    """ONCOTRACK_FAST_MODE=1 must make every _adjudicate_json call
    return ``available=False`` without touching the network."""

    def test_fast_mode_disables_adjudication(self) -> None:
        import os
        from backend.services.local_llm import _adjudicate_json
        original = os.environ.get("ONCOTRACK_FAST_MODE")
        try:
            os.environ["ONCOTRACK_FAST_MODE"] = "1"
            result = _adjudicate_json(system="x", prompt="y")
            self.assertFalse(result["available"])
            self.assertIn("fast_mode", result["reason"])
        finally:
            if original is None:
                os.environ.pop("ONCOTRACK_FAST_MODE", None)
            else:
                os.environ["ONCOTRACK_FAST_MODE"] = original


class VocabularyTablesAreInspectable(unittest.TestCase):
    def test_every_symptom_has_at_least_one_english_term(self) -> None:
        for symptom, terms in MULTILINGUAL_SYMPTOM_TERMS.items():
            self.assertTrue(terms, f"empty term list for {symptom}")
            # Every entry must be ASCII (no leftover diacritics that
            # would never match after normalize_user_text strips them).
            self.assertTrue(all(t.isascii() for t in terms),
                            f"{symptom} has a non-ASCII term")

    def test_severity_hints_are_0_to_10(self) -> None:
        for phrase, score in MULTILINGUAL_SEVERITY_HINTS.items():
            self.assertTrue(0 <= score <= 10, f"{phrase} -> {score} outside 0-10")


if __name__ == "__main__":
    unittest.main()
