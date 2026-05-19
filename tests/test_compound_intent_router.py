"""Tests for ``backend.services.compound_intent_router``.

Pins the contract that ``detect_compound_intents`` returns:

  - the right ``primary_intent`` precedence (tool > education >
    capability > vocab > conversation > general)
  - segment kinds for casual openers, tool requests, education
    requests, capability requests, and vocab fallbacks
  - tool target inference for English / Taglish / Spanish
  - compound vs single-intent flagging
  - suggested acknowledgment text for casual+tool / casual+education
    combos
  - that bare safety-boundary phrasing (e.g. "should i stop chemo") is
    NOT classified as a tool request by mistake
"""
from __future__ import annotations

import unittest

from backend.services.compound_intent_router import (
    CompoundIntent,
    IntentSegment,
    detect_compound_intents,
)


class CasualOpener(unittest.TestCase):
    def test_bare_hi(self) -> None:
        out = detect_compound_intents("hi")
        self.assertEqual(out.primary_intent, "conversation")
        self.assertTrue(out.has_casual_opener)
        self.assertFalse(out.has_tool_request)
        self.assertFalse(out.is_compound)

    def test_taglish_kamusta(self) -> None:
        out = detect_compound_intents("kamusta")
        self.assertEqual(out.primary_intent, "conversation")
        self.assertTrue(out.has_casual_opener)

    def test_identity_question_is_conversation(self) -> None:
        out = detect_compound_intents("who are you")
        self.assertEqual(out.primary_intent, "conversation")


class GreetingPlusToolRequest(unittest.TestCase):
    def test_hi_can_you_log_my_symptoms(self) -> None:
        out = detect_compound_intents("hi, can you log my symptoms?")
        self.assertTrue(out.is_compound)
        self.assertTrue(out.has_casual_opener)
        self.assertTrue(out.has_tool_request)
        self.assertEqual(out.primary_intent, "data_entry_intention")
        self.assertEqual(out.tool_request_targets, ["save_symptom"])
        self.assertIsNotNone(out.suggested_acknowledgment)

    def test_hello_please_save_my_cbc(self) -> None:
        out = detect_compound_intents("hello! please save my CBC")
        self.assertTrue(out.is_compound)
        self.assertEqual(out.primary_intent, "data_entry_intention")
        self.assertIn("save_complete_cbc", out.tool_request_targets)

    def test_thanks_log_my_mri_report(self) -> None:
        out = detect_compound_intents("thanks, log my MRI report")
        self.assertTrue(out.has_tool_request)
        self.assertIn("save_imaging_report", out.tool_request_targets)


class TaglishToolRequests(unittest.TestCase):
    def test_taglish_greeting_plus_log(self) -> None:
        out = detect_compound_intents("kamusta, gusto kong ilog ang lagnat ko severity 7")
        self.assertTrue(out.is_compound)
        self.assertEqual(out.primary_intent, "data_entry_intention")
        self.assertIn("save_symptom", out.tool_request_targets)

    def test_taglish_imperative_isave_medication(self) -> None:
        out = detect_compound_intents("isave mo ang gamot ko")
        self.assertTrue(out.has_tool_request)
        self.assertIn("save_medication", out.tool_request_targets)

    def test_taglish_pwede_mo_bang(self) -> None:
        out = detect_compound_intents("pwede mo bang i-log ang sintomas ko")
        self.assertTrue(out.has_tool_request)
        self.assertIn("save_symptom", out.tool_request_targets)


class SpanishToolRequests(unittest.TestCase):
    def test_spanish_registrar_medication(self) -> None:
        out = detect_compound_intents("registrar mi medicación")
        self.assertTrue(out.has_tool_request)
        self.assertIn("save_medication", out.tool_request_targets)

    def test_spanish_guardar_labs(self) -> None:
        out = detect_compound_intents("guardar mis laboratorios")
        self.assertTrue(out.has_tool_request)


class GreetingPlusEducation(unittest.TestCase):
    def test_hi_what_is_pcr(self) -> None:
        out = detect_compound_intents("hi, what is pCR?")
        self.assertTrue(out.is_compound)
        self.assertTrue(out.has_casual_opener)
        self.assertTrue(out.has_education_request)
        self.assertEqual(out.primary_intent, "education")
        self.assertIsNotNone(out.suggested_acknowledgment)

    def test_education_request_alone(self) -> None:
        out = detect_compound_intents("what is pCR?")
        self.assertEqual(out.primary_intent, "education")
        self.assertFalse(out.has_casual_opener)
        self.assertFalse(out.is_compound)


class MultipleSegments(unittest.TestCase):
    def test_tool_plus_education(self) -> None:
        out = detect_compound_intents("log my symptom and explain what neutropenia is")
        self.assertTrue(out.is_compound)
        self.assertTrue(out.has_tool_request)
        self.assertTrue(out.has_education_request)
        # Tool wins.
        self.assertEqual(out.primary_intent, "data_entry_intention")


class SafetyBoundaryNotMistakenForTool(unittest.TestCase):
    def test_treatment_decision_does_not_become_tool_request(self) -> None:
        out = detect_compound_intents("should i stop chemo?")
        self.assertFalse(out.has_tool_request)
        # Safety is enforced by the upstream safety_scope_check; this
        # router just doesn't pretend the message is a tool request.

    def test_diagnostic_does_not_become_tool_request(self) -> None:
        out = detect_compound_intents("do i have cancer?")
        self.assertFalse(out.has_tool_request)


class EmptyAndDegenerate(unittest.TestCase):
    def test_empty_message_is_general_support(self) -> None:
        out = detect_compound_intents("")
        self.assertEqual(out.primary_intent, "general_support")

    def test_whitespace_only_is_general_support(self) -> None:
        out = detect_compound_intents("   \n  ")
        self.assertEqual(out.primary_intent, "general_support")


class Serialization(unittest.TestCase):
    def test_compound_intent_to_dict_has_required_keys(self) -> None:
        out = detect_compound_intents("hi, can you log my symptoms?")
        d = out.to_dict()
        for key in (
            "segments", "primary_intent", "is_compound",
            "has_casual_opener", "has_tool_request",
            "has_education_request", "tool_request_targets",
            "suggested_acknowledgment",
        ):
            self.assertIn(key, d)

    def test_segment_to_dict_has_required_keys(self) -> None:
        out = detect_compound_intents("hi, can you log my symptoms?")
        for seg in out.segments:
            d = seg.to_dict()
            for key in ("intent", "kind", "span", "tool_targets"):
                self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main()
