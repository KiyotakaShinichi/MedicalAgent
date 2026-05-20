"""Tests for the LLM-backed multilingual compound-intent classifier.

We never hit the real Groq endpoint here — every test patches
``backend.services.local_llm._adjudicate_json`` so behavior is
deterministic + offline.  The tests pin:

  - LLM unavailable -> ``classify_compound_intent_with_llm`` returns None
  - Schema clamping: unknown kinds / tool_targets / primary_intent
    values get dropped instead of trusted blindly
  - Merge: LLM tool_request the heuristic missed gets added; the
    deterministic suggested_acknowledgment is preserved
  - Safety floor: a deterministic safety_boundary always wins, even
    if the LLM votes "data_entry_intention"
  - LRU cache: identical normalized messages produce a single LLM call
  - End-to-end: ``detect_compound_intents_with_llm(use_llm=False)`` is
    hermetic and identical to the deterministic-only call
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.services import compound_intent_router


def _fake_adj(verdict: dict):
    """Build a ``_adjudicate_json`` stand-in that returns the given verdict."""

    def _inner(*, system, prompt, tier="router"):
        return {"available": True, **verdict}

    return _inner


class ClassifyWhenLlmUnavailable(unittest.TestCase):
    def setUp(self) -> None:
        compound_intent_router._invalidate_llm_cache()

    def test_returns_none_when_adjudicator_unavailable(self) -> None:
        def unavailable(*, system, prompt, tier="router"):
            return {"available": False, "reason": "llm_adjudicator_disabled_by_fast_mode"}

        with patch("backend.services.local_llm._adjudicate_json", side_effect=unavailable):
            verdict = compound_intent_router.classify_compound_intent_with_llm(
                "hi, can you log my symptoms?"
            )
        self.assertIsNone(verdict)

    def test_returns_none_on_empty_input(self) -> None:
        verdict = compound_intent_router.classify_compound_intent_with_llm("")
        self.assertIsNone(verdict)

    def test_returns_none_when_llm_returns_no_segments(self) -> None:
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({"segments": []}),
        ):
            verdict = compound_intent_router.classify_compound_intent_with_llm("hi")
        self.assertIsNone(verdict)


class SchemaClamping(unittest.TestCase):
    def setUp(self) -> None:
        compound_intent_router._invalidate_llm_cache()

    def test_unknown_kind_is_dropped(self) -> None:
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({
                "language": "tl",
                "segments": [
                    {"kind": "casual_opener",    "span": "hi", "tool_targets": []},
                    {"kind": "bogus_kind",       "span": "x",  "tool_targets": []},
                ],
                "primary_intent": "conversation",
                "confidence": 0.9,
            }),
        ):
            verdict = compound_intent_router.classify_compound_intent_with_llm("hi")
        self.assertIsNotNone(verdict)
        kinds = [s["kind"] for s in verdict["segments"]]
        self.assertIn("casual_opener", kinds)
        self.assertNotIn("bogus_kind", kinds)

    def test_unknown_primary_intent_falls_back(self) -> None:
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({
                "segments": [{"kind": "casual_opener", "span": "hi", "tool_targets": []}],
                "primary_intent": "I_made_this_up",
                "confidence": 1.0,
            }),
        ):
            verdict = compound_intent_router.classify_compound_intent_with_llm("hi")
        self.assertEqual(verdict["primary_intent"], "general_support")

    def test_unknown_tool_target_is_dropped(self) -> None:
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({
                "segments": [
                    {"kind": "tool_request", "span": "log my X",
                     "tool_targets": ["save_symptom", "save_unicorn"]},
                ],
                "primary_intent": "data_entry_intention",
            }),
        ):
            verdict = compound_intent_router.classify_compound_intent_with_llm("log my X")
        seg = verdict["segments"][0]
        self.assertEqual(seg["tool_targets"], ["save_symptom"])

    def test_confidence_clamps_to_0_1(self) -> None:
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({
                "segments": [{"kind": "casual_opener", "span": "hi", "tool_targets": []}],
                "primary_intent": "conversation",
                "confidence": 17.5,
            }),
        ):
            verdict = compound_intent_router.classify_compound_intent_with_llm("hi")
        self.assertLessEqual(verdict["llm_confidence"], 1.0)
        self.assertGreaterEqual(verdict["llm_confidence"], 0.0)


class MergeBehavior(unittest.TestCase):
    def setUp(self) -> None:
        compound_intent_router._invalidate_llm_cache()

    def test_llm_adds_tool_request_heuristic_missed(self) -> None:
        # User writes Vietnamese — the heuristic tables don't cover
        # this, so deterministic returns general_support; the LLM
        # catches the tool request and we should end up with
        # data_entry_intention.
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({
                "language": "vi",
                "segments": [
                    {"kind": "tool_request",
                     "span": "ghi lại triệu chứng của tôi",
                     "tool_targets": ["save_symptom"]},
                ],
                "primary_intent": "data_entry_intention",
                "confidence": 0.85,
            }),
        ):
            envelope, raw = compound_intent_router.detect_compound_intents_with_llm(
                "ghi lại triệu chứng của tôi"
            )
        self.assertEqual(envelope.primary_intent, "data_entry_intention")
        self.assertTrue(envelope.has_tool_request)
        self.assertIn("save_symptom", envelope.tool_request_targets)
        self.assertIsNotNone(raw)
        self.assertEqual(raw["language"], "vi")

    def test_deterministic_safety_wins_over_llm(self) -> None:
        # Even if the LLM thinks the message is a tool request, a
        # deterministic safety_boundary (treatment-decision wording)
        # must keep the primary intent on safety_boundary so the
        # downstream pipeline routes to the refusal lane.
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({
                "segments": [
                    {"kind": "tool_request",
                     "span": "save this", "tool_targets": ["save_symptom"]},
                ],
                "primary_intent": "data_entry_intention",
                "confidence": 0.95,
            }),
        ):
            # Manually inject a deterministic safety segment to verify
            # the merge guard.
            from backend.services.compound_intent_router import (
                CompoundIntent, IntentSegment, merge_compound_intent_with_llm,
            )
            det = CompoundIntent(
                segments=[IntentSegment("safety_boundary", "safety_boundary", "should i stop chemo")],
                primary_intent="safety_boundary",
            )
            llm = compound_intent_router.classify_compound_intent_with_llm("should i stop chemo, save this")
            merged = merge_compound_intent_with_llm(det, llm)
        self.assertEqual(merged.primary_intent, "safety_boundary")

    def test_llm_preserves_deterministic_acknowledgment(self) -> None:
        with patch(
            "backend.services.local_llm._adjudicate_json",
            side_effect=_fake_adj({
                "segments": [{"kind": "casual_opener", "span": "hi", "tool_targets": []}],
                "primary_intent": "conversation",
                "casual_opener_acknowledgment": "LLM's own greeting",
            }),
        ):
            envelope, _ = compound_intent_router.detect_compound_intents_with_llm(
                "hi, can you log my symptoms?"
            )
        # Deterministic envelope already had its own acknowledgment for
        # the greeting+tool combo — it should not be replaced by the
        # LLM's bare "LLM's own greeting".
        self.assertIsNotNone(envelope.suggested_acknowledgment)
        self.assertNotEqual(envelope.suggested_acknowledgment, "LLM's own greeting")


class LruCache(unittest.TestCase):
    def setUp(self) -> None:
        compound_intent_router._invalidate_llm_cache()

    def test_identical_normalized_message_hits_cache(self) -> None:
        calls = {"count": 0}

        def fake_adj(*, system, prompt, tier="router"):
            calls["count"] += 1
            return {
                "available": True,
                "segments": [{"kind": "casual_opener", "span": "hi", "tool_targets": []}],
                "primary_intent": "conversation",
            }

        with patch("backend.services.local_llm._adjudicate_json", side_effect=fake_adj):
            v1 = compound_intent_router.classify_compound_intent_with_llm("hi")
            v2 = compound_intent_router.classify_compound_intent_with_llm("HI")  # normalized to "hi"
            v3 = compound_intent_router.classify_compound_intent_with_llm("hi   ")
        self.assertEqual(calls["count"], 1)
        self.assertEqual(v1, v2)
        self.assertEqual(v1, v3)


class UseLlmFlag(unittest.TestCase):
    def setUp(self) -> None:
        compound_intent_router._invalidate_llm_cache()

    def test_use_llm_false_skips_classifier(self) -> None:
        calls = {"count": 0}

        def fake_adj(*, system, prompt, tier="router"):
            calls["count"] += 1
            return {"available": True, "segments": [], "primary_intent": "conversation"}

        with patch("backend.services.local_llm._adjudicate_json", side_effect=fake_adj):
            envelope, raw = compound_intent_router.detect_compound_intents_with_llm(
                "hi, can you log my symptoms?",
                use_llm=False,
            )
        self.assertEqual(calls["count"], 0)
        self.assertIsNone(raw)
        # The deterministic envelope still gives us the compound result.
        self.assertEqual(envelope.primary_intent, "data_entry_intention")


if __name__ == "__main__":
    unittest.main()
