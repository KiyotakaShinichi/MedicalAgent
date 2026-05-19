"""Tests for the optional post-gen 120B escalation.

The behavior under test is gated by
``ONCOTRACK_POSTGEN_ANSWER_ESCALATION=1`` AND the reply containing one
of the borderline patterns.  When both conditions hold, the post-gen
layer calls the answer tier and respects its verdict.

We don't exercise the real Groq call here — every test patches
``_adjudicate_json`` so they stay deterministic + offline.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.services import agent_post_gen


class EscalationDisabledByDefault(unittest.TestCase):
    def setUp(self) -> None:
        self._original = os.environ.get("ONCOTRACK_POSTGEN_ANSWER_ESCALATION")
        os.environ.pop("ONCOTRACK_POSTGEN_ANSWER_ESCALATION", None)

    def tearDown(self) -> None:
        if self._original is None:
            os.environ.pop("ONCOTRACK_POSTGEN_ANSWER_ESCALATION", None)
        else:
            os.environ["ONCOTRACK_POSTGEN_ANSWER_ESCALATION"] = self._original

    def test_no_escalation_when_env_var_unset(self) -> None:
        result = {"reply": "in your case this likely means recurrence."}
        verdict = agent_post_gen._maybe_escalate_to_answer_tier(result)
        self.assertIsNone(verdict)


class EscalationFiresOnBorderline(unittest.TestCase):
    def setUp(self) -> None:
        self._original = os.environ.get("ONCOTRACK_POSTGEN_ANSWER_ESCALATION")
        os.environ["ONCOTRACK_POSTGEN_ANSWER_ESCALATION"] = "1"

    def tearDown(self) -> None:
        if self._original is None:
            os.environ.pop("ONCOTRACK_POSTGEN_ANSWER_ESCALATION", None)
        else:
            os.environ["ONCOTRACK_POSTGEN_ANSWER_ESCALATION"] = self._original

    def test_borderline_reply_calls_answer_tier(self) -> None:
        result = {"reply": "in your case, no need to worry — would be safe to skip the next cycle."}
        captured: dict = {}

        def fake_adjudicate(*, system, prompt, tier="router"):
            captured["tier"] = tier
            return {
                "available": True,
                "decision": "blocked",
                "triggered_rules": ["llm_treatment_directive_borderline"],
                "matched_excerpts": ["skip the next cycle"],
                "suggested_response": "I cannot advise on changing your treatment plan; please contact your oncology care team.",
                "confidence": 0.92,
            }

        with patch("backend.services.local_llm._adjudicate_json", side_effect=fake_adjudicate):
            verdict = agent_post_gen._maybe_escalate_to_answer_tier(result)
        self.assertIsNotNone(verdict)
        self.assertEqual(captured["tier"], "answer")
        self.assertEqual(verdict["decision"], "blocked")
        self.assertGreater(verdict["confidence"], 0.5)
        self.assertTrue(verdict["borderline_patterns_matched"])

    def test_non_borderline_reply_skips_escalation(self) -> None:
        result = {"reply": "General information: pCR means pathologic complete response. Discuss with your care team."}
        verdict = agent_post_gen._maybe_escalate_to_answer_tier(result)
        self.assertIsNone(verdict)


class ApplyPostGenWithEscalation(unittest.TestCase):
    """End-to-end: apply_post_gen_validator with a borderline reply +
    escalation enabled + LLM voting 'blocked' should rewrite the reply
    and update post_gen_validator with the escalation block."""

    def setUp(self) -> None:
        self._original = os.environ.get("ONCOTRACK_POSTGEN_ANSWER_ESCALATION")
        os.environ["ONCOTRACK_POSTGEN_ANSWER_ESCALATION"] = "1"

    def tearDown(self) -> None:
        if self._original is None:
            os.environ.pop("ONCOTRACK_POSTGEN_ANSWER_ESCALATION", None)
        else:
            os.environ["ONCOTRACK_POSTGEN_ANSWER_ESCALATION"] = self._original

    def test_borderline_block_rewrites_reply(self) -> None:
        result = {
            "reply": "in your case, no need to worry — would be safe to skip the next cycle.",
            "citations": [],
            "intent": "education",
            "safety": {"level": "low_risk"},
        }

        def fake_adjudicate(*, system, prompt, tier="router"):
            return {
                "available": True,
                "decision": "blocked",
                "triggered_rules": ["llm_treatment_directive_borderline"],
                "suggested_response": "I cannot advise on changing treatment plans; please contact your oncology care team.",
                "confidence": 0.95,
            }

        original_reply = result["reply"]
        with patch("backend.services.local_llm._adjudicate_json", side_effect=fake_adjudicate):
            agent_post_gen.apply_post_gen_validator(result, output_guardrails={"status": "passed", "issues": []})

        self.assertNotEqual(result["reply"], original_reply)
        self.assertEqual(result["post_gen_validator"]["decision"], "blocked")

    def test_borderline_allowed_keeps_reply(self) -> None:
        result = {
            "reply": "in your case, the next cycle is what your team should advise on.",
            "citations": [],
            "intent": "education",
            "safety": {"level": "low_risk"},
        }

        def fake_adjudicate(*, system, prompt, tier="router"):
            return {
                "available": True,
                "decision": "allowed",
                "triggered_rules": [],
                "confidence": 0.88,
            }

        original_reply = result["reply"]
        with patch("backend.services.local_llm._adjudicate_json", side_effect=fake_adjudicate):
            agent_post_gen.apply_post_gen_validator(result, output_guardrails={"status": "passed", "issues": []})

        self.assertEqual(result["reply"], original_reply)
        self.assertEqual(result["post_gen_validator"]["decision"], "allowed")
        self.assertIsNotNone(result["post_gen_validator"].get("answer_tier_escalation"))


if __name__ == "__main__":
    unittest.main()
