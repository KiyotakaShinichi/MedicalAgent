"""Unit tests for the split helpers extracted from `_finalize_result`.

`_apply_post_gen_validator` and `_apply_intent_aware_rag_layer` are pure
functions that mutate the `result` dict in place.  Testing them directly
is much faster + clearer than going through a full agent call, and
guarantees the refactor preserved behavior on every code path the
original monolithic function covered:

  - validator allows → result["post_gen_validator"]["decision"] == "allowed"
  - validator blocks → reply substituted with safe refusal, citations
    stripped, output_guardrails["status"] flipped to
    "blocked_by_post_gen_validator", issues list extended with
    "post_gen::<rule>" entries
  - intent-aware layer with no mode → does nothing, no exception
  - intent-aware layer with insufficient grade + non-blocked validator
    → substitutes the mode's insufficient_evidence_default reply
  - intent-aware layer raises → evidence_grade.grade == "missing"
    (never crashes chat)
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.services.agent_rag import (
    _apply_intent_aware_rag_layer,
    _apply_post_gen_validator,
)


# ─── Post-gen validator helper ───────────────────────────────────────────────


class PostGenValidatorAllowsPath(unittest.TestCase):
    def test_safe_reply_leaves_result_intact(self) -> None:
        result = {"reply": "WBC stands for white blood cells. They help fight infection.", "citations": ["src_a"]}
        guardrails = {"status": "passed", "issues": []}
        out_guardrails, decision = _apply_post_gen_validator(result, guardrails)
        self.assertEqual(decision.decision, "allowed")
        # Reply + citations untouched.
        self.assertEqual(result["reply"], "WBC stands for white blood cells. They help fight infection.")
        self.assertEqual(result["citations"], ["src_a"])
        # post_gen_validator block records the allow decision.
        self.assertEqual(result["post_gen_validator"]["decision"], "allowed")
        self.assertEqual(result["post_gen_validator"]["triggered_rules"], [])
        # output_guardrails returned unchanged.
        self.assertEqual(out_guardrails["status"], "passed")


class PostGenValidatorBlocksPath(unittest.TestCase):
    def test_diagnosis_claim_is_blocked_and_reply_replaced(self) -> None:
        result = {
            "reply": "Based on your symptoms, you have breast cancer.",
            "citations": ["src_should_be_stripped"],
        }
        guardrails = {"status": "passed", "issues": []}
        out_guardrails, decision = _apply_post_gen_validator(result, guardrails)

        self.assertEqual(decision.decision, "blocked")
        self.assertIn("diagnosis_claim", decision.triggered_rules)
        # Reply was substituted with the safe refusal.
        self.assertNotEqual(result["reply"], "Based on your symptoms, you have breast cancer.")
        # Citations stripped.
        self.assertEqual(result["citations"], [])
        # post_gen_validator block reflects the block + carries excerpt + original preview.
        block = result["post_gen_validator"]
        self.assertEqual(block["decision"], "blocked")
        self.assertIn("diagnosis_claim", block["triggered_rules"])
        self.assertIn("breast cancer", block["original_reply_preview"])
        # output_guardrails status flipped + post_gen rules surfaced.
        self.assertEqual(out_guardrails["status"], "blocked_by_post_gen_validator")
        self.assertIn("post_gen::diagnosis_claim", out_guardrails["issues"])

    def test_blocking_creates_a_new_guardrails_dict(self) -> None:
        """The helper must not mutate the original output_guardrails dict
        in place — caller may still hold the original reference."""
        result = {"reply": "Take 200 mg twice a day."}
        original_guardrails = {"status": "passed", "issues": ["seed"]}
        out_guardrails, _ = _apply_post_gen_validator(result, original_guardrails)
        # Original untouched (still has only the seed issue).
        self.assertEqual(original_guardrails["status"], "passed")
        self.assertEqual(original_guardrails["issues"], ["seed"])
        # New dict has the block status.
        self.assertEqual(out_guardrails["status"], "blocked_by_post_gen_validator")
        self.assertIn("post_gen::dosage_instruction", out_guardrails["issues"])


# ─── Intent-aware RAG layer helper ───────────────────────────────────────────


class _StubDecision:
    """Stand-in for ValidatorDecision so the intent-aware tests don't
    need to construct one — only `decision` is read."""
    def __init__(self, decision: str = "allowed") -> None:
        self.decision = decision


class IntentAwareLayerNoOp(unittest.TestCase):
    def test_unknown_intent_skips_layer_silently(self) -> None:
        """`select_mode` returns None for non-RAG intents; the helper must
        not touch the result dict in that case."""
        result = {"intent": "conversation", "reply": "Hi there!"}
        _apply_intent_aware_rag_layer(result, retrieved=[], input_guardrails={}, pgv_decision=_StubDecision())
        # No mode-related fields written.
        for key in ("rag_mode", "tier_filter", "claim_validation", "evidence_grade"):
            self.assertNotIn(key, result)


class IntentAwareLayerWritesFields(unittest.TestCase):
    def test_education_intent_populates_grade_and_mode_fields(self) -> None:
        result = {
            "intent": "education",
            "reply": "WBC stands for white blood cells.",
            "retrieval_context": [],  # no chunks → insufficient grade
            "citations": ["c1"],
        }
        _apply_intent_aware_rag_layer(result, retrieved=[], input_guardrails={}, pgv_decision=_StubDecision())
        self.assertEqual(result["rag_mode"], "education_rag")
        self.assertIn("evidence_grade", result)
        self.assertIn("tier_filter", result)
        self.assertIn("claim_validation", result)
        # No chunks → insufficient grade → reply substituted to the mode's default.
        self.assertEqual(result["evidence_grade"]["grade"], "insufficient")
        self.assertIn("insufficient_evidence_substitution", result)
        # Citations stripped on the substitution.
        self.assertEqual(result["citations"], [])

    def test_blocked_validator_blocks_insufficient_substitution(self) -> None:
        """When the post-gen validator already blocked the reply, the
        insufficient-evidence layer must NOT overwrite with its own
        default — the validator's safe refusal wins."""
        result = {
            "intent": "education",
            "reply": "Safe refusal text from validator.",
            "retrieval_context": [],
            "citations": [],
        }
        _apply_intent_aware_rag_layer(
            result, retrieved=[], input_guardrails={},
            pgv_decision=_StubDecision(decision="blocked"),
        )
        # Reply NOT overwritten by the mode default.
        self.assertEqual(result["reply"], "Safe refusal text from validator.")
        # But the grade is still recorded for the trace.
        self.assertEqual(result["evidence_grade"]["grade"], "insufficient")
        # No substitution receipt.
        self.assertNotIn("insufficient_evidence_substitution", result)


class IntentAwareLayerNeverCrashes(unittest.TestCase):
    def test_internal_exception_records_missing_grade(self) -> None:
        """If any of the four sub-services raises, the helper logs a
        `missing` grade and returns — never propagates the exception."""
        result = {"intent": "education", "reply": "x", "retrieval_context": []}
        with patch(
            "backend.services.rag_intent_modes.select_mode",
            side_effect=RuntimeError("simulated upstream failure"),
        ):
            _apply_intent_aware_rag_layer(result, retrieved=[], input_guardrails={}, pgv_decision=_StubDecision())
        self.assertEqual(result["evidence_grade"]["grade"], "missing")
        self.assertIn("simulated upstream failure", result["evidence_grade"]["reasoning"])


class IntentAwareLayerActorRoleUpgrade(unittest.TestCase):
    def test_clinician_actor_upgrades_education_to_clinician_context(self) -> None:
        result = {
            "intent": "education",
            "reply": "x",
            "retrieval_context": [],
            "actor_role": "clinician",
        }
        _apply_intent_aware_rag_layer(result, retrieved=[], input_guardrails={}, pgv_decision=_StubDecision())
        self.assertEqual(result["rag_mode"], "clinician_context_rag")


if __name__ == "__main__":
    unittest.main()
