"""Tests for ``agent_turn_trace``.

Lock-ins:

* ``TurnTrace.to_dict`` only emits the whitelisted top-level keys.
* ``_scrub_cot`` actually strips chain-of-thought-suspect keys.
* ``validate_trace_payload`` flags both unknown top-level keys AND
  nested CoT-suspect keys.
* ``correlation_id`` is non-empty when generated.
"""
from __future__ import annotations

import unittest

from backend.services.agent_turn_trace import (
    COT_DENYLIST,
    TURN_TRACE_TOP_LEVEL_KEYS,
    build_turn_trace,
    validate_trace_payload,
)
from backend.services.trace_diagnostics_coverage import build_trace_diagnostics_coverage


class BuildTrace(unittest.TestCase):
    def test_minimal_trace_has_required_keys(self) -> None:
        trace = build_turn_trace(
            safety_scope={"level": "low_risk", "scope": "education_or_tracking"},
            intent={"deterministic_intent": "education", "llm_confidence": 0.0},
        )
        payload = trace.to_dict()
        self.assertIn("schema_version", payload)
        self.assertIn("correlation_id", payload)
        self.assertIn("generated_at", payload)
        self.assertIn("safety_scope", payload)
        self.assertIn("intent", payload)
        self.assertNotIn("emotional_distress", payload)  # empty -> dropped

    def test_correlation_id_non_empty(self) -> None:
        trace = build_turn_trace()
        self.assertTrue(trace.correlation_id)
        self.assertGreater(len(trace.correlation_id), 10)

    def test_no_unexpected_top_level_keys(self) -> None:
        trace = build_turn_trace(
            safety_scope={"level": "low_risk"},
            intent={"deterministic_intent": "education"},
            retrieval_summary={"answerability_status": "answerable_with_citations"},
        )
        payload = trace.to_dict()
        for key in payload.keys():
            self.assertIn(key, TURN_TRACE_TOP_LEVEL_KEYS, key)


class ScrubChainOfThought(unittest.TestCase):
    def test_thinking_key_stripped(self) -> None:
        trace = build_turn_trace(
            intent={"deterministic_intent": "education", "thinking": "I considered..."},
        )
        self.assertNotIn("thinking", trace.intent)
        self.assertIn("deterministic_intent", trace.intent)

    def test_nested_cot_stripped(self) -> None:
        trace = build_turn_trace(
            retrieval_summary={
                "answerability_status": "answerable_with_citations",
                "internal_monologue": "I should ...",
                "details": {"scratchpad": "..."},
            },
        )
        self.assertNotIn("internal_monologue", trace.retrieval_summary)
        self.assertNotIn("scratchpad", trace.retrieval_summary["details"])

    def test_denylist_completeness(self) -> None:
        # Catch a contributor who weakens the list.
        for token in ("thinking", "chain_of_thought", "cot", "scratchpad"):
            self.assertIn(token, COT_DENYLIST)


class ValidatePayload(unittest.TestCase):
    def test_unknown_top_level_key_flagged(self) -> None:
        bad = {"correlation_id": "x", "schema_version": "1.0", "generated_at": "now",
               "raw_response_text": "hello"}
        ok, problems = validate_trace_payload(bad)
        self.assertFalse(ok)
        self.assertTrue(any(p.startswith("unexpected_top_level_key") for p in problems))

    def test_nested_cot_flagged(self) -> None:
        bad = {
            "correlation_id": "x",
            "schema_version": "1.0",
            "generated_at": "now",
            "intent": {"deterministic_intent": "education", "thinking_text": "..."},
        }
        ok, problems = validate_trace_payload(bad)
        self.assertFalse(ok)
        self.assertTrue(any("cot_suspect_key" in p for p in problems))

    def test_clean_payload_passes(self) -> None:
        ok, problems = validate_trace_payload(build_turn_trace(
            safety_scope={"level": "low_risk", "scope": "education_or_tracking"},
            intent={"deterministic_intent": "education"},
        ).to_dict())
        self.assertTrue(ok, problems)


class CoverageArtifact(unittest.TestCase):
    def test_trace_diagnostics_coverage_artifact_shape(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            report = build_trace_diagnostics_coverage(output_path=Path(tmp) / "coverage.json", db=None, limit=1)
        self.assertEqual(report["schema_version"], "trace_diagnostics_coverage_v1")
        self.assertIn(report["status"], {"strong", "needs_attention"})
        self.assertFalse(report["summary"]["private_chain_of_thought_allowed"])
        self.assertTrue(report["summary"]["sample_trace_schema_valid"])


if __name__ == "__main__":
    unittest.main()
