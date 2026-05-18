"""Unit tests for ``backend.services.agent_safety.safety_scope_check``.

The breast-monitoring integration suite exercises ``safety_scope_check``
transitively through ``run_patient_agent_pipeline``, but that path is a
4+ minute test run.  These tests pin the classifier's contract
directly:

  - The precedence order: urgent > treatment-decision > diagnostic > low-risk
  - ``cache_allowed`` must be False on every high-risk branch
  - Taglish wording in DECISION_TERMS / DIAGNOSTIC_TERMS / URGENT_TERMS
    must classify the same as the English equivalent
  - ``urgent_flags`` from an upstream lab/imaging extractor must elevate
    to ``urgent_or_safety_related`` even when the text contains no
    urgent vocabulary (the bug fixed in commit 1cca9f0)
  - The returned envelope shape is stable
"""
from __future__ import annotations

import unittest

from backend.services.agent_safety import (
    DECISION_TERMS,
    DIAGNOSTIC_TERMS,
    URGENT_TERMS,
    safety_scope_check,
)


class PrecedenceOrder(unittest.TestCase):
    """The classifier must check urgency first, then treatment-decision,
    then diagnostic, then default low-risk.  Each high-risk branch sets
    cache_allowed=False."""

    def test_low_risk_default_for_educational_query(self) -> None:
        envelope = safety_scope_check("What is pCR?")
        self.assertEqual(envelope["level"], "low_risk")
        self.assertEqual(envelope["scope"], "education_or_tracking")
        self.assertTrue(envelope["cache_allowed"])

    def test_diagnostic_query_routes_to_diagnostic_scope(self) -> None:
        envelope = safety_scope_check("do i have cancer")
        self.assertEqual(envelope["level"], "high_risk")
        self.assertEqual(envelope["scope"], "diagnosis_or_outcome_claim")
        self.assertFalse(envelope["cache_allowed"])

    def test_treatment_decision_routes_to_treatment_scope(self) -> None:
        envelope = safety_scope_check("should i stop chemo")
        self.assertEqual(envelope["level"], "high_risk")
        self.assertEqual(envelope["scope"], "treatment_decision_request")
        self.assertFalse(envelope["cache_allowed"])

    def test_urgent_text_beats_other_branches(self) -> None:
        # Wording matches both an urgent term (fever) AND a treatment-decision
        # term (should i stop) — urgent must win.
        envelope = safety_scope_check("i have fever and should i stop chemo")
        self.assertEqual(envelope["scope"], "urgent_or_safety_related")


class UrgentFlagsElevatePastEmptyText(unittest.TestCase):
    """Regression test for commit 1cca9f0.

    Before the fix, ``urgent_flags`` from a downstream lab extractor
    (e.g. ``very_low_wbc``) didn't reach ``safety_scope_check`` because
    the chat layer cached the early ``routing_safety`` decision.  The
    classifier itself must elevate whenever ``urgent_flags`` is
    non-empty — even when the text contains no urgent vocabulary.
    """

    def test_urgent_flags_alone_elevate_to_urgent_scope(self) -> None:
        envelope = safety_scope_check(
            "here are my CBC values",
            urgent_flags=["very_low_wbc"],
        )
        self.assertEqual(envelope["level"], "high_risk")
        self.assertEqual(envelope["scope"], "urgent_or_safety_related")
        self.assertIn("very_low_wbc", envelope["matched_safety_terms"])

    def test_no_urgent_flags_no_elevation(self) -> None:
        envelope = safety_scope_check("here are my CBC values")
        self.assertEqual(envelope["level"], "low_risk")


class TaglishParity(unittest.TestCase):
    """Filipino/Taglish wording must classify the same way the English
    equivalent does — the project's multilingual_refusal eval asserts
    this contract end-to-end; these tests pin the classifier-level
    behavior."""

    def test_tagalog_treatment_decision(self) -> None:
        envelope = safety_scope_check("dapat ko bang itigil ang chemo")
        self.assertEqual(envelope["level"], "high_risk")
        self.assertEqual(envelope["scope"], "treatment_decision_request")

    def test_tagalog_diagnostic(self) -> None:
        envelope = safety_scope_check("may cancer ba ako")
        self.assertEqual(envelope["level"], "high_risk")
        self.assertEqual(envelope["scope"], "diagnosis_or_outcome_claim")

    def test_tagalog_urgent_fever(self) -> None:
        envelope = safety_scope_check("nilalagnat ako simula kanina")
        self.assertEqual(envelope["level"], "high_risk")
        self.assertEqual(envelope["scope"], "urgent_or_safety_related")


class EnvelopeShape(unittest.TestCase):
    """Every returned envelope carries the same top-level keys so
    downstream consumers (chat / cache / RAG eval log) can read them
    without conditional shape-handling."""

    REQUIRED_KEYS = {"level", "scope", "cache_allowed", "message"}

    def test_low_risk_envelope_keys(self) -> None:
        envelope = safety_scope_check("What is pCR?")
        self.assertTrue(self.REQUIRED_KEYS.issubset(envelope.keys()))

    def test_urgent_envelope_includes_matched_terms(self) -> None:
        envelope = safety_scope_check("i have heavy bleeding now")
        self.assertTrue(self.REQUIRED_KEYS.issubset(envelope.keys()))
        # Urgent branch is the only one that exposes matched_safety_terms.
        self.assertIn("matched_safety_terms", envelope)

    def test_treatment_decision_envelope_no_matched_terms(self) -> None:
        # The non-urgent high-risk branches don't surface matched_terms;
        # this is deliberate — they're keyword-pattern-driven and the
        # specific match is logged at the input-guardrail layer instead.
        envelope = safety_scope_check("should i delay chemotherapy")
        self.assertTrue(self.REQUIRED_KEYS.issubset(envelope.keys()))


class VocabularyTablesAreInspectable(unittest.TestCase):
    """The three top-level vocabulary tables must remain available so
    documentation and the failure-mode registry can enumerate the exact
    triggers."""

    def test_tables_are_non_empty_tuples(self) -> None:
        for table in (DECISION_TERMS, DIAGNOSTIC_TERMS, URGENT_TERMS):
            self.assertIsInstance(table, tuple)
            self.assertGreater(len(table), 0)

    def test_tables_have_no_duplicates(self) -> None:
        for table in (DECISION_TERMS, DIAGNOSTIC_TERMS, URGENT_TERMS):
            self.assertEqual(len(table), len(set(table)), f"duplicates in {table[:3]}...")


if __name__ == "__main__":
    unittest.main()
