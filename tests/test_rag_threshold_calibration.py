"""Tests for the RAG-agent threshold sensitivity sweep.

Lock-ins:

* The report exposes every constant the critique flagged
  (SUPPORTED_THRESHOLD, WEAKLY_SUPPORTED_THRESHOLD, LLM_CONFIDENCE_FLOOR).
* The probe set has at least one supported, one weakly-supported, and
  one unsupported case so the sweep can actually flip statuses.
* No probe carries clinical claims that could be mistaken for advice.
"""
from __future__ import annotations

import re
import unittest

from backend.services.rag_threshold_calibration import (
    PROBES,
    build_calibration_report,
)


class CalibrationReport(unittest.TestCase):
    def setUp(self) -> None:
        self.report = build_calibration_report()

    def test_constants_present(self) -> None:
        keys = set(self.report["constants"].keys())
        self.assertSetEqual(
            keys,
            {"SUPPORTED_THRESHOLD", "WEAKLY_SUPPORTED_THRESHOLD", "LLM_CONFIDENCE_FLOOR"},
        )

    def test_probe_set_covers_three_status_classes(self) -> None:
        expected = {p["expected_default"] for p in PROBES}
        self.assertIn("supported", expected)
        self.assertIn("weakly_supported", expected)
        self.assertIn("unsupported", expected)

    def test_label_and_disclaimer(self) -> None:
        self.assertEqual(self.report["label"], "internal_engineering_eval_threshold_sensitivity")
        self.assertIn("engineering signal only", self.report["claim_boundary"])
        self.assertIn("does not validate", self.report["claim_boundary"])

    def test_no_clinical_advice_in_probes(self) -> None:
        # No probe sentence should look like prescription, dose, or
        # diagnosis wording.  Engineering probes only.
        forbidden = re.compile(
            r"\byou\s+should\b|"
            r"\btake\s+\d+\s*mg\b|"
            r"\bI\s+recommend\b|"
            r"\byou\s+have\s+\w+\s+cancer\b",
            re.IGNORECASE,
        )
        for p in PROBES:
            self.assertIsNone(forbidden.search(p["sentence"]), p["id"])

    def test_verdict_field_present_per_constant(self) -> None:
        for name, block in self.report["constants"].items():
            self.assertIn("verdict", block, name)


if __name__ == "__main__":
    unittest.main()
