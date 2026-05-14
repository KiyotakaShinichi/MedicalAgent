"""Tests for the Safety & Evaluation Center service layer.

These tests focus on the deterministic shape and behavior of the new
safety-red-team, RAG-eval, drift, and failure-case-gallery services.
They are hermetic: no real KB, no real DB, no real LLM. The safety
red-team and RAG eval suites build their own in-memory SQLite DBs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.services.drift_monitoring import build_drift_report
from backend.services.failure_case_gallery import load_failure_case_gallery
from backend.services.rag_eval_suite import run_rag_eval_suite
from backend.services.safety_red_team import (
    _default_cases,
    load_safety_red_team_cases,
    run_safety_red_team_suite,
)


class SafetyRedTeamSuiteTests(unittest.TestCase):
    def test_case_catalog_loads_with_required_fields(self):
        cases = load_safety_red_team_cases()
        self.assertGreater(len(cases), 0)
        required = {"id", "category", "input"}
        for case in cases:
            self.assertTrue(required.issubset(case.keys()), f"missing fields in {case}")

    def test_falls_back_to_default_cases_when_file_missing(self):
        cases = load_safety_red_team_cases(path="this/path/does/not/exist.json")
        self.assertEqual(cases, _default_cases())

    def test_suite_writes_artifact_with_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "safety.json"
            csv_output = Path(tmp) / "safety.csv"
            payload = run_safety_red_team_suite(
                output_path=str(output),
                csv_path=str(csv_output),
                cases=_default_cases(),
            )

            self.assertEqual(payload["schema_version"], "safety_red_team_eval_v1")
            self.assertIn("summary", payload)
            self.assertIn("cases", payload)
            self.assertGreaterEqual(payload["case_count"], 1)
            summary = payload["summary"]
            self.assertIn("pass_rate", summary)
            self.assertIn("category_counts", summary)
            self.assertIn("refusal_type_counts", summary)
            self.assertTrue(output.exists(), "JSON artifact should be written")
            self.assertTrue(csv_output.exists(), "CSV artifact should be written")

            # Round-trip the artifact
            reloaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["case_count"], payload["case_count"])

    def test_each_case_records_pass_reason_and_timestamp(self):
        payload = run_safety_red_team_suite(
            output_path=None, csv_path=None, cases=_default_cases()
        )
        for case in payload["cases"]:
            self.assertIn("pass", case)
            self.assertIn("timestamp", case)
            self.assertIn("observed", case)
            self.assertIsInstance(case["checks"], list)
            if not case["pass"]:
                self.assertIsNotNone(case["reason"])


class RagEvalSuiteTests(unittest.TestCase):
    def test_suite_runs_against_default_case_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.json"
            payload = run_rag_eval_suite(output_path=str(output))
            self.assertEqual(payload["schema_version"], "rag_eval_suite_v1")
            self.assertIn("summary", payload)
            self.assertIn("cases", payload)
            summary = payload["summary"]
            for key in (
                "pass_rate",
                "citation_coverage_rate",
                "expected_source_hit_rate",
                "refusal_correct_rate",
                "average_grounding_score",
            ):
                self.assertIn(key, summary)
            self.assertTrue(output.exists())

    def test_each_case_emits_metrics_block(self):
        payload = run_rag_eval_suite(output_path=None)
        for case in payload["cases"]:
            self.assertIn("metrics", case)
            self.assertIn("pass", case)
            self.assertIn("checks", case)


class DriftReportTests(unittest.TestCase):
    def test_report_shape_when_synthetic_inputs_present(self):
        payload = build_drift_report(output_path=None)
        self.assertEqual(payload["schema_version"], "drift_report_v1")
        # Either we have a full report or an unavailable one — both shapes
        # must be safe for the dashboard to consume.
        if payload.get("status") == "unavailable":
            self.assertIn("message", payload)
            return
        self.assertIn("lab_distribution_shift", payload)
        self.assertIn("symptom_frequency_shift", payload)
        self.assertIn("model_confidence_drift", payload)
        self.assertIn("calibration_drift", payload)
        self.assertIn("subgroup_performance_drift", payload)
        self.assertIn("data_completeness_score", payload)


class FailureCaseGalleryTests(unittest.TestCase):
    def test_gallery_loads_or_reports_not_generated(self):
        payload = load_failure_case_gallery()
        # Either the seeded gallery exists with cases, or we report not_generated.
        self.assertIn("schema_version", payload)
        self.assertIn("cases", payload)
        if payload.get("status") == "not_generated":
            self.assertEqual(payload["cases"], [])
        else:
            self.assertGreater(len(payload["cases"]), 0)
            for case in payload["cases"]:
                for required in (
                    "id",
                    "category",
                    "what_happened",
                    "why_risky",
                    "system_response",
                    "mitigation",
                    "unresolved",
                ):
                    self.assertIn(required, case)

    def test_missing_gallery_returns_not_generated(self):
        payload = load_failure_case_gallery(path="does/not/exist.json")
        self.assertEqual(payload["status"], "not_generated")
        self.assertEqual(payload["cases"], [])


if __name__ == "__main__":
    unittest.main()
