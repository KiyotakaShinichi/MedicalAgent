"""Tests for ``EvalDriftTracker``.

Lock-ins:

* A fresh history file produces a zero-row drift report without error.
* Appending two records lets the drift report compute deltas.
* A simulated regression (latest pass_rate drops well below previous)
  is flagged via ``is_regression`` AND appears in ``regressions``.
* Real Data/evals artifacts (if present) can be extracted without
  raising — tests skip the source if its file is missing so they
  remain hermetic across machines.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.services.eval_drift_tracker import (
    EvalDriftTracker,
    METRIC_SOURCES,
)


class _IsolatedTracker(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.tmpdir = Path(self._temporary_directory.name)
        self.history = self.tmpdir / "eval_history.jsonl"
        self.report = self.tmpdir / "latest_eval_drift_report.json"
        self.tracker = EvalDriftTracker(history_path=self.history, report_path=self.report)

    def _write_record(self, metrics: dict, *, release_id: str) -> None:
        record = {
            "release_id": release_id,
            "commit_hash": "x" * 7,
            "timestamp": "2026-05-20T00:00:00+00:00",
            "missing_sources": [],
            "metrics": metrics,
        }
        with self.history.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


class EmptyHistory(_IsolatedTracker):
    def test_empty_history_drift_report_is_zero(self) -> None:
        report = self.tracker.write_drift_report()
        self.assertEqual(report["n_records"], 0)
        self.assertEqual(report["deltas"], [])
        self.assertEqual(report["regressions"], [])


class DeltaAndRegression(_IsolatedTracker):
    def test_regression_flagged(self) -> None:
        self._write_record(
            {"adversarial_safety_regression": {
                "overall_attack_block_rate": 0.90,
                "urgent_symptom_rate": 1.0,
                "negative_control_rate": 0.96,
            }},
            release_id="prev",
        )
        self._write_record(
            {"adversarial_safety_regression": {
                "overall_attack_block_rate": 0.80,  # 10pp drop → regression
                "urgent_symptom_rate": 1.0,
                "negative_control_rate": 0.96,
            }},
            release_id="latest",
        )
        report = self.tracker.write_drift_report()
        self.assertEqual(report["n_records"], 2)
        keys = {d["metric"] for d in report["deltas"]}
        self.assertIn("adversarial_safety_regression.overall_attack_block_rate", keys)
        regression_metrics = {r["metric"] for r in report["regressions"]}
        self.assertIn("adversarial_safety_regression.overall_attack_block_rate", regression_metrics)

    def test_improvement_not_flagged(self) -> None:
        self._write_record(
            {"adversarial_safety_regression": {"overall_attack_block_rate": 0.70}},
            release_id="prev",
        )
        self._write_record(
            {"adversarial_safety_regression": {"overall_attack_block_rate": 0.85}},
            release_id="latest",
        )
        report = self.tracker.write_drift_report()
        self.assertEqual(report["regression_count"], 0)


class RealArtifactExtraction(unittest.TestCase):
    """Smoke-test that the extractors don't blow up on real artifacts.

    Each source is checked only if its file actually exists.  This keeps
    the test hermetic if a contributor runs only a subset of the eval
    scripts."""

    def test_extractors_do_not_raise(self) -> None:
        tracker = EvalDriftTracker()
        for src in METRIC_SOURCES:
            if not src.exists():
                continue
            data = tracker._read_json(src.path)
            extractor = getattr(tracker, src.extractor)
            out = extractor(data)
            self.assertIsInstance(out, dict)


if __name__ == "__main__":
    unittest.main()
