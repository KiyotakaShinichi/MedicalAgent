"""Tests for the synthetic data quality (generator-proxy) report.

Lock-ins:

* The disclaimer string contains the exact substring
  "NOT a measure of clinical realism" so it cannot be silently
  weakened.
* The report labels itself ``synthetic_generator_quality_proxy``.
* Every requested feature in ``LAB_RANGES`` that is present in the
  CSV appears in the ``features`` array.
* Correlations array contains exactly the configured pairs.
"""
from __future__ import annotations

import unittest

import pandas as pd

from backend.services.synthetic_data_quality import (
    DEFAULT_ROWS_PATH,
    EXPECTED_CORRELATIONS,
    LAB_RANGES,
    build_quality_report,
)


class ReportShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_quality_report(DEFAULT_ROWS_PATH)
        cls.cols = set(pd.read_csv(DEFAULT_ROWS_PATH, nrows=1).columns)

    def test_label_is_explicit_proxy(self) -> None:
        self.assertEqual(self.report["label"], "synthetic_generator_quality_proxy")

    def test_disclaimer_is_present_and_explicit(self) -> None:
        self.assertIn("NOT a measure of clinical realism", self.report["disclaimer"])

    def test_features_cover_present_columns(self) -> None:
        emitted = {f["feature"] for f in self.report["features"]}
        expected = {name for name in LAB_RANGES if name in self.cols}
        self.assertEqual(emitted, expected)

    def test_each_feature_has_required_keys(self) -> None:
        for f in self.report["features"]:
            for k in ("n_observed", "n_missing", "missing_rate", "min", "max",
                      "mean", "std", "n_out_of_range", "out_of_range_rate",
                      "range_low", "range_high"):
                self.assertIn(k, f, f"{f['feature']} missing {k}")

    def test_correlations_array_matches_config(self) -> None:
        configured = {(a, b) for a, b, _ in EXPECTED_CORRELATIONS}
        emitted = {(c["feature_a"], c["feature_b"]) for c in self.report["correlations"]}
        self.assertEqual(emitted, configured)

    def test_observed_pearson_is_numeric_or_none(self) -> None:
        for c in self.report["correlations"]:
            self.assertTrue(c["observed_pearson"] is None or isinstance(c["observed_pearson"], (int, float)))


if __name__ == "__main__":
    unittest.main()
