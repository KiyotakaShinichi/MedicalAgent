"""Tests for ``patient_temporal_cv``.

Lock-ins:

* Patient-level temporal CV never produces overlapping patient_ids
  across train/test in a fold.
* The naive row-level KFold demonstrably *does* produce overlap (so
  the comparison in the JSON report is meaningful, not vacuous).
* Walk-forward ordering: each fold's training cohort's earliest
  treatment date is <= the test cohort's earliest treatment date.
* Report schema is stable (top-level keys + strategy keys).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from backend.services.patient_temporal_cv import (
    build_cv_comparison,
    patient_temporal_folds,
    run_naive_row_level_cv,
    run_patient_temporal_cv,
)


ML_CSV = Path("Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")


def _load_rows() -> pd.DataFrame:
    rows = pd.read_csv(ML_CSV)
    rows["treatment_date"] = pd.to_datetime(rows["treatment_date"])
    return rows


class PatientTemporalFolds(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _load_rows()

    def test_no_patient_overlap_in_any_fold(self) -> None:
        folds = patient_temporal_folds(self.rows, n_folds=5)
        self.assertGreaterEqual(len(folds), 4)
        for train_pids, test_pids in folds:
            overlap = set(train_pids) & set(test_pids)
            self.assertFalse(overlap, f"patient overlap: {sorted(overlap)[:5]}")

    def test_walk_forward_ordering_on_first_dates(self) -> None:
        first_dates = self.rows.groupby("patient_id")["treatment_date"].min()
        folds = patient_temporal_folds(self.rows, n_folds=5)
        for train_pids, test_pids in folds:
            train_max_first = first_dates.loc[list(train_pids)].max()
            test_min_first = first_dates.loc[list(test_pids)].min()
            self.assertLessEqual(train_max_first, test_min_first)


class StrategyReports(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _load_rows()

    def test_patient_level_temporal_cv_has_no_overlap(self) -> None:
        report = run_patient_temporal_cv(self.rows, n_folds=4)
        self.assertEqual(report.patient_overlap_pairs, 0)
        self.assertEqual(report.temporal_violations, 0)
        self.assertGreater(report.train_rows_censored_after_test_start, 0)
        # All folds should produce a finite AUC on this dataset.
        for fold in report.folds:
            self.assertIsNotNone(fold.roc_auc)
            self.assertGreater(fold.train_n_patients, 0)
            self.assertGreater(fold.test_n_patients, 0)
            self.assertLess(
                pd.to_datetime(fold.train_date_max),
                pd.to_datetime(fold.test_date_min),
            )

    def test_naive_row_level_kfold_demonstrates_overlap(self) -> None:
        report = run_naive_row_level_cv(self.rows, n_folds=4)
        # If this is ever zero, the comparison is vacuous and the report
        # loses its meaning — guard against silent dataset changes.
        self.assertGreater(report.patient_overlap_pairs, 0)


class ReportSchema(unittest.TestCase):
    def test_build_cv_comparison_schema(self) -> None:
        report = build_cv_comparison(n_folds=4)
        for key in [
            "schema_version",
            "generated_at",
            "source_csv",
            "target",
            "n_folds",
            "seed",
            "n_rows_total",
            "n_patients_total",
            "patient_level_temporal_cv",
            "naive_row_level_kfold",
            "headline",
        ]:
            self.assertIn(key, report, key)
        self.assertEqual(report["patient_level_temporal_cv"]["patient_overlap_pairs"], 0)
        self.assertEqual(report["patient_level_temporal_cv"]["temporal_violations"], 0)
        self.assertTrue(report["patient_level_temporal_cv"]["row_temporal_censoring_applied"])
        self.assertIn("auc_optimism_from_naive_cv", report["headline"])
        # Round-trips as JSON.
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
