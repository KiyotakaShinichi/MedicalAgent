"""Tests for the regression-side robustness work (Phase 10b + 10c).

Contract guarded:
  - Modality-dropout regression trainer writes a valid model + metadata.
  - The legacy vs. modality-robust comparison artifact:
      * lands on disk with the expected schema,
      * reports a per-scenario MAE delta for every scenario,
      * `no_imaging` shows the robust variant beating the legacy regressor
        (the regression analog of the +8.3pp classifier improvement),
      * full_data MAE does not regress catastrophically (>5 on the 0-100
        scale would be a training collapse).
  - The status summary aligns with the per-scenario deltas.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backend.services.modality_dropout_regression_training import (
    DEFAULT_MODEL_PATH as DEFAULT_ROBUST_REGRESSION_PATH,
    load_modality_robust_regression_metadata,
    train_modality_robust_regressor,
)
from backend.services.regression_robustness_comparison import (
    REGRESSORS_UNDER_TEST,
    load_regression_robustness_comparison,
    run_regression_robustness_comparison,
)


LEGACY_REGRESSOR_PATH = REGRESSORS_UNDER_TEST["legacy"]


def _both_regressor_artifacts_present() -> bool:
    return Path(LEGACY_REGRESSOR_PATH).exists() and Path(DEFAULT_ROBUST_REGRESSION_PATH).exists()


def _feature_signal_frame(n_patients: int = 80, cycles: int = 3) -> pd.DataFrame:
    """Mini training frame with real signal — mirrors the quantile-test
    fixture so a tiny trainer run produces a learnable target."""
    rows = []
    for pid in range(n_patients):
        strength = (pid * 13 % 100) / 100.0
        success = 1 if strength > 0.5 else 0
        for c in range(1, cycles + 1):
            mri_change = -30 * strength * c + (pid % 5) - 2
            rows.append({
                "patient_id": f"P{pid:03d}",
                "cycle": c,
                "treatment_date": f"2026-0{c}-01",
                "age": 40 + (pid % 30),
                "stage": "II",
                "molecular_subtype": "HR_positive",
                "regimen": "AC_T",
                "pre_wbc": 6.0, "pre_anc": 3.0,
                "pre_hemoglobin": 12.0, "pre_platelets": 200.0,
                "nadir_wbc": 2.5 - strength, "nadir_anc": 0.8,
                "nadir_hemoglobin": 10.5 - 0.5 * (1 - strength),
                "nadir_platelets": 100 - 30 * (1 - strength),
                "recovery_wbc": 5.0, "recovery_hemoglobin": 11.5, "recovery_platelets": 180.0,
                "mri_tumor_size_cm": 3.5 - 2 * strength * c / cycles,
                "mri_percent_change_from_baseline": mri_change,
                "max_symptom_severity": 3 + (c % 3),
                "symptom_count": 2,
                "intervention_count": 1, "dose_delayed": 0, "dose_reduced": 0,
                "treatment_success_binary": success,
                "response_score_percent": 100 * strength + (c % 3),
                "toxicity_risk_binary": (pid + c) % 2,
                "urgent_intervention_needed": 0,
                "support_intervention_needed": 0,
                "latent_response_strength": strength,
                "final_response_category": "partial_response",
                "final_response_multiclass": "PR",
                "final_cancer_status": "responding",
                "maintenance_needed": 0,
                "cycle_response_trend_class": "improving",
            })
    return pd.DataFrame(rows)


# ─── Modality-dropout regression training contract ───────────────────────────


class ModalityRobustRegressionTraining(unittest.TestCase):
    def test_training_writes_model_plus_metadata_with_sensible_mae(self) -> None:
        frame = _feature_signal_frame(n_patients=80, cycles=3)
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv = tmp_path / "rows.csv"
            model_path = tmp_path / "robust.joblib"
            meta_path = tmp_path / "meta.json"
            frame.to_csv(csv, index=False)

            metadata = train_modality_robust_regressor(
                ml_csv_path=str(csv),
                model_output_path=str(model_path),
                metadata_output_path=str(meta_path),
                n_aug_per_row=2,
                p_drop_per_modality=0.30,
                seed=0,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(meta_path.exists())
            self.assertTrue(metadata["patient_split"]["split_disjoint"])
            # On this controlled fixture, MAE should be well under 15
            # (the floor between "acceptable" and "needs_attention").
            self.assertIn(metadata["status"], {"strong", "acceptable"})
            self.assertLess(metadata["test_metrics"]["mae"], 15)

    def test_loader_returns_missing_shell_when_artifact_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_modality_robust_regression_metadata(
                path=str(Path(tmp) / "absent.json"),
            )
            self.assertEqual(payload["status"], "missing")
            self.assertIn("message", payload)


# ─── Regression robustness comparison ────────────────────────────────────────


class RegressionRobustnessComparisonContract(unittest.TestCase):
    """Production-data gate: when both regressor artifacts are present, the
    comparison must produce a valid 8-scenario report with the no_imaging
    delta showing the robust variant winning."""

    def test_comparison_artifact_meets_per_scenario_contract(self) -> None:
        if not _both_regressor_artifacts_present():
            self.skipTest("Required regressor artifacts not present.")
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "cmp.json"
            payload = run_regression_robustness_comparison(
                output_path=str(out_path),
                sample_size=600,
            )
            self.assertEqual(payload["summary"]["scenario_count"], 8)
            self.assertIn(payload["status"], {"robust", "acceptable", "needs_attention"})
            self.assertTrue(out_path.exists())

            # Every scenario must carry a per-head MAE + a per-scenario delta.
            for s in payload["scenarios"]:
                for head in ("legacy", "robust"):
                    self.assertIn("mae", s["force_score"][head])
                    self.assertIsNotNone(s["force_score"][head]["mae"])
                self.assertIn(
                    "force_score_mae_robust_minus_legacy", s["deltas"],
                )

            # full_data must not be catastrophic (MAE > 5 on a synthetic
            # 0-100 target would be a training collapse).
            full = next(s for s in payload["scenarios"] if s["scenario"] == "full_data")
            self.assertLess(full["force_score"]["robust"]["mae"], 5)

    def test_no_imaging_robust_beats_legacy_on_production_data(self) -> None:
        """The headline result that justifies the dropout retraining."""
        if not _both_regressor_artifacts_present():
            self.skipTest("Required regressor artifacts not present.")
        payload = load_regression_robustness_comparison()
        if payload.get("status") == "missing":
            self.skipTest("No live comparison artifact in this checkout.")
        no_imaging = next(
            (s for s in payload["scenarios"] if s["scenario"] == "no_imaging"),
            None,
        )
        self.assertIsNotNone(no_imaging, "no_imaging scenario must be present")
        delta = no_imaging["deltas"]["force_score_mae_robust_minus_legacy"]
        # Negative delta = robust has lower MAE on no_imaging rows.  We
        # require at least a 1-point improvement on the 0-100 scale to
        # justify the training cost.
        self.assertLess(delta, -1.0, f"Robust regressor did not beat legacy on no_imaging: {delta}")


class RegressionComparisonLoaderShell(unittest.TestCase):
    def test_loader_returns_missing_shell_when_artifact_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_regression_robustness_comparison(
                path=str(Path(tmp) / "absent.json"),
            )
            self.assertEqual(payload["status"], "missing")
            self.assertEqual(payload["scenarios"], [])


class RegressionComparisonStatusLogic(unittest.TestCase):
    """The summary-status logic should mirror the classifier comparison:
    `needs_attention` only on full_data regression; `robust` when wins ≥
    losses; `acceptable` otherwise."""

    def _stub(self, full_delta: float, wins: int, losses: int, ties: int = 0) -> dict:
        from backend.services.regression_robustness_comparison import _summarise
        scenarios = []
        # Always include a full_data row carrying the supplied delta.
        scenarios.append({
            "scenario": "full_data",
            "deltas": {"force_score_mae_robust_minus_legacy": full_delta},
        })
        # Plus enough decoy scenarios to hit the requested wins/losses/ties.
        for i in range(wins):
            scenarios.append({
                "scenario": f"win_{i}",
                "deltas": {"force_score_mae_robust_minus_legacy": -2.0},
            })
        for i in range(losses):
            scenarios.append({
                "scenario": f"loss_{i}",
                "deltas": {"force_score_mae_robust_minus_legacy": 2.0},
            })
        for i in range(ties):
            scenarios.append({
                "scenario": f"tie_{i}",
                "deltas": {"force_score_mae_robust_minus_legacy": 0.0},
            })
        return _summarise(scenarios)

    def test_full_data_regression_marks_needs_attention(self) -> None:
        # full_data delta > 2.0 trips the regression guard.
        out = self._stub(full_delta=3.0, wins=5, losses=0)
        self.assertEqual(out["status"], "needs_attention")

    def test_wins_at_least_losses_is_robust(self) -> None:
        out = self._stub(full_delta=0.5, wins=3, losses=2)
        self.assertEqual(out["status"], "robust")

    def test_more_losses_than_wins_is_acceptable(self) -> None:
        out = self._stub(full_delta=0.5, wins=1, losses=4)
        self.assertEqual(out["status"], "acceptable")


if __name__ == "__main__":
    unittest.main()
