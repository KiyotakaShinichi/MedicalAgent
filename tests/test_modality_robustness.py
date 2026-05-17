"""Tests for the modality-dropout training + champion-vs-robust comparison.

Pure-function tests over the augmentation logic don't need any trained
artifacts; end-to-end tests skip when the model files aren't present so
fresh checkouts still pass.
"""
from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from backend.services.evidence_sufficiency import MODALITY_GROUPS
from backend.services.modality_dropout_training import (
    DEFAULT_MODEL_PATH,
    _augment_with_modality_dropout,
    _draw_dropouts,
    load_modality_robust_training_metadata,
    train_modality_robust_classifier,
)


def _tiny_training_frame(n_patients: int = 40, cycles: int = 3) -> pd.DataFrame:
    """Minimal valid training frame in the schema the trainer expects.

    Labels are driven by a strong synthetic relationship between imaging
    response (`mri_percent_change_from_baseline`) and `treatment_success_binary`
    so the trained classifier can actually learn something — without that
    signal, AUC on the unaugmented test split collapses to chance and the
    contract test would be meaningless.
    """
    rows = []
    for pid in range(n_patients):
        # Each patient has a stable "response strength" between 0 and 1 that
        # drives both their imaging trend and their final label.
        response_strength = (pid * 13 % 100) / 100.0
        success = 1 if response_strength > 0.5 else 0
        for c in range(1, cycles + 1):
            mri_change = -30 * response_strength * c + (pid % 5) - 2
            rows.append({
                "patient_id": f"P{pid:03d}",
                "cycle": c,
                "treatment_date": f"2026-0{c}-01",
                "age": 40 + (pid % 30),
                "stage": "II",
                "molecular_subtype": "HR_positive",
                "regimen": "AC_T",
                "pre_wbc": 6.0 + 0.1 * c,
                "pre_anc": 3.0,
                "pre_hemoglobin": 12.0,
                "pre_platelets": 200.0,
                "nadir_wbc": 2.5 - response_strength,
                "nadir_anc": 0.8,
                "nadir_hemoglobin": 10.5 - 0.5 * (1 - response_strength),
                "nadir_platelets": 100 - 30 * (1 - response_strength),
                "recovery_wbc": 5.0,
                "recovery_hemoglobin": 11.5,
                "recovery_platelets": 180.0,
                "mri_tumor_size_cm": 3.5 - 2 * response_strength * c / cycles,
                "mri_percent_change_from_baseline": mri_change,
                "max_symptom_severity": 3 + (c % 3),
                "symptom_count": 2,
                "intervention_count": 1,
                "dose_delayed": 0,
                "dose_reduced": 0,
                "treatment_success_binary": success,
                "toxicity_risk_binary": (pid + c) % 2,
                "urgent_intervention_needed": 0,
                "support_intervention_needed": 0,
                "latent_response_strength": 0.5 + 0.01 * pid,
                "response_score_percent": 10.0 * c,
                "final_response_category": "partial_response",
                "final_response_multiclass": "PR",
                "final_cancer_status": "responding",
                "maintenance_needed": 0,
                "cycle_response_trend_class": "improving",
            })
    return pd.DataFrame(rows)


# ─── Augmentation primitives ─────────────────────────────────────────────────


class DrawDropoutsRule(unittest.TestCase):
    """`_draw_dropouts` must respect the cap and only pick droppable modalities."""

    def test_never_exceeds_max_simultaneous(self) -> None:
        rng = random.Random(0)
        droppable = ["a", "b", "c", "d", "e"]
        draws = _draw_dropouts(n=200, droppable=droppable, p=0.9, max_simultaneous=3, rng=rng)
        self.assertEqual(len(draws), 200)
        self.assertTrue(all(len(d) <= 3 for d in draws), "max_simultaneous violated")
        for d in draws:
            self.assertTrue(set(d).issubset(droppable))

    def test_p_zero_produces_no_dropouts(self) -> None:
        rng = random.Random(0)
        draws = _draw_dropouts(n=50, droppable=["a", "b"], p=0.0, max_simultaneous=2, rng=rng)
        self.assertTrue(all(d == [] for d in draws))


class AugmentationContract(unittest.TestCase):
    """`_augment_with_modality_dropout` must keep the original rows intact,
    add the right number of augmented copies, and only mask droppable
    modality groups."""

    def test_originals_are_preserved_and_count_is_correct(self) -> None:
        frame = _tiny_training_frame()
        rng = random.Random(42)
        out, stats = _augment_with_modality_dropout(
            frame,
            rng=rng,
            n_aug_per_row=3,
            p_drop_per_modality=0.5,
            max_simultaneous_dropouts=3,
            protected_modalities=("demographics",),
        )
        # 1 original copy + 2 augmented copies = 3x input length.
        self.assertEqual(len(out), 3 * len(frame))
        # The first chunk in the output must match the originals row-for-row.
        np.testing.assert_array_equal(
            out.iloc[:len(frame)].values, frame.values,
        )
        # Stats must add up.
        self.assertEqual(stats["input_rows"], len(frame))
        self.assertEqual(stats["augmented_rows_added"], 2 * len(frame))

    def test_demographics_is_never_dropped(self) -> None:
        frame = _tiny_training_frame()
        rng = random.Random(99)
        out, stats = _augment_with_modality_dropout(
            frame,
            rng=rng,
            n_aug_per_row=4,
            p_drop_per_modality=0.95,  # aggressive — drop everything possible
            max_simultaneous_dropouts=7,
            protected_modalities=("demographics",),
        )
        # Demographics columns should never be NaN/empty in the augmented frame.
        for column in MODALITY_GROUPS["demographics"]:
            self.assertFalse(
                out[column].isna().any(),
                f"demographics column {column} was masked (should be protected)",
            )
        self.assertNotIn("demographics", stats["droppable_modalities"])

    def test_at_least_one_modality_is_actually_dropped_at_reasonable_p(self) -> None:
        """Sanity: with p=0.5 and 5 droppable groups, *some* dropouts must fire."""
        frame = _tiny_training_frame()
        rng = random.Random(7)
        _, stats = _augment_with_modality_dropout(
            frame,
            rng=rng,
            n_aug_per_row=3,
            p_drop_per_modality=0.5,
            max_simultaneous_dropouts=3,
            protected_modalities=("demographics",),
        )
        self.assertGreater(stats["total_dropouts_applied"], 0)
        self.assertGreater(stats["mean_dropouts_per_augmented_row"], 0)


# ─── End-to-end training contract ────────────────────────────────────────────


class TrainingContract(unittest.TestCase):
    """`train_modality_robust_classifier` must produce a model + metadata
    artifact and report sane test metrics on the unaugmented test split."""

    def test_full_training_writes_artifacts_and_meets_floor(self) -> None:
        frame = _tiny_training_frame(n_patients=60, cycles=3)
        with TemporaryDirectory() as tmp:
            csv = Path(tmp) / "rows.csv"
            model_path = Path(tmp) / "robust.joblib"
            meta_path = Path(tmp) / "meta.json"
            frame.to_csv(csv, index=False)

            metadata = train_modality_robust_classifier(
                ml_csv_path=str(csv),
                model_output_path=str(model_path),
                metadata_output_path=str(meta_path),
                n_aug_per_row=2,  # keep test fast
                p_drop_per_modality=0.3,
                seed=0,
            )

            self.assertEqual(metadata["status"], "passed")
            self.assertTrue(model_path.exists(), "Model artifact must be persisted.")
            self.assertTrue(meta_path.exists(), "Metadata artifact must be persisted.")
            self.assertTrue(metadata["patient_split"]["split_disjoint"])
            # Even on this tiny synthetic frame the labels are perfectly
            # separable by `age % 2`, so test AUC should be near 1.0.
            self.assertGreater(metadata["test_metrics"]["roc_auc"], 0.7)


# ─── load_* helper ───────────────────────────────────────────────────────────


class LoadModalityRobustMetadata(unittest.TestCase):
    def test_returns_missing_shell_when_file_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            meta = load_modality_robust_training_metadata(
                path=str(Path(tmp) / "does_not_exist.json"),
            )
            self.assertEqual(meta["status"], "missing")
            self.assertIn("message", meta)

    def test_round_trips_a_real_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps({
                "schema_version": "modality_robust_training_v1",
                "status": "passed",
                "test_metrics": {"roc_auc": 0.91},
            }))
            loaded = load_modality_robust_training_metadata(path=str(path))
            self.assertEqual(loaded["status"], "passed")
            self.assertEqual(loaded["test_metrics"]["roc_auc"], 0.91)


# ─── Production-data comparison gate ─────────────────────────────────────────


class ProductionComparisonGate(unittest.TestCase):
    """If both trained models exist on disk, the comparison must produce a
    non-regressing verdict.  This is the hard CI gate for the modality-
    robustness work."""

    def test_comparison_status_is_at_least_acceptable(self) -> None:
        champion = Path("Data/complete_synthetic_training/gradient_boosting_treatment_success_binary.joblib")
        robust = Path(DEFAULT_MODEL_PATH)
        if not (champion.exists() and robust.exists()):
            self.skipTest("Required model artifacts not present in this checkout.")
        from backend.services.modality_robustness_comparison import (
            run_modality_robustness_comparison,
        )
        with TemporaryDirectory() as tmp:
            payload = run_modality_robustness_comparison(
                output_path=str(Path(tmp) / "cmp.json"),
                sample_size=300,
            )
            self.assertIn(payload["status"], {"robust", "acceptable"})
            self.assertEqual(payload["summary"]["scenario_count"], 8)
            # Robust variant must not regress on full_data accuracy by more
            # than 1 percentage point.
            full_delta = payload["summary"]["full_data_accuracy_delta"]
            self.assertGreater(full_delta, -0.01, f"Full-data regression: {full_delta}")


if __name__ == "__main__":
    unittest.main()
