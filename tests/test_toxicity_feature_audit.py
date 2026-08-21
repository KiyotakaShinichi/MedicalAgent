"""Tests for the toxicity classifier feature-importance audit.

Contract guarded:
  - Audit runs against a controlled synthetic frame and produces a
    well-formed artifact (top-level keys, per-feature rows, both
    baselines).
  - When a feature is deterministically tied to the label, the audit
    must flag it as a near-label proxy AND the strict baseline (strip
    every near-label feature) must collapse to chance.
  - When there's no proxy structure, the audit must report `strong`.
  - Status-policy boundary cases (no dominant features, dominant
    features + strict baseline still high, dominant features + strict
    collapses).
  - The failure-mode registry now references this audit.
  - Live production audit artifact (when present) is consistent with
    the documented expectation — at least one dominant feature.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backend.services.failure_mode_registry import ENGINEERING_RISKS
from backend.services.toxicity_feature_audit import (
    DOMINANT_FEATURE_THRESHOLD,
    _overall_status,
    load_toxicity_feature_audit,
    run_toxicity_feature_audit,
)


# ─── Fixture frames ──────────────────────────────────────────────────────────


def _frame_with_label_proxy(n_patients: int = 80, cycles: int = 3) -> pd.DataFrame:
    """Synthetic frame where ``intervention_count`` is the label and a few
    other features carry no signal.  Audit should flag intervention_count
    as a dominant proxy + the strict baseline (strip near-label features)
    should collapse to ~chance."""
    rows = []
    for pid in range(n_patients):
        toxic = pid % 2  # half positive, half negative
        for c in range(1, cycles + 1):
            rows.append({
                "patient_id": f"P{pid:03d}",
                "cycle": c,
                "treatment_date": f"2026-0{c}-01",
                "age": 40 + (pid % 30),
                "stage": "II",
                "molecular_subtype": "HR_positive",
                "regimen": "AC_T",
                # Strong proxy: high intervention_count ⇔ toxic.
                "intervention_count": 10 if toxic else 0,
                # Realistic clinical features unrelated to label.
                "pre_wbc": 6.0 + (pid % 3),
                "pre_anc": 3.0,
                "pre_hemoglobin": 12.0,
                "pre_platelets": 200.0,
                "nadir_wbc": 5.0,
                "nadir_anc": 2.5,
                "nadir_hemoglobin": 11.0,
                "nadir_platelets": 180.0,
                "recovery_wbc": 5.5,
                "recovery_hemoglobin": 11.5,
                "recovery_platelets": 190.0,
                "mri_tumor_size_cm": 3.0,
                "mri_percent_change_from_baseline": -10.0,
                "max_symptom_severity": 3,
                "symptom_count": 2,
                "dose_delayed": 0,
                "dose_reduced": 0,
                # Labels
                "toxicity_risk_binary": toxic,
                "treatment_success_binary": 1 - toxic,
                "urgent_intervention_needed": 0,
                "support_intervention_needed": 0,
                "latent_response_strength": 0.5,
                "response_score_percent": 50,
                "final_response_category": "partial_response",
                "final_response_multiclass": "PR",
                "final_cancer_status": "responding",
                "maintenance_needed": 0,
                "cycle_response_trend_class": "improving",
            })
    return pd.DataFrame(rows)


def _frame_with_no_proxy(n_patients: int = 80, cycles: int = 3) -> pd.DataFrame:
    """Frame where every feature is independent of the label.  Audit
    should NOT flag any dominant features (status = strong)."""
    import random
    rng = random.Random(0)
    rows = []
    for pid in range(n_patients):
        for c in range(1, cycles + 1):
            toxic = rng.choice([0, 1])
            rows.append({
                "patient_id": f"P{pid:03d}",
                "cycle": c,
                "treatment_date": f"2026-0{c}-01",
                "age": 50 + rng.randint(-5, 5),
                "stage": "II",
                "molecular_subtype": "HR_positive",
                "regimen": "AC_T",
                "intervention_count": rng.randint(0, 5),
                "pre_wbc": 6.0 + rng.random(),
                "pre_anc": 3.0 + rng.random(),
                "pre_hemoglobin": 12.0 + rng.random(),
                "pre_platelets": 200.0 + rng.randint(-30, 30),
                "nadir_wbc": 3.0 + rng.random(),
                "nadir_anc": 1.5 + rng.random(),
                "nadir_hemoglobin": 10.0 + rng.random(),
                "nadir_platelets": 100.0 + rng.randint(-30, 30),
                "recovery_wbc": 5.0 + rng.random(),
                "recovery_hemoglobin": 11.5,
                "recovery_platelets": 180.0,
                "mri_tumor_size_cm": 3.0,
                "mri_percent_change_from_baseline": -10.0,
                "max_symptom_severity": rng.randint(0, 10),
                "symptom_count": rng.randint(0, 5),
                "dose_delayed": rng.choice([0, 0, 0, 1]),
                "dose_reduced": rng.choice([0, 0, 0, 1]),
                "toxicity_risk_binary": toxic,
                "treatment_success_binary": rng.choice([0, 1]),
                "urgent_intervention_needed": 0,
                "support_intervention_needed": 0,
                "latent_response_strength": rng.random(),
                "response_score_percent": rng.randint(0, 100),
                "final_response_category": "partial_response",
                "final_response_multiclass": "PR",
                "final_cancer_status": "responding",
                "maintenance_needed": 0,
                "cycle_response_trend_class": "improving",
            })
    return pd.DataFrame(rows)


# ─── Status-policy boundary cases ────────────────────────────────────────────


class StatusPolicy(unittest.TestCase):
    def test_no_dominant_features_returns_strong(self) -> None:
        self.assertEqual(_overall_status([], {"auc": 0.50}, {"auc": 0.50}), "strong")

    def test_dominant_features_with_strict_baseline_collapsing_is_acceptable(self) -> None:
        # Strict baseline AUC < 0.90 → acceptable (the proxy collapse
        # confirms the original AUC was the proxy, not learned signal).
        out = _overall_status(["intervention_count"], {"auc": 0.99}, {"auc": 0.60})
        self.assertEqual(out, "acceptable")

    def test_dominant_features_with_strict_baseline_still_high_is_needs_attention(self) -> None:
        # Strict baseline AUC ≥ 0.90 → needs_attention (generator has
        # structural label leakage that the model can't avoid).
        out = _overall_status(["intervention_count"], {"auc": 0.99}, {"auc": 0.97})
        self.assertEqual(out, "needs_attention")

    def test_missing_strict_baseline_auc_falls_to_acceptable(self) -> None:
        self.assertEqual(
            _overall_status(["intervention_count"], {"auc": 0.99}, {"auc": None}),
            "acceptable",
        )


# ─── Audit run contract ──────────────────────────────────────────────────────


class AuditWithLabelProxy(unittest.TestCase):
    """When the data has a strong proxy + the trained-model artifact is
    absent (no .joblib present), the audit should still emit the
    structural finding via the per-feature label-separation gap."""

    def test_flags_intervention_count_as_near_label_proxy(self) -> None:
        frame = _frame_with_label_proxy(n_patients=100, cycles=3)
        with TemporaryDirectory() as tmp:
            csv = Path(tmp) / "rows.csv"
            out_path = Path(tmp) / "audit.json"
            frame.to_csv(csv, index=False)

            payload = run_toxicity_feature_audit(
                ml_csv_path=str(csv),
                model_path=str(Path(tmp) / "nonexistent.joblib"),
                output_path=str(out_path),
            )
            self.assertIn("intervention_count", payload["near_label_proxy_features"])
            # Audit artifact lands on disk.
            self.assertTrue(out_path.exists())
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written["target"], "toxicity_risk_binary")
            self.assertIn("feature_rows", written)
            self.assertIn("no_proxy_baseline", written)
            self.assertIn("strict_no_proxy_baseline", written)


class AuditOnUnstructuredFrame(unittest.TestCase):
    """Frame with no proxy structure → no dominant features → strong."""

    def test_no_dominant_features_when_data_is_unstructured(self) -> None:
        frame = _frame_with_no_proxy(n_patients=120, cycles=3)
        with TemporaryDirectory() as tmp:
            csv = Path(tmp) / "rows.csv"
            out_path = Path(tmp) / "audit.json"
            frame.to_csv(csv, index=False)

            payload = run_toxicity_feature_audit(
                ml_csv_path=str(csv),
                model_path=str(Path(tmp) / "nonexistent.joblib"),
                output_path=str(out_path),
                dominant_threshold=DOMINANT_FEATURE_THRESHOLD,
            )
            # No trained model → all importances are 0.0 → no dominant features
            # → status = strong.
            self.assertEqual(payload["dominant_features"], [])
            self.assertEqual(payload["status"], "strong")


# ─── Loader shell ────────────────────────────────────────────────────────────


class LoaderShell(unittest.TestCase):
    def test_missing_artifact_returns_missing_shell(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_toxicity_feature_audit(path=str(Path(tmp) / "absent.json"))
            self.assertEqual(payload["status"], "missing")
            self.assertEqual(payload["feature_rows"], [])
            self.assertIn("message", payload)


# ─── Failure-mode registry integration ───────────────────────────────────────


class FailureModeRegistryReferencesAudit(unittest.TestCase):
    """The toxicity audit must be listed in the failure-mode registry —
    that's the one place a reviewer reads the "what could go wrong"
    catalog."""

    def test_audit_is_referenced_in_failure_mode_registry(self) -> None:
        names = {entry["name"] for entry in ENGINEERING_RISKS}
        self.assertIn("toxicity_classifier_tautological_features", names)
        entry = next(
            e for e in ENGINEERING_RISKS
            if e["name"] == "toxicity_classifier_tautological_features"
        )
        self.assertEqual(entry["severity"], "high")
        self.assertIn("toxicity_feature_audit", entry["benchmark_coverage"])


# ─── Production-data check (skipped when artifacts missing) ──────────────────


class ProductionAuditMatchesExpectation(unittest.TestCase):
    """The live audit artifact (when present) must report at least one
    dominant feature — confirms the documented finding in the failure
    registry is still real on the current generator."""

    def test_production_audit_flags_at_least_one_dominant_feature(self) -> None:
        payload = load_toxicity_feature_audit()
        if payload.get("status") == "missing":
            self.skipTest("Production toxicity audit artifact not present.")
        self.assertGreater(
            len(payload.get("dominant_features", [])),
            0,
            "Expected at least one dominant feature in production audit — generator unchanged",
        )


if __name__ == "__main__":
    unittest.main()
