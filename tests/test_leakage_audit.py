"""Unit tests for backend.services.leakage_audit.

These tests don't touch the FastAPI app or the database — the audit is a
pure function over a pandas frame.  They build small synthetic frames in
the same shape `temporal_ml_rows.csv` carries so the rules can be exercised
without depending on the live ~1.4k-row generator output.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backend.services.leakage_audit import (
    DEFAULT_CLASSIFICATION_TARGETS,
    KNOWN_LABEL_PROXIES,
    run_leakage_audit,
)
from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
)


# A synthetic feature contract sanity-check.  If a future edit removes a
# proxy from EXCLUDED_COLUMNS or accidentally adds it to NUMERIC_FEATURES,
# this test fails before the audit is even run.
class FeatureContractInvariants(unittest.TestCase):
    def test_no_label_proxy_is_in_the_active_feature_lists(self) -> None:
        feature_lists = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
        leaked = feature_lists & set(KNOWN_LABEL_PROXIES)
        self.assertFalse(
            leaked,
            f"Label proxies must never appear in NUMERIC/CATEGORICAL features: {leaked}",
        )

    def test_every_known_proxy_is_in_the_exclusion_set(self) -> None:
        missing = set(KNOWN_LABEL_PROXIES) - set(EXCLUDED_COLUMNS)
        self.assertFalse(
            missing,
            f"EXCLUDED_COLUMNS should cover every known proxy; missing: {missing}",
        )


def _make_clean_frame(n_patients: int = 40, cycles_per_patient: int = 3) -> pd.DataFrame:
    """Build a minimal valid training frame.  Every column the audit reads is
    present and free of leakage so the audit should return 'passed'."""
    rows = []
    for pid in range(n_patients):
        # Alternate the binary outcomes so stratified splitting has work to do.
        success = pid % 2
        for c in range(1, cycles_per_patient + 1):
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
                "nadir_wbc": 2.0,
                "nadir_anc": 0.8,
                "nadir_hemoglobin": 10.0,
                "nadir_platelets": 90.0,
                "recovery_wbc": 5.0,
                "recovery_hemoglobin": 11.5,
                "recovery_platelets": 180.0,
                "mri_tumor_size_cm": 3.5 - 0.2 * c,
                "mri_percent_change_from_baseline": -10.0 * c,
                "max_symptom_severity": 3 + (c % 3),
                "symptom_count": 2,
                "intervention_count": 1,
                "dose_delayed": 0,
                "dose_reduced": 0,
                # Targets — outside the feature lists, present so the audit can
                # exercise patient_split + label-identity checks.
                "treatment_success_binary": success,
                "toxicity_risk_binary": (pid + c) % 2,
                "urgent_intervention_needed": 0,
                "support_intervention_needed": 0,
                # Generator-internal label proxies — present in the dataset
                # but must never appear in features.
                "latent_response_strength": 0.5 + 0.01 * pid,
                "response_score_percent": 10.0 * c,
                "final_response_category": "partial_response",
                "final_response_multiclass": "PR",
                "final_cancer_status": "responding",
                "maintenance_needed": 0,
                "cycle_response_trend_class": "improving",
            })
    return pd.DataFrame(rows)


class CleanFrameAudit(unittest.TestCase):
    """Every check must pass for a well-formed synthetic frame."""

    def test_clean_frame_passes_audit_and_writes_artifact(self) -> None:
        frame = _make_clean_frame()
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "rows.csv"
            out_path = Path(tmp) / "out.json"
            frame.to_csv(csv_path, index=False)

            payload = run_leakage_audit(
                training_rows_path=str(csv_path),
                output_path=str(out_path),
                temporal_output_path=str(Path(tmp) / "temporal.json"),
                classification_targets=("treatment_success_binary",),
                split_seeds=(0, 7),
            )

            self.assertEqual(payload["status"], "passed", payload)
            self.assertEqual(payload["summary"]["checks_failed"], 0)
            self.assertGreater(payload["summary"]["checks_passed"], 0)
            self.assertTrue(out_path.exists(), "Audit must write the artifact to disk.")
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "passed")
            self.assertIn("known_label_proxies", written)


class DuplicatePatientCycleDetected(unittest.TestCase):
    """A duplicated (patient_id, cycle) row must fail the uniqueness check."""

    def test_duplicate_patient_cycle_fails(self) -> None:
        frame = _make_clean_frame()
        # Duplicate the first row exactly — same patient, same cycle.
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "rows.csv"
            out_path = Path(tmp) / "out.json"
            frame.to_csv(csv_path, index=False)

            payload = run_leakage_audit(
                training_rows_path=str(csv_path),
                output_path=str(out_path),
                temporal_output_path=str(Path(tmp) / "temporal.json"),
                classification_targets=("treatment_success_binary",),
                split_seeds=(0,),
            )

            self.assertEqual(payload["status"], "failed")
            failed_names = [
                item["name"] for item in payload["findings"] if item["status"] != "passed"
            ]
            self.assertIn("patient_cycle_pair_is_unique", failed_names)


class LabelIdentityDetected(unittest.TestCase):
    """A feature column byte-equal to the label must fail the identity check.

    We simulate this by hand-crafting an audit input where one feature is
    literally a copy of the target.  Since the public service reads from CSV
    and the feature list is fixed, we *temporarily* monkey-patch the column
    to demonstrate the rule works end-to-end."""

    def test_feature_equal_to_label_is_flagged(self) -> None:
        frame = _make_clean_frame()
        # Force a known feature to equal the label numerically.
        frame["age"] = frame["treatment_success_binary"]
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "rows.csv"
            out_path = Path(tmp) / "out.json"
            frame.to_csv(csv_path, index=False)

            payload = run_leakage_audit(
                training_rows_path=str(csv_path),
                output_path=str(out_path),
                temporal_output_path=str(Path(tmp) / "temporal.json"),
                classification_targets=("treatment_success_binary",),
                split_seeds=(0,),
            )

            self.assertEqual(payload["status"], "failed")
            label_finding = next(
                item for item in payload["findings"]
                if item["name"] == "label_identity_check::treatment_success_binary"
            )
            self.assertEqual(label_finding["status"], "failed")
            self.assertIn("age", label_finding["evidence"]["features_equal_to_target"])


class PatientSplitDisjoint(unittest.TestCase):
    """The configured `_patient_split` must produce disjoint train/test
    patient sets across multiple seeds."""

    def test_split_is_disjoint_for_every_seed(self) -> None:
        frame = _make_clean_frame()
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "rows.csv"
            out_path = Path(tmp) / "out.json"
            frame.to_csv(csv_path, index=False)

            payload = run_leakage_audit(
                training_rows_path=str(csv_path),
                output_path=str(out_path),
                temporal_output_path=str(Path(tmp) / "temporal.json"),
                classification_targets=("treatment_success_binary",),
                split_seeds=(0, 7, 42, 123),
            )

            split_findings = [
                item for item in payload["findings"]
                if item["name"].startswith("patient_split_disjoint")
            ]
            self.assertEqual(len(split_findings), 4)
            self.assertTrue(all(item["status"] == "passed" for item in split_findings))


class RealDatasetAuditRuns(unittest.TestCase):
    """End-to-end: the actual production CSV must pass the audit.  This is
    the real CI signal — if a future commit adds leakage, the suite fails."""

    def test_production_dataset_audit_passes(self) -> None:
        production_csv = Path("Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
        if not production_csv.exists():
            self.skipTest("Synthetic dataset not present in this checkout.")

        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.json"
            payload = run_leakage_audit(
                training_rows_path=str(production_csv),
                output_path=str(out_path),
                temporal_output_path=str(Path(tmp) / "temporal.json"),
                classification_targets=DEFAULT_CLASSIFICATION_TARGETS,
                split_seeds=(0, 42),
            )
            self.assertEqual(
                payload["status"],
                "passed",
                f"Production-data leakage audit FAILED. Failed checks: "
                f"{[item['name'] for item in payload['findings'] if item['status'] != 'passed']}",
            )


if __name__ == "__main__":
    unittest.main()
