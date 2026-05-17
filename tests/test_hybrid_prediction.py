"""Tests for the hybrid prediction layer (classification + regression + toxicity).

These guard the contract that completes the user's stated hybrid spec:
  - Each head emits its own evidence envelope.
  - Heads can abstain independently — toxicity has different sufficiency
    rules than response classification, so the same row can produce a
    scoring toxicity decision while the response heads abstain.
  - The regression head returns ``response_score=None`` on abstention.
  - The hybrid bundle's `to_dict` is JSON-safe.
  - Live integration writes one trace row per head when an artifact is present.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import Patient, PredictionTrace
from backend.services.hybrid_prediction import (
    DEFAULT_REGRESSION_MODEL_PATH,
    DEFAULT_TOXICITY_MODEL_PATH,
    HybridPrediction,
    HIGH_TOXICITY_THRESHOLD,
    LOW_TOXICITY_THRESHOLD,
    STRONG_RESPONSE_THRESHOLD,
    WEAK_RESPONSE_THRESHOLD,
    predict_hybrid,
    predict_response_score_with_abstention,
    predict_toxicity_with_abstention,
)
from backend.services.predict_with_abstention import DEFAULT_MODEL_PATH


REQUIRED_ARTIFACTS = (
    Path(DEFAULT_MODEL_PATH),
    Path(DEFAULT_REGRESSION_MODEL_PATH),
    Path(DEFAULT_TOXICITY_MODEL_PATH),
)


def _artifacts_present() -> bool:
    return all(p.exists() for p in REQUIRED_ARTIFACTS)


def _full_row() -> dict:
    return {
        "age": 55, "cycle": 2, "stage": "II",
        "molecular_subtype": "HR_positive", "regimen": "AC_T",
        "pre_wbc": 6.0, "pre_anc": 3.0, "pre_hemoglobin": 12.0, "pre_platelets": 200.0,
        "nadir_wbc": 2.0, "nadir_anc": 0.8, "nadir_hemoglobin": 10.0, "nadir_platelets": 90.0,
        "recovery_wbc": 5.0, "recovery_hemoglobin": 11.5, "recovery_platelets": 180.0,
        "mri_tumor_size_cm": 3.5, "mri_percent_change_from_baseline": -10.0,
        "max_symptom_severity": 3, "symptom_count": 2,
        "intervention_count": 1, "dose_delayed": 0, "dose_reduced": 0,
    }


# ─── Per-head abstention contract ────────────────────────────────────────────


class RegressionHeadContract(unittest.TestCase):
    def test_demographics_only_abstains_with_null_score(self) -> None:
        out = predict_response_score_with_abstention(
            {"age": 55, "cycle": 1, "stage": "II",
             "molecular_subtype": "HR_positive", "regimen": "AC_T"},
        )
        self.assertEqual(out.decision, "insufficient_evidence")
        self.assertIsNone(out.response_score)
        self.assertIsNone(out.raw_response_score)
        self.assertIsNone(out.uncertainty_band)
        self.assertEqual(out.question, "response_score_regression")

    def test_full_row_returns_score_with_uncertainty_band(self) -> None:
        if not _artifacts_present():
            self.skipTest("Required model artifacts not present.")
        out = predict_response_score_with_abstention(_full_row())
        self.assertNotEqual(out.decision, "insufficient_evidence")
        self.assertIsNotNone(out.response_score)
        self.assertGreaterEqual(out.response_score, 0.0)
        self.assertLessEqual(out.response_score, 1.0)
        self.assertIsNotNone(out.uncertainty_band)
        # Band must contain the score.
        self.assertLessEqual(out.uncertainty_band[0], out.response_score)
        self.assertGreaterEqual(out.uncertainty_band[1], out.response_score)
        self.assertIn("evidence_adjusted", out.uncertainty_method)

    def test_decision_thresholds_align_with_score(self) -> None:
        if not _artifacts_present():
            self.skipTest("Required model artifacts not present.")
        out = predict_response_score_with_abstention(_full_row())
        if out.response_score is None:
            self.skipTest("Score abstained on real artifact unexpectedly.")
        if out.response_score >= STRONG_RESPONSE_THRESHOLD:
            self.assertEqual(out.decision, "strong_response_signal")
        elif out.response_score <= WEAK_RESPONSE_THRESHOLD:
            self.assertEqual(out.decision, "weak_response_signal")
        else:
            self.assertEqual(out.decision, "moderate_response_signal")


class ToxicityHeadContract(unittest.TestCase):
    def test_toxicity_uses_different_sufficiency_rules_than_response(self) -> None:
        if not _artifacts_present():
            self.skipTest("Required model artifacts not present.")
        # Row with CBC + symptoms but NO imaging.  Response classification
        # would abstain; toxicity should still score.
        row = {
            "age": 55, "cycle": 2, "stage": "II",
            "molecular_subtype": "HR_positive", "regimen": "AC_T",
            "pre_wbc": 6.0, "pre_anc": 3.0,
            "max_symptom_severity": 4, "symptom_count": 2,
        }
        out = predict_toxicity_with_abstention(row)
        self.assertNotEqual(out.decision, "insufficient_evidence")
        self.assertIsNotNone(out.probability)

    def test_no_evidence_abstains(self) -> None:
        out = predict_toxicity_with_abstention(
            {"age": 55, "cycle": 1, "stage": "II",
             "molecular_subtype": "HR_positive", "regimen": "AC_T"},
        )
        self.assertEqual(out.decision, "insufficient_evidence")
        self.assertIsNone(out.probability)

    def test_decision_labels_align_with_thresholds(self) -> None:
        if not _artifacts_present():
            self.skipTest("Required model artifacts not present.")
        out = predict_toxicity_with_abstention(_full_row())
        if out.probability is None:
            self.skipTest("Toxicity abstained unexpectedly.")
        if out.probability >= HIGH_TOXICITY_THRESHOLD:
            self.assertEqual(out.decision, "high_toxicity_signal")
        elif out.probability <= LOW_TOXICITY_THRESHOLD:
            self.assertEqual(out.decision, "low_toxicity_signal")
        else:
            self.assertEqual(out.decision, "moderate_toxicity_signal")


# ─── Hybrid bundle ───────────────────────────────────────────────────────────


class HybridBundleContract(unittest.TestCase):
    def test_bundle_to_dict_has_three_named_heads(self) -> None:
        if not _artifacts_present():
            self.skipTest("Required model artifacts not present.")
        out = predict_hybrid(_full_row())
        self.assertIsInstance(out, HybridPrediction)
        d = out.to_dict()
        self.assertIn("classification", d)
        self.assertIn("response_score", d)
        self.assertIn("toxicity", d)
        self.assertIn("claim_boundary", d)
        # Each head has its own evidence envelope.
        for head in ("classification", "response_score", "toxicity"):
            self.assertIn("evidence", d[head])
            self.assertIn("model_version", d[head])
            self.assertIn("question", d[head])
        self.assertIn("uncertainty_method", d["response_score"])

    def test_partial_evidence_can_abstain_some_heads_not_others(self) -> None:
        if not _artifacts_present():
            self.skipTest("Required model artifacts not present.")
        # CBC + symptoms only — response heads should abstain, toxicity should score.
        row = {
            "age": 55, "cycle": 2, "stage": "II",
            "molecular_subtype": "HR_positive", "regimen": "AC_T",
            "pre_wbc": 6.0, "pre_anc": 3.0,
            "max_symptom_severity": 4, "symptom_count": 2,
        }
        out = predict_hybrid(row)
        self.assertEqual(out.classification.decision, "insufficient_evidence")
        self.assertEqual(out.response_score.decision, "insufficient_evidence")
        self.assertIsNone(out.response_score.response_score)
        # Toxicity should be willing to score on CBC + symptoms.
        self.assertNotEqual(out.toxicity.decision, "insufficient_evidence")


# ─── Live integration: hybrid → patient report → traces ─────────────────────


def _fresh_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _write_tiny_timeline(directory: Path, patient_id: str) -> Path:
    rows = []
    for cycle in (1, 2):
        rows.append({
            "patient_id": patient_id,
            "cycle": cycle,
            "treatment_date": f"2026-0{cycle}-15",
            "age": 50, "stage": "II",
            "molecular_subtype": "HR_positive", "regimen": "AC_T",
            "pre_wbc": 6.0, "pre_anc": 3.0,
            "pre_hemoglobin": 12.0, "pre_platelets": 200.0,
            "nadir_wbc": 2.0, "nadir_anc": 0.8,
            "nadir_hemoglobin": 10.0, "nadir_platelets": 90.0,
            "recovery_wbc": 5.0, "recovery_hemoglobin": 11.5, "recovery_platelets": 180.0,
            "mri_tumor_size_cm": 3.5, "mri_percent_change_from_baseline": -15.0,
            "max_symptom_severity": 3, "symptom_count": 2,
            "intervention_count": 1, "dose_delayed": 0, "dose_reduced": 0,
            "treatment_success_binary": 1,
        })
    csv = directory / "rows.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    # Reset the row-loader cache so the test sees the fresh CSV.
    from backend.services.live_evidence_prediction import _load_timeline_index
    _load_timeline_index.cache_clear()
    return csv


class LiveHybridReport(unittest.TestCase):
    def test_hybrid_build_writes_three_traces_with_matching_snapshot(self) -> None:
        if not _artifacts_present():
            self.skipTest("Required model artifacts not present.")
        from backend.services.live_evidence_prediction import build_hybrid_prediction

        db = _fresh_db()
        db.add(Patient(id="P-HYB", name="Hybrid"))
        db.commit()
        with TemporaryDirectory() as tmp:
            csv = _write_tiny_timeline(Path(tmp), "P-HYB")
            bundle = build_hybrid_prediction(
                "P-HYB", db, timeline_csv=str(csv), actor_role="patient",
            )
        self.assertIsNotNone(bundle)
        # Three traces (one per head) all sharing the same timeline_snapshot_hash.
        rows = db.query(PredictionTrace).filter(PredictionTrace.patient_id == "P-HYB").all()
        self.assertEqual(len(rows), 3)
        snapshot_hashes = {r.timeline_snapshot_hash for r in rows}
        self.assertEqual(len(snapshot_hashes), 1, "snapshot hash should match across heads")
        questions = {r.question for r in rows}
        self.assertEqual(questions, {
            "response_classification",
            "response_score_regression",
            "toxicity_classification",
        })


if __name__ == "__main__":
    unittest.main()
