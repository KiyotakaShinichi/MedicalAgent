"""Tests for the live evidence-aware prediction wired into the patient report.

The contract this guards:
  - Patients in the synthetic cohort get an `evidence_aware_prediction`
    envelope on their report.
  - A `PredictionTrace` row is written for every live build that produces
    an envelope.
  - Patients NOT in the cohort get `None` cleanly (no crash, no fake data).
  - The envelope carries the same provenance fields the trace records.
"""
from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import (
    BreastCancerProfile,
    ImagingReport,
    LabResult,
    Patient,
    PredictionTrace,
    SymptomReport,
    Treatment,
)
from backend.services.live_evidence_prediction import (
    _load_timeline_index,
    build_hybrid_prediction,
    build_evidence_aware_prediction,
)


REQUIRED_ARTIFACT = Path(
    "Data/complete_synthetic_training/gradient_boosting_treatment_success_binary.joblib",
)


def _fresh_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _write_tiny_timeline(directory: Path, patient_ids: list[str]) -> Path:
    """Build a synthetic temporal_ml_rows.csv with the columns the
    inference path actually reads.  All fields are present so the trained
    model has the columns its pipeline expects."""
    rows: list[dict[str, object]] = []
    for i, pid in enumerate(patient_ids):
        for cycle in (1, 2):
            rows.append({
                "patient_id": pid,
                "cycle": cycle,
                "treatment_date": f"2026-0{cycle}-15",
                "age": 50 + i,
                "stage": "II",
                "molecular_subtype": "HR_positive",
                "regimen": "AC_T",
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
    # The resolver caches via lru_cache keyed on the path — clear so each test
    # sees its own fresh file.
    _load_timeline_index.cache_clear()
    return csv


# ─── Cohort membership ───────────────────────────────────────────────────────


class CohortMembershipContract(unittest.TestCase):
    def test_patient_in_cohort_gets_envelope_with_required_keys(self) -> None:
        if not REQUIRED_ARTIFACT.exists():
            self.skipTest("Trained model artifact not present.")
        db = _fresh_db()
        db.add(Patient(id="P-IN", name="In Cohort"))
        db.commit()
        with TemporaryDirectory() as tmp:
            csv = _write_tiny_timeline(Path(tmp), ["P-IN"])
            envelope = build_evidence_aware_prediction(
                "P-IN", db, timeline_csv=str(csv), actor_role="patient",
            )
        self.assertIsNotNone(envelope)
        for key in (
            "decision", "probability", "raw_probability", "calibrated",
            "confidence", "evidence", "model_version", "question", "claim_boundary",
        ):
            self.assertIn(key, envelope)
        for key in ("modalities_present", "modalities_missing", "sufficiency", "abstain"):
            self.assertIn(key, envelope["evidence"])

    def test_patient_not_in_cohort_returns_none_without_writing_trace(self) -> None:
        db = _fresh_db()
        db.add(Patient(id="P-OUT", name="Not In Cohort"))
        db.commit()
        with TemporaryDirectory() as tmp:
            csv = _write_tiny_timeline(Path(tmp), ["P-OTHER"])  # not the target patient
            envelope = build_evidence_aware_prediction(
                "P-OUT", db, timeline_csv=str(csv),
            )
        self.assertIsNone(envelope)
        self.assertEqual(db.query(PredictionTrace).count(), 0)

    def test_missing_timeline_csv_returns_none(self) -> None:
        db = _fresh_db()
        envelope = build_evidence_aware_prediction(
            "P-X", db, timeline_csv="/nonexistent/path.csv",
        )
        self.assertIsNone(envelope)

    def test_demo_patient_record_can_use_live_adapter_when_not_in_csv(self) -> None:
        if not REQUIRED_ARTIFACT.exists():
            self.skipTest("Trained model artifact not present.")
        db = _fresh_db()
        db.add(Patient(id="P-DEMO", name="Demo Patient"))
        db.add(BreastCancerProfile(
            patient_id="P-DEMO",
            cancer_stage="Stage II",
            molecular_subtype="HR-positive / HER2-negative",
        ))
        db.add(Treatment(
            patient_id="P-DEMO",
            date=date(2026, 1, 5),
            cycle=1,
            drug="Dose-dense AC (doxorubicin/cyclophosphamide)",
        ))
        db.add(Treatment(
            patient_id="P-DEMO",
            date=date(2026, 3, 16),
            cycle=6,
            drug="Paclitaxel",
        ))
        db.add_all([
            LabResult(patient_id="P-DEMO", date=date(2026, 1, 4), wbc=6.5, hemoglobin=13.0, platelets=248, source="test"),
            LabResult(patient_id="P-DEMO", date=date(2026, 3, 25), wbc=3.4, hemoglobin=10.9, platelets=178, source="test"),
            LabResult(patient_id="P-DEMO", date=date(2026, 3, 29), wbc=4.7, hemoglobin=11.1, platelets=214, source="test"),
        ])
        db.add(SymptomReport(patient_id="P-DEMO", date=date(2026, 3, 24), symptom="fatigue", severity=5))
        db.add_all([
            ImagingReport(
                patient_id="P-DEMO",
                date=date(2026, 1, 3),
                modality="Breast MRI",
                report_type="Baseline",
                body_site="Breast",
                findings="Right breast mass measures 4.2 cm.",
                impression="Baseline MRI.",
            ),
            ImagingReport(
                patient_id="P-DEMO",
                date=date(2026, 3, 30),
                modality="Breast MRI",
                report_type="Follow-up",
                body_site="Breast",
                findings="Residual enhancement measures 1.8 cm.",
                impression="Interval decrease; clinician interpretation required.",
            ),
        ])
        db.commit()
        with TemporaryDirectory() as tmp:
            csv = _write_tiny_timeline(Path(tmp), ["P-OTHER"])
            bundle = build_hybrid_prediction(
                "P-DEMO",
                db,
                timeline_csv=str(csv),
                actor_role="patient",
                record_trace=False,
            )
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["inference_source"], "live_patient_record_adapter")
        self.assertEqual(bundle["classification"]["evidence"]["abstain"], False)
        self.assertIn("response_score", bundle)
        self.assertIn("toxicity", bundle)


# ─── Trace persistence ───────────────────────────────────────────────────────


class TracePersistenceContract(unittest.TestCase):
    def test_each_build_writes_exactly_one_trace_row(self) -> None:
        if not REQUIRED_ARTIFACT.exists():
            self.skipTest("Trained model artifact not present.")
        db = _fresh_db()
        db.add(Patient(id="P-LIVE", name="Live"))
        db.commit()
        with TemporaryDirectory() as tmp:
            csv = _write_tiny_timeline(Path(tmp), ["P-LIVE"])
            build_evidence_aware_prediction("P-LIVE", db, timeline_csv=str(csv), actor_role="patient")
            build_evidence_aware_prediction("P-LIVE", db, timeline_csv=str(csv), actor_role="patient")
        rows = db.query(PredictionTrace).all()
        self.assertEqual(len(rows), 2)
        # Provenance fields all populated; abstain decisions captured.
        for r in rows:
            self.assertEqual(r.patient_id, "P-LIVE")
            self.assertEqual(r.actor_role, "patient")
            self.assertIsNotNone(r.model_version)
            self.assertIsNotNone(r.timeline_snapshot_hash)

    def test_record_trace_false_does_not_persist(self) -> None:
        if not REQUIRED_ARTIFACT.exists():
            self.skipTest("Trained model artifact not present.")
        db = _fresh_db()
        db.add(Patient(id="P-NOLOG", name="No Log"))
        db.commit()
        with TemporaryDirectory() as tmp:
            csv = _write_tiny_timeline(Path(tmp), ["P-NOLOG"])
            envelope = build_evidence_aware_prediction(
                "P-NOLOG", db, timeline_csv=str(csv), record_trace=False,
            )
        self.assertIsNotNone(envelope)
        # With commit=False inside predict_and_trace, the row is in the session
        # but not committed.  Rolling back proves it never reached the DB.
        db.rollback()
        self.assertEqual(db.query(PredictionTrace).count(), 0)


if __name__ == "__main__":
    unittest.main()
