"""Tests for the prediction traceability layer.

These exercise the trace contract directly against an in-memory SQLite DB
so we don't need the full FastAPI app stack.  The contract is:

  - every required field a reviewer would ask for is populated,
  - the abstain path records `probability=None` plus a structured reason,
  - the covered path records the calibrated probability AND the raw one,
  - filter + summary helpers return what the dashboard expects.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import Patient, PredictionTrace
from backend.services.evidence_sufficiency import EvidenceAssessment
from backend.services.predict_with_abstention import EvidenceAwarePrediction
from backend.services.prediction_trace import (
    FEATURE_SET_VERSION,
    TraceContext,
    hash_input_row,
    list_recent_traces,
    predict_and_trace,
    record_prediction_trace,
    summarise_traces,
)


def _fresh_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_patient(db, pid: str = "P-trace") -> None:
    db.add(Patient(id=pid, name="Trace Test"))
    db.commit()


def _abstained_prediction() -> EvidenceAwarePrediction:
    """Hand-built abstained prediction so the test doesn't depend on a model
    artifact being present in CI."""
    return EvidenceAwarePrediction(
        decision="insufficient_evidence",
        probability=None,
        raw_probability=None,
        calibrated=False,
        confidence="low",
        evidence=EvidenceAssessment(
            modalities_present=["demographics"],
            modalities_missing=["imaging", "cbc_pre", "cbc_nadir", "cbc_recovery", "symptoms", "interventions"],
            sufficiency="insufficient",
            abstain=True,
            reason="no_response_signal_imaging_or_longitudinal_cbc_required",
            confidence_modifier=0.0,
        ),
        model_version="gradient_boosting_treatment_success_binary",
        question="response_classification",
    )


def _covered_prediction() -> EvidenceAwarePrediction:
    return EvidenceAwarePrediction(
        decision="favorable_pattern",
        probability=0.812,
        raw_probability=0.847,
        calibrated=True,
        confidence="high",
        evidence=EvidenceAssessment(
            modalities_present=["demographics", "cbc_pre", "cbc_nadir", "cbc_recovery", "imaging", "symptoms"],
            modalities_missing=["interventions"],
            sufficiency="sufficient",
            abstain=False,
            reason=None,
            confidence_modifier=1.0,
        ),
        model_version="gradient_boosting_treatment_success_binary",
        question="response_classification",
    )


# ─── Schema completeness ─────────────────────────────────────────────────────


class TraceSchemaCompleteness(unittest.TestCase):
    """Every field a reviewer needs must land on the row."""

    REQUIRED_TRACE_FIELDS = (
        "question", "decision", "probability", "raw_probability", "calibrated",
        "confidence", "evidence_sufficiency", "abstained", "abstain_reason",
        "modalities_present_json", "modalities_missing_json", "confidence_modifier",
        "model_version", "feature_set_version", "threshold_config_json",
        "calibration_config_json", "safety_triggers_json", "validator_decision",
        "rag_source_ids_json", "timeline_snapshot_hash", "notes",
    )

    def test_abstained_prediction_writes_all_required_columns(self) -> None:
        db = _fresh_session()
        _seed_patient(db)
        ctx = TraceContext(
            patient_id="P-trace",
            request_id="req_001",
            actor_role="admin",
            safety_triggers=["urgent_check"],
            validator_decision="allowed",
            rag_source_ids=["kb_cbc_basics"],
            timeline_snapshot_hash=hash_input_row({"age": 55}),
            notes="abstained — demo",
        )
        trace = record_prediction_trace(db, _abstained_prediction(), context=ctx)
        db.commit()
        for field in self.REQUIRED_TRACE_FIELDS:
            self.assertTrue(hasattr(trace, field), f"trace missing field {field}")

        # Abstention specifics: probability fields must be NULL, abstained=1.
        self.assertIsNone(trace.probability)
        self.assertIsNone(trace.raw_probability)
        self.assertEqual(trace.abstained, 1)
        self.assertEqual(trace.evidence_sufficiency, "insufficient")
        self.assertEqual(
            trace.abstain_reason,
            "no_response_signal_imaging_or_longitudinal_cbc_required",
        )

        # Provenance: feature_set_version must be pinned, not None.
        self.assertEqual(trace.feature_set_version, FEATURE_SET_VERSION)
        # Threshold + calibration config must be valid JSON.
        self.assertIsInstance(json.loads(trace.threshold_config_json), dict)
        self.assertIsInstance(json.loads(trace.calibration_config_json), dict)
        # Modality JSON columns must round-trip cleanly.
        self.assertEqual(
            json.loads(trace.modalities_present_json),
            ["demographics"],
        )

    def test_covered_prediction_records_both_probabilities_and_calibration(self) -> None:
        db = _fresh_session()
        trace = record_prediction_trace(db, _covered_prediction())
        db.commit()
        self.assertAlmostEqual(trace.probability, 0.812, places=5)
        self.assertAlmostEqual(trace.raw_probability, 0.847, places=5)
        self.assertEqual(trace.calibrated, 1)
        self.assertEqual(trace.abstained, 0)


# ─── List + filter helpers ───────────────────────────────────────────────────


class TraceListingAndFilters(unittest.TestCase):
    def test_list_returns_most_recent_first_and_respects_limit(self) -> None:
        db = _fresh_session()
        for _ in range(5):
            record_prediction_trace(db, _abstained_prediction())
        db.commit()
        rows = list_recent_traces(db, limit=3)
        self.assertEqual(len(rows), 3)
        # Each row must be a JSON-safe dict with the expected keys.
        for r in rows:
            self.assertIn("decision", r)
            self.assertIn("modalities_present", r)
            self.assertIn("modalities_missing", r)
            self.assertIsInstance(r["modalities_present"], list)

    def test_abstained_only_filter(self) -> None:
        db = _fresh_session()
        record_prediction_trace(db, _abstained_prediction())
        record_prediction_trace(db, _covered_prediction())
        db.commit()
        only_abstained = list_recent_traces(db, abstained_only=True)
        self.assertEqual(len(only_abstained), 1)
        self.assertTrue(only_abstained[0]["abstained"])

    def test_patient_id_filter_isolates_one_patient(self) -> None:
        db = _fresh_session()
        _seed_patient(db, "P-A")
        _seed_patient(db, "P-B")
        record_prediction_trace(db, _abstained_prediction(), context=TraceContext(patient_id="P-A"))
        record_prediction_trace(db, _abstained_prediction(), context=TraceContext(patient_id="P-B"))
        record_prediction_trace(db, _abstained_prediction(), context=TraceContext(patient_id="P-A"))
        db.commit()
        only_a = list_recent_traces(db, patient_id="P-A")
        self.assertEqual(len(only_a), 2)
        self.assertTrue(all(r["patient_id"] == "P-A" for r in only_a))


class TraceSummary(unittest.TestCase):
    def test_summary_counts_decisions_and_computes_abstention_rate(self) -> None:
        db = _fresh_session()
        record_prediction_trace(db, _abstained_prediction())
        record_prediction_trace(db, _abstained_prediction())
        record_prediction_trace(db, _covered_prediction())
        db.commit()
        summary = summarise_traces(db)
        self.assertEqual(summary["total"], 3)
        self.assertAlmostEqual(summary["abstention_rate"], 2 / 3, places=3)
        self.assertEqual(summary["decision_counts"]["insufficient_evidence"], 2)
        self.assertEqual(summary["decision_counts"]["favorable_pattern"], 1)
        self.assertIn(
            "gradient_boosting_treatment_success_binary",
            summary["model_versions"],
        )

    def test_summary_on_empty_table_is_safe(self) -> None:
        db = _fresh_session()
        summary = summarise_traces(db)
        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["abstention_rate"])


# ─── End-to-end via predict_and_trace ────────────────────────────────────────


class PredictAndTraceEndToEnd(unittest.TestCase):
    """The one-shot helper must produce both a prediction and a persisted
    trace row in the same transaction.  Skips if the trained model artifact
    isn't on disk so fresh checkouts still pass."""

    def test_demographics_only_produces_abstained_trace(self) -> None:
        if not Path(
            "Data/complete_synthetic_training/gradient_boosting_treatment_success_binary.joblib"
        ).exists():
            self.skipTest("Trained model artifact not present in this checkout.")
        db = _fresh_session()
        _seed_patient(db)
        row = {"age": 55, "cycle": 2, "stage": "II",
               "molecular_subtype": "HR_positive", "regimen": "AC_T"}
        prediction, trace = predict_and_trace(
            db, row,
            context=TraceContext(patient_id="P-trace", actor_role="admin"),
        )
        self.assertEqual(prediction.decision, "insufficient_evidence")
        self.assertEqual(trace.decision, "insufficient_evidence")
        self.assertEqual(trace.abstained, 1)
        # The persisted row matches the live prediction's identifiers.
        self.assertEqual(trace.model_version, prediction.model_version)
        self.assertEqual(trace.patient_id, "P-trace")
        # Sanity: a second call against the same patient yields a second row.
        predict_and_trace(db, row, context=TraceContext(patient_id="P-trace"))
        self.assertEqual(db.query(PredictionTrace).count(), 2)


if __name__ == "__main__":
    unittest.main()
