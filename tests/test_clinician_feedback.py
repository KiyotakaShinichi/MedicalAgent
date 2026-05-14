"""Tests for the extended clinician feedback loop.

Verifies that the schema accepts the new decisions and reason categories,
that the aggregate summary surfaces reason_category_counts and
review_target_counts, and that invalid decisions are rejected.
"""

from __future__ import annotations

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import Patient
from backend.services.clinician_feedback import (
    VALID_REVIEW_DECISIONS,
    clinical_feedback_summary,
    create_clinical_summary_review,
    latest_clinical_summary_review,
)
from backend.services.review_constants import (
    REVIEW_DECISIONS,
    REVIEW_REASON_CATEGORIES,
    REVIEW_TARGETS,
)


def _make_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    db.add(Patient(id="P-TEST-001", name="Test Patient", diagnosis="Breast cancer - confirmed"))
    db.commit()
    return db


class ReviewConstantsContractTests(unittest.TestCase):
    def test_valid_decisions_match_review_constants(self):
        self.assertEqual(VALID_REVIEW_DECISIONS, set(REVIEW_DECISIONS))

    def test_new_decisions_are_supported(self):
        for decision in ("unsafe", "missing_evidence", "wrong_escalation"):
            self.assertIn(decision, VALID_REVIEW_DECISIONS)

    def test_review_targets_include_risk_flag_and_timeline_event(self):
        self.assertIn("risk_flag", REVIEW_TARGETS)
        self.assertIn("timeline_event", REVIEW_TARGETS)

    def test_reason_categories_include_safety_critical_options(self):
        for reason in (
            "missing_imaging_followup",
            "wrong_escalation_level",
            "diagnostic_overreach",
            "prompt_injection_concern",
        ):
            self.assertIn(reason, REVIEW_REASON_CATEGORIES)


class CreateReviewTests(unittest.TestCase):
    def test_each_canonical_decision_persists(self):
        db = _make_db()
        for decision in REVIEW_DECISIONS:
            review = create_clinical_summary_review(
                db=db,
                patient_id="P-TEST-001",
                reviewer_role="clinician",
                decision=decision,
                summary_snapshot={"headline": "synthetic demo"},
            )
            self.assertEqual(review["decision"], decision)

    def test_invalid_decision_raises(self):
        db = _make_db()
        with self.assertRaises(ValueError):
            create_clinical_summary_review(
                db=db,
                patient_id="P-TEST-001",
                reviewer_role="clinician",
                decision="marked_unsafe",  # legacy / wrong name
                summary_snapshot={},
            )

    def test_reason_category_round_trip(self):
        db = _make_db()
        review = create_clinical_summary_review(
            db=db,
            patient_id="P-TEST-001",
            reviewer_role="clinician",
            decision="rejected",
            summary_snapshot={"headline": "synthetic demo"},
            reason_category="diagnostic_overreach",
            review_target="summary",
            model_version="gradient_boosting@2026-05",
            rag_version="rag_v3",
        )
        self.assertEqual(review["reason_category"], "diagnostic_overreach")
        self.assertEqual(review["review_target"], "summary")
        self.assertEqual(review["model_version"], "gradient_boosting@2026-05")
        self.assertEqual(review["rag_version"], "rag_v3")

    def test_summary_surfaces_counts(self):
        db = _make_db()
        for decision in ("approved", "approved", "edited", "unsafe", "missing_evidence"):
            create_clinical_summary_review(
                db=db,
                patient_id="P-TEST-001",
                reviewer_role="clinician",
                decision=decision,
                summary_snapshot={"headline": "synthetic demo"},
                reason_category="evidence_quality" if decision != "approved" else None,
            )
        summary = clinical_feedback_summary(db)
        self.assertEqual(summary["review_count"], 5)
        self.assertEqual(summary["decision_counts"].get("approved"), 2)
        self.assertEqual(summary["decision_counts"].get("unsafe"), 1)
        self.assertEqual(summary["decision_counts"].get("missing_evidence"), 1)
        self.assertGreaterEqual(summary["reason_category_counts"].get("evidence_quality", 0), 3)
        self.assertGreaterEqual(summary["review_target_counts"].get("summary", 0), 5)

    def test_latest_review_returns_most_recent(self):
        db = _make_db()
        create_clinical_summary_review(
            db=db,
            patient_id="P-TEST-001",
            reviewer_role="clinician",
            decision="approved",
            summary_snapshot={},
        )
        last = create_clinical_summary_review(
            db=db,
            patient_id="P-TEST-001",
            reviewer_role="clinician",
            decision="unsafe",
            summary_snapshot={},
        )
        latest = latest_clinical_summary_review(db, "P-TEST-001")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], last["id"])
        self.assertEqual(latest["decision"], "unsafe")


if __name__ == "__main__":
    unittest.main()
