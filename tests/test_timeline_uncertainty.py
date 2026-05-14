"""Tests that the patient timeline forwards the uncertainty / AI metadata
the frontend AIGeneratedLabel depends on.

Background: the frontend renders confidence, uncertainty reason, and the
"Clinician review required" amber badge on AI-generated events.  Those
fields must be populated by the backend; otherwise the safe-language
labeling story is half-built. This test pins the contract.
"""

from __future__ import annotations

import unittest

import pandas as pd

from backend.processing.timeline import build_clinical_timeline


def _risk(severity: str = "urgent_review") -> dict:
    return {
        "type": "low_wbc",
        "category": "lab",
        "severity": severity,
        "message": "WBC reached 2.5 x10^3/uL, suggesting possible suppression.",
        "uncertainty": {
            "confidence_level": "high",
            "uncertainty_reason": "Threshold-based rule on structured lab values.",
            "missing_data_indicators": [],
            "clinician_review_required": True,
        },
        "evidence": {
            "metric": "wbc",
            "date": "2026-04-01",
            "value": 2.5,
            "unit": "x10^3/uL",
            "threshold_config_version": "v1.1.0",
        },
    }


class TimelineUncertaintyContractTests(unittest.TestCase):
    def test_risk_events_carry_uncertainty_block(self):
        events = build_clinical_timeline(
            labs=pd.DataFrame(),
            treatments=pd.DataFrame(),
            imaging_reports=pd.DataFrame(),
            symptoms=pd.DataFrame(),
            risks=[_risk()],
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertTrue(event["ai_generated"])
        self.assertEqual(event["type"], "ai_risk_flag")
        self.assertEqual(event["evidence_source"], "risk_engine")
        self.assertEqual(event["model_version"], "v1.1.0")
        uncertainty = event["uncertainty"]
        self.assertEqual(uncertainty["confidence_level"], "high")
        self.assertTrue(uncertainty["clinician_review_required"])
        self.assertIn("uncertainty_reason", uncertainty)
        self.assertIn("missing_data_indicators", uncertainty)

    def test_non_ai_events_carry_evidence_source_but_not_uncertainty(self):
        labs = pd.DataFrame(
            [{"date": "2026-04-02", "wbc": 4.1, "hemoglobin": 12.5, "platelets": 230}]
        )
        symptoms = pd.DataFrame(
            [
                {
                    "date": "2026-04-02",
                    "symptom": "fatigue",
                    "severity": 3,
                    "notes": "",
                }
            ]
        )
        events = build_clinical_timeline(
            labs=labs,
            treatments=pd.DataFrame(),
            imaging_reports=pd.DataFrame(),
            symptoms=symptoms,
            risks=[],
        )
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertFalse(event["ai_generated"])
            self.assertIn("evidence_source", event)
            self.assertNotIn("uncertainty", event)

    def test_risk_event_with_missing_uncertainty_block_does_not_crash(self):
        risk_with_no_uncertainty = {
            "type": "low_wbc",
            "severity": "watch",
            "message": "WBC trending down.",
            "evidence": {"date": "2026-04-03", "threshold_config_version": "v1.1.0"},
        }
        events = build_clinical_timeline(
            labs=pd.DataFrame(),
            treatments=pd.DataFrame(),
            imaging_reports=pd.DataFrame(),
            symptoms=pd.DataFrame(),
            risks=[risk_with_no_uncertainty],
        )
        self.assertEqual(len(events), 1)
        # uncertainty key may be None / missing — the frontend handles both safely.
        self.assertTrue(events[0]["ai_generated"])

    def test_chronological_ordering_preserved(self):
        labs = pd.DataFrame(
            [
                {"date": "2026-04-05", "wbc": 4.1, "hemoglobin": 12.5, "platelets": 230},
                {"date": "2026-04-01", "wbc": 3.9, "hemoglobin": 12.3, "platelets": 220},
            ]
        )
        events = build_clinical_timeline(
            labs=labs,
            treatments=pd.DataFrame(),
            imaging_reports=pd.DataFrame(),
            symptoms=pd.DataFrame(),
            risks=[_risk()],
        )
        dates = [event["date"] for event in events]
        self.assertEqual(dates, sorted(dates))


if __name__ == "__main__":
    unittest.main()
