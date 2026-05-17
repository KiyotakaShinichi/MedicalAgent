"""Unit tests for the evidence-sufficiency rules + predict_with_abstention.

These tests do not load the live trained model.  The sufficiency rules
are pure functions; the prediction wrapper's *abstention* path is also pure
(it never touches joblib when evidence is insufficient).  The covered-path
tests load the real artifact only if it's present, to keep CI fast on
fresh checkouts.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.services.evidence_sufficiency import (
    MODALITY_GROUPS,
    assess_evidence,
    assess_response_classification,
    assess_toxicity_classification,
    assess_urgent_intervention,
    detect_modalities,
)
from backend.services.predict_with_abstention import predict_with_abstention


# ─── Modality detection ──────────────────────────────────────────────────────


class ModalityDetectionTests(unittest.TestCase):
    """`detect_modalities` is the data-presence reader.  Every other rule
    depends on it being honest about which modality groups are populated."""

    def test_full_row_reports_every_group_present(self) -> None:
        row = {
            "age": 55, "cycle": 2, "stage": "II",
            "molecular_subtype": "HR_positive", "regimen": "AC_T",
            "pre_wbc": 6.0, "pre_anc": 3.0, "pre_hemoglobin": 12.0, "pre_platelets": 200.0,
            "nadir_wbc": 2.0, "nadir_anc": 0.8, "nadir_hemoglobin": 10.0, "nadir_platelets": 90.0,
            "recovery_wbc": 5.0, "recovery_hemoglobin": 11.5, "recovery_platelets": 180.0,
            "mri_tumor_size_cm": 3.5, "mri_percent_change_from_baseline": -10.0,
            "max_symptom_severity": 3, "symptom_count": 2,
            "intervention_count": 1, "dose_delayed": 0, "dose_reduced": 0,
        }
        present, missing = detect_modalities(row)
        self.assertEqual(set(present), set(MODALITY_GROUPS))
        self.assertEqual(missing, [])

    def test_nan_string_and_none_all_count_as_missing(self) -> None:
        row = {
            "age": 55, "cycle": 2, "stage": "II",
            "molecular_subtype": "HR_positive", "regimen": "AC_T",
            "pre_wbc": None, "pre_anc": "", "pre_hemoglobin": float("nan"), "pre_platelets": "n/a",
            "mri_tumor_size_cm": None, "mri_percent_change_from_baseline": None,
            "max_symptom_severity": 3, "symptom_count": 2,
        }
        present, missing = detect_modalities(row)
        self.assertIn("symptoms", present)
        self.assertIn("cbc_pre", missing)
        self.assertIn("imaging", missing)

    def test_demographics_below_threshold_falls_to_missing(self) -> None:
        # Demographics requires ≥2 fields; supplying only `age` should miss.
        row = {"age": 55}
        present, missing = detect_modalities(row)
        self.assertIn("demographics", missing)


# ─── Response-classification rules ───────────────────────────────────────────


class ResponseClassificationRules(unittest.TestCase):
    """Response classification is the highest-risk decision; it must require
    imaging OR longitudinal CBC."""

    def test_imaging_only_with_demographics_is_partial_not_insufficient(self) -> None:
        assessment = assess_response_classification(
            present=["demographics", "imaging"],
            missing=["cbc_pre", "cbc_nadir", "cbc_recovery", "symptoms", "interventions"],
        )
        self.assertEqual(assessment.sufficiency, "partial")
        self.assertFalse(assessment.abstain)
        self.assertLess(assessment.confidence_modifier, 1.0)

    def test_longitudinal_cbc_without_imaging_is_partial(self) -> None:
        assessment = assess_response_classification(
            present=["demographics", "cbc_pre", "cbc_nadir", "cbc_recovery"],
            missing=["imaging", "symptoms", "interventions"],
        )
        self.assertEqual(assessment.sufficiency, "partial")
        self.assertFalse(assessment.abstain)

    def test_no_response_signal_at_all_abstains(self) -> None:
        assessment = assess_response_classification(
            present=["demographics", "cbc_pre", "symptoms"],
            missing=["cbc_nadir", "cbc_recovery", "imaging", "interventions"],
        )
        self.assertEqual(assessment.sufficiency, "insufficient")
        self.assertTrue(assessment.abstain)
        self.assertEqual(
            assessment.reason,
            "no_response_signal_imaging_or_longitudinal_cbc_required",
        )

    def test_missing_demographics_always_abstains(self) -> None:
        assessment = assess_response_classification(
            present=["imaging", "cbc_pre", "cbc_nadir", "cbc_recovery", "symptoms"],
            missing=["demographics", "interventions"],
        )
        self.assertTrue(assessment.abstain)
        self.assertEqual(assessment.reason, "missing_minimum_context")

    def test_both_response_signals_present_is_sufficient(self) -> None:
        assessment = assess_response_classification(
            present=["demographics", "imaging", "cbc_pre", "cbc_nadir", "cbc_recovery"],
            missing=["symptoms", "interventions"],
        )
        self.assertEqual(assessment.sufficiency, "sufficient")
        self.assertEqual(assessment.confidence_modifier, 1.0)


# ─── Toxicity + urgent-intervention rules ────────────────────────────────────


class ToxicityAndUrgentRules(unittest.TestCase):
    def test_toxicity_cbc_alone_is_partial(self) -> None:
        out = assess_toxicity_classification(
            present=["demographics", "cbc_pre"], missing=["symptoms"],
        )
        self.assertEqual(out.sufficiency, "partial")

    def test_toxicity_no_signal_abstains(self) -> None:
        out = assess_toxicity_classification(
            present=["demographics"], missing=["cbc_pre", "symptoms"],
        )
        self.assertTrue(out.abstain)

    def test_urgent_requires_symptoms_or_nadir(self) -> None:
        out = assess_urgent_intervention(
            present=["demographics", "cbc_pre"], missing=["symptoms", "cbc_nadir"],
        )
        self.assertTrue(out.abstain)
        self.assertEqual(out.reason, "no_acute_signal_symptoms_or_nadir_cbc_required")

    def test_urgent_with_both_acute_signals_is_sufficient(self) -> None:
        out = assess_urgent_intervention(
            present=["demographics", "symptoms", "cbc_nadir"], missing=[],
        )
        self.assertEqual(out.sufficiency, "sufficient")
        self.assertEqual(out.confidence_modifier, 1.0)


# ─── assess_evidence dispatcher ──────────────────────────────────────────────


class AssessEvidenceDispatcher(unittest.TestCase):
    def test_unknown_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            assess_evidence({"age": 55, "cycle": 1}, question="not_a_real_question")

    def test_dispatches_to_correct_assessor(self) -> None:
        row = {
            "age": 55, "cycle": 1, "stage": "II",
            "molecular_subtype": "HR_positive", "regimen": "AC_T",
            "pre_wbc": 6.0, "pre_anc": 3.0,
            "max_symptom_severity": 4, "symptom_count": 2,
        }
        # No imaging or full CBC pattern → response classification abstains
        self.assertTrue(assess_evidence(row, question="response_classification").abstain)
        # But toxicity has CBC + symptoms → sufficient
        self.assertEqual(
            assess_evidence(row, question="toxicity_classification").sufficiency,
            "sufficient",
        )


# ─── predict_with_abstention ─────────────────────────────────────────────────


class PredictWithAbstentionAbstainPath(unittest.TestCase):
    """The abstain branch doesn't load the model artifact; we can exercise
    it without needing the .joblib file on disk."""

    def test_demographics_only_abstains_with_no_probability(self) -> None:
        row = {"age": 55, "cycle": 2, "stage": "II",
               "molecular_subtype": "HR_positive", "regimen": "AC_T"}
        pred = predict_with_abstention(row)
        self.assertEqual(pred.decision, "insufficient_evidence")
        self.assertIsNone(pred.probability)
        self.assertIsNone(pred.raw_probability)
        self.assertEqual(pred.confidence, "low")
        self.assertTrue(pred.evidence.abstain)

    def test_response_returns_to_dict_with_required_keys(self) -> None:
        row = {"age": 55, "cycle": 2, "stage": "II",
               "molecular_subtype": "HR_positive", "regimen": "AC_T"}
        d = predict_with_abstention(row).to_dict()
        for key in (
            "decision", "probability", "raw_probability", "calibrated",
            "confidence", "evidence", "model_version", "question", "claim_boundary",
        ):
            self.assertIn(key, d)


class PredictWithAbstentionCoveredPath(unittest.TestCase):
    """End-to-end smoke against the real artifact (skipped on fresh
    checkouts where the model hasn't been trained yet)."""

    def test_full_row_does_not_abstain_when_model_present(self) -> None:
        artifact = Path(
            "Data/complete_synthetic_training/gradient_boosting_treatment_success_binary.joblib",
        )
        if not artifact.exists():
            self.skipTest("Trained champion model not present in this checkout.")

        row = {
            "age": 55, "cycle": 2, "stage": "II",
            "molecular_subtype": "HR_positive", "regimen": "AC_T",
            "pre_wbc": 6.0, "pre_anc": 3.0, "pre_hemoglobin": 12.0, "pre_platelets": 200.0,
            "nadir_wbc": 2.0, "nadir_anc": 0.8, "nadir_hemoglobin": 10.0, "nadir_platelets": 90.0,
            "recovery_wbc": 5.0, "recovery_hemoglobin": 11.5, "recovery_platelets": 180.0,
            "mri_tumor_size_cm": 3.5, "mri_percent_change_from_baseline": -10.0,
            "max_symptom_severity": 3, "symptom_count": 2,
            "intervention_count": 1, "dose_delayed": 0, "dose_reduced": 0,
        }
        pred = predict_with_abstention(row)
        self.assertNotEqual(pred.decision, "insufficient_evidence")
        self.assertIsNotNone(pred.probability)
        self.assertEqual(pred.evidence.sufficiency, "sufficient")


# ─── End-to-end eval ─────────────────────────────────────────────────────────


class EndToEndEvalGate(unittest.TestCase):
    """Run the full abstention sweep on a small sample and confirm the
    contract: full_data must cover everything, demographics_only must
    abstain on everything, and the artifact must be written."""

    def test_eval_artifact_meets_contract(self) -> None:
        artifact = Path(
            "Data/complete_synthetic_training/gradient_boosting_treatment_success_binary.joblib",
        )
        if not artifact.exists():
            self.skipTest("Trained champion model not present in this checkout.")

        from backend.services.evidence_abstention_eval import run_evidence_abstention_eval

        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.json"
            payload = run_evidence_abstention_eval(
                output_path=str(out_path),
                sample_size=120,  # keep CI fast
            )
            self.assertEqual(payload["status"] in {"strong", "acceptable"}, True, payload["status"])
            self.assertEqual(payload["summary"]["full_data_coverage_rate"], 1.0)
            self.assertEqual(payload["summary"]["demographics_only_abstention_rate"], 1.0)
            # The artifact must actually land on disk so the dashboard can read it.
            self.assertTrue(out_path.exists())
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], payload["status"])


if __name__ == "__main__":
    unittest.main()
