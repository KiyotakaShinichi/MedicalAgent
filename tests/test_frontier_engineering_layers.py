"""Tests for the six frontier engineering layers added in this pass:

1. iterative_rag_sufficiency (eval-only)
2. conflict_aware_rag_adjudicator (eval-only)
3. longitudinal_context_cards
4. uncertainty_response_policy_eval
5. external_red_team_readiness (anti-fabrication)
6. synthetic_timeline_drift_stress (review_only_boundary locked)

Every artifact carries ``clinical_validation: false`` and a
``claim_boundary`` that says ``not clinical validation``.  Several
artifacts have additional hard invariants (no live-LLM call, no
goldset mutation, review_only_boundary, anti-fabrication).
"""
from __future__ import annotations

import os
import unittest


REQUIRED_DISCLAIMER = "not clinical validation"


def _has_disclaimer(payload: dict) -> bool:
    cb = str(payload.get("claim_boundary") or "").lower()
    return REQUIRED_DISCLAIMER in cb and payload.get("clinical_validation") is False


# ─── 1. Iterative RAG sufficiency ────────────────────────────────────────


class IterativeRagSufficiency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ONCOTRACK_FAST_MODE"] = "1"
        from backend.services.iterative_rag_sufficiency import (
            OUTPUT_PATH, build_report,
        )
        cls.OUTPUT_PATH = OUTPUT_PATH
        cls.report = build_report()

    def test_artifact_present(self) -> None:
        self.assertTrue(self.OUTPUT_PATH.exists())

    def test_disclaimer_present(self) -> None:
        self.assertTrue(_has_disclaimer(self.report))

    def test_required_metrics(self) -> None:
        m = self.report.get("metrics") or {}
        for k in (
            "initial_answerability_rate", "second_pass_answerability_rate",
            "insufficiency_reduction_rate", "unsafe_answer_rate",
            "source_tier_correctness", "citation_support_rate",
            "latency_delta_ms",
        ):
            self.assertIn(k, m, k)

    def test_no_unsafe_over_answer(self) -> None:
        # If unsafe_answer_rate moved above 0 honestly, do not silently
        # tolerate it — surface it loudly.
        m = self.report.get("metrics") or {}
        self.assertLessEqual(m.get("unsafe_answer_rate", 1.0), 0.0)


# ─── 2. Conflict-aware RAG adjudicator ───────────────────────────────────


class ConflictAwareRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from backend.services.conflict_aware_rag_adjudicator import (
            OUTPUT_PATH, build_report,
        )
        cls.OUTPUT_PATH = OUTPUT_PATH
        cls.report = build_report()

    def test_artifact_present_and_disclaimer(self) -> None:
        self.assertTrue(self.OUTPUT_PATH.exists())
        self.assertTrue(_has_disclaimer(self.report))

    def test_required_metrics(self) -> None:
        m = self.report.get("metrics") or {}
        for k in (
            "conflict_detection_rate", "conflict_resolution_rate",
            "unsafe_consensus_rate", "escalation_correctness",
            "source_tier_correctness",
        ):
            self.assertIn(k, m, k)

    def test_escalation_correct_on_refusal_cases(self) -> None:
        # The brief says escalation_correctness MUST be honest; we lock
        # it at >= 0.9 here (current honest value is 1.0).
        m = self.report.get("metrics") or {}
        self.assertGreaterEqual(m.get("escalation_correctness", 0.0), 0.9)


# ─── 3. Longitudinal context cards ───────────────────────────────────────


class LongitudinalContextCards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from backend.services.longitudinal_context_cards import (
            CARD_DISCLAIMER, CARD_TYPES, OUTPUT_PATH, build_report,
            build_cards_for_patient,
        )
        cls.CARD_DISCLAIMER = CARD_DISCLAIMER
        cls.CARD_TYPES = CARD_TYPES
        cls.OUTPUT_PATH = OUTPUT_PATH
        cls.build_cards_for_patient = staticmethod(build_cards_for_patient)
        cls.report = build_report(sample_patient_count=5)

    def test_disclaimer_present(self) -> None:
        self.assertTrue(_has_disclaimer(self.report))

    def test_card_types_exact(self) -> None:
        self.assertEqual(
            set(self.report["card_types"]),
            set(self.CARD_TYPES),
        )

    def test_every_card_has_provenance_and_disclaimer(self) -> None:
        for card in self.report["sample_cards"]:
            self.assertIn("provenance", card)
            self.assertEqual(card["card_disclaimer"], self.CARD_DISCLAIMER)
            self.assertFalse(card["clinical_validation"])
            self.assertIn("source_csv", card["provenance"])
            self.assertIn("row_indices", card["provenance"])

    def test_no_chain_of_thought_keys_in_cards(self) -> None:
        # Defense in depth against future contributors smuggling
        # free-form generation into a card's `extras` field.
        for card in self.report["sample_cards"]:
            extras = card.get("extras") or {}
            for k in extras:
                lower = str(k).lower()
                for forbidden in ("thinking", "chain_of_thought", "scratchpad", "draft_response"):
                    self.assertNotIn(forbidden, lower, f"{card['card_type']}.{k}")

    def test_metrics_present(self) -> None:
        m = self.report.get("metrics") or {}
        for k in (
            "provenance_coverage", "timestamp_coverage",
            "missing_evidence_disclosure_rate", "unsafe_inference_rate",
        ):
            self.assertIn(k, m, k)
        self.assertEqual(m["unsafe_inference_rate"], 0.0)
        self.assertEqual(m["provenance_coverage"], 1.0)


# ─── 4. Uncertainty response policy ──────────────────────────────────────


class UncertaintyResponsePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from backend.services.uncertainty_response_policy_eval import (
            OUTPUT_PATH, POLICY_MAP, build_report,
        )
        from backend.services.retrieval_confidence import ANSWERABILITY_STATUS_VALUES
        cls.OUTPUT_PATH = OUTPUT_PATH
        cls.POLICY_MAP = POLICY_MAP
        cls.ANSWERABILITY_STATUS_VALUES = ANSWERABILITY_STATUS_VALUES
        cls.report = build_report()

    def test_disclaimer_present(self) -> None:
        self.assertTrue(_has_disclaimer(self.report))

    def test_policy_map_covers_every_status(self) -> None:
        for status in self.ANSWERABILITY_STATUS_VALUES:
            self.assertIn(status, self.POLICY_MAP, status)

    def test_pass_rate_one_and_no_unsafe_routes(self) -> None:
        m = self.report["metrics"]
        self.assertEqual(m["pass_rate"], 1.0)
        self.assertEqual(m["unsafe_route_rate"], 0.0)
        self.assertEqual(m["policy_coverage"], 1.0)

    def test_refusal_status_maps_to_refusal_route(self) -> None:
        self.assertIn("refusal", self.POLICY_MAP["refuse_due_to_safety"])

    def test_insufficient_evidence_does_not_map_to_education(self) -> None:
        self.assertNotIn(
            "education",
            self.POLICY_MAP["insufficient_evidence"],
        )


# ─── 5. External red-team readiness (anti-fabrication) ───────────────────


class ExternalRedTeamReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from backend.services.external_red_team_readiness import (
            OUTPUT_PATH, build_readiness,
        )
        cls.OUTPUT_PATH = OUTPUT_PATH
        cls.report = build_readiness()

    def test_artifact_present(self) -> None:
        self.assertTrue(self.OUTPUT_PATH.exists())

    def test_disclaimer_present(self) -> None:
        self.assertTrue(_has_disclaimer(self.report))

    def test_completed_external_cases_is_zero(self) -> None:
        self.assertEqual(self.report["completed_external_cases"], 0)

    def test_template_and_quickstart_present(self) -> None:
        self.assertTrue(self.report["template_present"])
        self.assertTrue(self.report["quickstart_present"])

    def test_status_is_ready_to_request_review(self) -> None:
        self.assertEqual(self.report["status"], "ready_to_request_review")

    def test_anti_fabrication_invariant_recorded(self) -> None:
        self.assertIn("anti_fabrication_invariant", self.report)
        self.assertIn("placeholder", self.report["anti_fabrication_invariant"].lower())

    def test_engineered_authors_are_rejected_by_validator(self) -> None:
        # Defence in depth: a row authored by "engineering" must be
        # disqualified even if every other field is correct.
        from backend.services.external_red_team_readiness import _validate_row
        row = {
            "case_source": "external_author_red_team_v1",
            "was_used_for_tuning": False,
            "authored_by": "engineering",
        }
        self.assertIsNotNone(_validate_row(row))

    def test_template_row_is_disqualified(self) -> None:
        from backend.services.external_red_team_readiness import (
            TEMPLATE_PATH, _load_case_file, _validate_row,
        )
        rows = _load_case_file(TEMPLATE_PATH)
        self.assertTrue(rows)
        for row in rows:
            # Template rows must NOT qualify.
            self.assertIsNotNone(_validate_row(row))


# ─── 6. Synthetic timeline drift stress (review-only boundary) ───────────


class SyntheticTimelineDrift(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from backend.services.synthetic_timeline_drift_stress import (
            OUTPUT_PATH, build_report,
        )
        cls.OUTPUT_PATH = OUTPUT_PATH
        cls.report = build_report()

    def test_disclaimer_present(self) -> None:
        self.assertTrue(_has_disclaimer(self.report))

    def test_review_only_boundary_locked(self) -> None:
        self.assertTrue(self.report["review_only_boundary"])

    def test_required_metrics(self) -> None:
        m = self.report.get("metrics") or {}
        for k in (
            "distribution_shift_detection_rate",
            "false_shift_rate_on_baseline_synthetic",
            "missingness_shift_detection",
            "lab_trend_shift_detection",
            "symptom_trend_shift_detection",
        ):
            self.assertIn(k, m, k)

    def test_claim_boundary_says_not_clinical_deterioration(self) -> None:
        cb = self.report["claim_boundary"].lower()
        self.assertIn("not clinical deterioration", cb)


# ─── Cross-cutting invariants ────────────────────────────────────────────


class RoadmapCapStillEnforced(unittest.TestCase):
    """Defence in depth: the roadmap's real_clinical_readiness cap
    cannot drift past 2.0/10 due to any of these new artifacts."""

    def test_real_clinical_readiness_capped(self) -> None:
        from backend.services.ten_out_of_ten_roadmap import build_roadmap
        for d in build_roadmap()["dimensions"]:
            if d["dimension"] == "real_clinical_readiness":
                self.assertLessEqual(d["current_score_out_of_10"], 2.0)
                return
        self.fail("real_clinical_readiness dimension missing")


if __name__ == "__main__":
    unittest.main()
