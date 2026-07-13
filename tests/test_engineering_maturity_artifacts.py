"""Tests for the four engineering-maturity artifacts added in the
deployment-shaped polish pass:

1. multiturn_adversarial_agent_eval
2. noisier_synthetic_v2_stress
3. claim_source_alignment_hardening
4. deployment_boundary_check

Every artifact must carry ``clinical_validation: false`` and a
``claim_boundary`` that says "not clinical validation".  The
deployment artifact's ``status_label`` is permanently locked to
``production_shaped_not_healthcare_production_ready``.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path


from backend.services.multiturn_adversarial_agent_eval import (
    CASES_PATH as MT_CASES_PATH,
    OUTPUT_PATH as MT_OUTPUT,
    build_report as build_mt,
)
from backend.services.noisier_synthetic_v2_stress import (
    NOISE_TYPES,
    OUTPUT_PATH as STRESS_OUTPUT,
    build_report as build_stress,
)
from backend.services.claim_source_alignment_hardening import (
    OUTPUT_PATH as ALIGN_OUTPUT,
    build_report as build_align,
)
from backend.services.deployment_boundary_check import (
    CHECKS as DEPLOYMENT_CHECKS,
    FIXED_STATUS_LABEL,
    OUTPUT_PATH as DEPLOY_OUTPUT,
    build_report as build_deploy,
)


# ─── 1. Multiturn adversarial agent eval ─────────────────────────────────


class MultiturnAdversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ONCOTRACK_FAST_MODE"] = "1"
        cls.report = build_mt()

    def test_artifact_written(self) -> None:
        self.assertTrue(MT_OUTPUT.exists())

    def test_clinical_validation_false(self) -> None:
        self.assertFalse(self.report["clinical_validation"])

    def test_claim_boundary_says_not_clinical_validation(self) -> None:
        cb = self.report["claim_boundary"].lower()
        self.assertTrue("not clinical" in cb or "clinical validation" in cb)
        self.assertIn("does not", cb)

    def test_cases_loaded(self) -> None:
        self.assertGreaterEqual(self.report["n_cases"], 10)

    def test_required_metrics_present(self) -> None:
        m = self.report["metrics"]
        for key in (
            "unsafe_leakage_rate", "refusal_consistency", "route_stability",
            "escalation_correctness", "memory_boundary_correctness",
            "tool_use_correctness", "over_refusal_rate",
            "safe_negative_control_pass_rate",
            "post_gen_validator_catch_rate",
        ):
            self.assertIn(key, m, key)

    def test_per_case_carries_turn_results(self) -> None:
        for case in self.report["per_case"]:
            self.assertGreaterEqual(len(case["turns"]), 1)
            for turn in case["turns"]:
                for key in ("text", "expected_blocked", "actual_blocked"):
                    self.assertIn(key, turn)

    def test_was_used_for_tuning_false_in_cases_file(self) -> None:
        for line in MT_CASES_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            self.assertFalse(case["was_used_for_tuning"], case.get("case_id"))


# ─── 2. Noisier synthetic v2 stress benchmark ────────────────────────────


class NoisierSyntheticV2Stress(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_stress()

    def test_artifact_written(self) -> None:
        self.assertTrue(STRESS_OUTPUT.exists())

    def test_clinical_validation_false(self) -> None:
        self.assertFalse(self.report["clinical_validation"])

    def test_claim_boundary_says_not_clinical_validation(self) -> None:
        self.assertIn("not clinical", self.report["claim_boundary"].lower())

    def test_global_promotion_decision_is_reject_or_hold(self) -> None:
        self.assertEqual(self.report["global_promotion_decision"], "reject_or_hold")

    def test_per_noise_type_complete(self) -> None:
        names = {entry["noise_type"] for entry in self.report["per_noise_type"]}
        expected = {name for name, _ in NOISE_TYPES}
        self.assertEqual(names, expected)

    def test_each_noise_type_has_deltas_and_promotion_reject(self) -> None:
        for entry in self.report["per_noise_type"]:
            self.assertEqual(entry["promotion_decision"], "reject_or_hold")
            for key in (
                "calibration_delta", "brier_delta", "AUROC_delta",
                "regression_MAE_delta", "abstention_rate_delta",
                "shortcut_risk_delta",
            ):
                self.assertIn(key, entry["deltas"], key)

    def test_leakage_status_per_noise_present(self) -> None:
        allowed = {
            "leakage_suspect_metric_too_high_under_noise",
            "no_leakage_tripwire_fired",
        }
        for entry in self.report["per_noise_type"]:
            self.assertIn(entry["leakage_status"], allowed)


# ─── 3. Claim-source alignment hardening ─────────────────────────────────


class ClaimSourceAlignmentHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_align()

    def test_artifact_written(self) -> None:
        self.assertTrue(ALIGN_OUTPUT.exists())

    def test_clinical_validation_false(self) -> None:
        self.assertFalse(self.report["clinical_validation"])

    def test_claim_boundary_says_not_clinical_grade(self) -> None:
        cb = self.report["claim_boundary"].lower()
        self.assertIn("not clinical", cb)
        # The brief is explicit: never claim clinical-grade entailment.
        self.assertIn("not clinical-grade entailment", cb)

    def test_validator_method_one_of_three(self) -> None:
        self.assertIn(
            self.report["validator_method"],
            {"heuristic", "embedding", "optional_nli"},
        )

    def test_per_row_required_fields(self) -> None:
        required = {
            "row_id", "case_id", "claim_text", "expected_source_ids",
            "source_tier", "allowed_use", "patient_facing_allowed",
            "support_status", "contradiction_category", "validator_method",
            "alignment_action", "clinical_validation",
        }
        for row in self.report["rows"]:
            missing = required - set(row.keys())
            self.assertFalse(missing, f"{row.get('row_id')} missing {missing}")
            self.assertFalse(row["clinical_validation"])

    def test_support_status_in_enum(self) -> None:
        allowed = {
            "supported", "partially_supported", "unsupported",
            "contradicted", "insufficient_evidence",
        }
        for row in self.report["rows"]:
            self.assertIn(row["support_status"], allowed, row["row_id"])


# ─── 4. Deployment-boundary check ────────────────────────────────────────


class DeploymentBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_deploy()

    def test_artifact_written(self) -> None:
        self.assertTrue(DEPLOY_OUTPUT.exists())

    def test_status_label_locked_to_production_shaped(self) -> None:
        # The brief and ADR-style invariant: this string never changes.
        self.assertEqual(
            self.report["status_label"],
            "production_shaped_not_healthcare_production_ready",
        )
        self.assertEqual(self.report["status_label"], FIXED_STATUS_LABEL)

    def test_no_hipaa_or_clinical_deployment_claim(self) -> None:
        self.assertTrue(self.report["no_hipaa_compliance_claim"])
        self.assertTrue(self.report["no_clinical_deployment_claim"])

    def test_clinical_validation_false(self) -> None:
        self.assertFalse(self.report["clinical_validation"])

    def test_what_this_does_not_certify_includes_hipaa(self) -> None:
        items = [s.lower() for s in self.report["what_this_does_not_certify"]]
        for required in ("hipaa", "soc 2", "fda", "clinical safety", "production healthcare deployment"):
            joined = " | ".join(items)
            self.assertIn(required, joined, required)

    def test_checks_have_expected_shape(self) -> None:
        names = {c.name for c in DEPLOYMENT_CHECKS}
        report_names = {r["name"] for r in self.report["checks"]}
        self.assertEqual(report_names, names)

    def test_doc_exists_and_pins_the_locked_label(self) -> None:
        doc = Path("docs/deployment_boundary.md")
        self.assertTrue(doc.exists())
        text = doc.read_text(encoding="utf-8")
        self.assertIn("production_shaped_not_healthcare_production_ready", text)
        lowered = text.lower()
        self.assertIn("not hipaa compliant", lowered)
        self.assertIn("not clinically deployed", lowered)


if __name__ == "__main__":
    unittest.main()
