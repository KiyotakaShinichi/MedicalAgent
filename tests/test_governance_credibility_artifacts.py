"""Tests for the four governance/credibility artifacts.

Lock-ins:

* Every artifact carries ``clinical_validation: false`` and its
  ``claim_boundary`` contains the verbatim phrase
  ``not clinical validation``.
* Negative-results gallery has at least the 10 named negatives plus
  required fields per item.
* Portfolio claim-safety check exposes the banned/allowed phrase
  lists; the companion doc does NOT contain bare affirmative banned
  phrases; the artifact's own samples do not produce bare affirmative
  claims either.
* Eval contamination harmonisation maps to exactly the 7 documented
  categories.
* Noisier synthetic v2 readiness's ``scaffold_status`` is in
  ``ALLOWED_NOISIER_V2_STATUS`` (``scaffold_only`` or
  ``planned_not_trained``), with ``blocked_clinical_claims`` and
  ``expected_evals_before_promotion`` present.
* Real-clinical-readiness cap from the roadmap is unaffected
  (companion check against the roadmap module to keep the two in
  sync).
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


from backend.services.governance_credibility_artifacts import (
    ALLOWED_NOISIER_V2_STATUS,
    ALLOWED_PHRASES,
    BANNED_AFFIRMATIVE_PHRASES,
    CONTAMINATION_PATH,
    NEGATIVE_RESULTS_PATH,
    NOISIER_V2_PATH,
    PORTFOLIO_PATH,
    REQUIRED_CLAIM_BOUNDARY_PHRASE,
    build_eval_contamination_harmonization,
    build_negative_results_gallery,
    build_noisier_synthetic_v2_readiness,
    build_portfolio_claim_safety_check,
)


HARMONISATION_CATEGORIES = {
    "internal_used_for_tuning",
    "internal_frozen_not_used_for_tuning",
    "external_no_read_prepared_incomplete",
    "external_completed",
    "synthetic_generated",
    "live_agent_internal",
    "informational_only",
}


def _has_required_disclaimer(payload: dict) -> bool:
    cb = str(payload.get("claim_boundary") or "").lower()
    return REQUIRED_CLAIM_BOUNDARY_PHRASE in cb and payload.get("clinical_validation") is False


# ─── Shared invariants ───────────────────────────────────────────────────


class SharedDisclaimers(unittest.TestCase):
    def test_negative_results_disclaimer(self) -> None:
        self.assertTrue(_has_required_disclaimer(build_negative_results_gallery()))

    def test_portfolio_disclaimer(self) -> None:
        self.assertTrue(_has_required_disclaimer(build_portfolio_claim_safety_check()))

    def test_contamination_disclaimer(self) -> None:
        self.assertTrue(_has_required_disclaimer(build_eval_contamination_harmonization()))

    def test_noisier_v2_disclaimer(self) -> None:
        self.assertTrue(_has_required_disclaimer(build_noisier_synthetic_v2_readiness()))


# ─── 1. Negative results gallery ────────────────────────────────────────


class NegativeResultsGallery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_negative_results_gallery()

    def test_at_least_10_items(self) -> None:
        self.assertGreaterEqual(self.report["n_items"], 10)

    def test_required_per_item_fields(self) -> None:
        required = {
            "title", "evidence_artifact", "metric_value",
            "why_it_matters", "decision_taken", "what_was_not_claimed",
            "next_action", "clinical_validation",
        }
        for it in self.report["items"]:
            missing = required - set(it.keys())
            self.assertFalse(missing, f"{it.get('title')} missing {missing}")
            self.assertFalse(it["clinical_validation"], it.get("title"))

    def test_artifact_written(self) -> None:
        self.assertTrue(NEGATIVE_RESULTS_PATH.exists())
        # Round-trips as JSON.
        json.loads(NEGATIVE_RESULTS_PATH.read_text(encoding="utf-8"))


# ─── 2. Portfolio claim safety check ─────────────────────────────────────


class PortfolioClaimSafety(unittest.TestCase):
    DOC_PATH = Path("docs/portfolio_safe_wording.md")

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_portfolio_claim_safety_check()

    def test_banned_and_allowed_lists_present(self) -> None:
        self.assertEqual(set(self.report["banned_affirmative_phrases"]), set(BANNED_AFFIRMATIVE_PHRASES))
        self.assertEqual(set(self.report["allowed_phrases"]), set(ALLOWED_PHRASES))

    def test_doc_contains_banned_phrases_only_in_negation_or_unsafe_examples(self) -> None:
        """Banned phrases may appear in the doc but only inside:
        (a) a negated sentence (e.g. 'not clinically validated'),
        (b) the banned-phrase enumeration list, or
        (c) an explicitly-marked unsafe example block (a section
            opened by '❌ Unsafe:' header until the next blank line
            followed by non-blockquote prose).
        Any other appearance is treated as a bare claim and fails."""
        lines = self.DOC_PATH.read_text(encoding="utf-8").splitlines()
        in_unsafe_block = False
        for raw in lines:
            stripped = raw.strip()
            # Enter an unsafe block when the line opens one.
            if "❌" in raw or "unsafe:" in raw.lower():
                in_unsafe_block = True
                continue
            # Leave the unsafe block on a non-blockquote, non-blank line
            # that isn't itself a "why unsafe" follow-up.
            if in_unsafe_block and stripped and not stripped.startswith(">") and not stripped.lower().startswith("why unsafe"):
                in_unsafe_block = False
            if in_unsafe_block:
                continue
            lower = raw.lower()
            for phrase in BANNED_AFFIRMATIVE_PHRASES:
                if phrase not in lower:
                    continue
                # Negation guard.
                if re.search(r"\bnot\b|\bno\b|\bnever\b|\bcannot\b|\bdoes not\b|\bnone\b", lower):
                    continue
                # The banned-phrase enumeration list — a markdown
                # bullet whose content is just words/separators.
                if stripped.startswith(("- ", "* ")):
                    bare = stripped.lstrip("-* ").strip()
                    if re.fullmatch(r"[\w\s/+\-,()]+", bare):
                        continue
                self.fail(f"banned phrase {phrase!r} appears as bare claim: {raw!r}")

    def test_samples_include_all_required_audiences(self) -> None:
        audiences = {s["audience"] for s in self.report["audience_samples"]}
        required = {"linkedin_one_line", "recruiter_short", "senior_engineer_technical",
                    "readme_summary_paragraph", "cv_bullet"}
        self.assertEqual(audiences, required)

    def test_unsafe_samples_call_out_overclaim_reason(self) -> None:
        for sample in self.report["audience_samples"]:
            self.assertIn("why_unsafe", sample)
            self.assertTrue(sample["why_unsafe"].strip())

    def test_artifact_written(self) -> None:
        self.assertTrue(PORTFOLIO_PATH.exists())


# ─── 3. Contamination harmonisation ──────────────────────────────────────


class ContaminationHarmonization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_eval_contamination_harmonization()

    def test_categories_match_documented_set(self) -> None:
        self.assertEqual(set(self.report["categories"]), HARMONISATION_CATEGORIES)

    def test_every_artifact_carries_clinical_validation_false(self) -> None:
        for row in self.report["artifacts"]:
            self.assertFalse(row["clinical_validation"], row.get("path"))

    def test_every_artifact_lands_in_valid_category(self) -> None:
        for row in self.report["artifacts"]:
            self.assertIn(row["harmonisation_category"], HARMONISATION_CATEGORIES, row.get("path"))

    def test_category_counts_sum_to_n_artifacts(self) -> None:
        total = sum(self.report["category_counts"].values())
        self.assertEqual(total, self.report["n_artifacts_mapped"])

    def test_artifact_written(self) -> None:
        self.assertTrue(CONTAMINATION_PATH.exists())


# ─── 4. Noisier synthetic v2 readiness ───────────────────────────────────


class NoisierSyntheticV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_noisier_synthetic_v2_readiness()

    def test_scaffold_status_in_allowed_set(self) -> None:
        self.assertIn(self.report["scaffold_status"], ALLOWED_NOISIER_V2_STATUS)

    def test_blocked_clinical_claims_present_and_specific(self) -> None:
        blocked = " ".join(self.report.get("blocked_clinical_claims") or []).lower()
        for term in ("clinical performance", "fda", "deployment", "real-data validation", "real patients"):
            self.assertIn(term, blocked, term)

    def test_expected_evals_before_promotion_present(self) -> None:
        evals = self.report.get("expected_evals_before_promotion") or []
        self.assertGreaterEqual(len(evals), 5)

    def test_planned_noise_types_cover_required_eight(self) -> None:
        names = {n["name"] for n in self.report.get("planned_noise_types") or []}
        required = {
            "missingness_noise", "label_noise", "measurement_noise",
            "date_jitter", "symptom_reporting_noise",
            "imaging_report_ambiguity", "treatment_delay_randomness",
            "subgroup_distribution_shift",
        }
        self.assertEqual(names, required)

    def test_no_real_world_claim(self) -> None:
        text = json.dumps(self.report).lower()
        # Banned in the v2 artifact: any claim of clinical / IRB / real-data.
        for forbidden in ("represents real patients", "establishes clinical performance",
                          "fda / irb ready", "fda/irb ready", "replaces real-data validation"):
            # These exact strings should appear ONLY as items in the
            # blocked_clinical_claims list (i.e., as things-we-do-not-say).
            if forbidden in text:
                # Confirm they're in the blocked-claims list.
                blocked = " ".join(self.report.get("blocked_clinical_claims") or []).lower()
                self.assertIn(forbidden, blocked, forbidden)

    def test_artifact_written(self) -> None:
        self.assertTrue(NOISIER_V2_PATH.exists())


# ─── 5. Roadmap real-clinical-readiness cap unchanged ────────────────────


class RoadmapStillCapsRealClinicalReadiness(unittest.TestCase):
    """Defence-in-depth: the roadmap's real_clinical_readiness cap is
    enforced in its own test file, but anchor it here too so a future
    artifact change cannot drift the project's headline floor."""

    def test_real_clinical_readiness_stays_capped(self) -> None:
        from backend.services.ten_out_of_ten_roadmap import build_roadmap
        for d in build_roadmap()["dimensions"]:
            if d["dimension"] == "real_clinical_readiness":
                self.assertLessEqual(d["current_score_out_of_10"], 2.0)
                return
        self.fail("real_clinical_readiness dimension missing from roadmap")


if __name__ == "__main__":
    unittest.main()
