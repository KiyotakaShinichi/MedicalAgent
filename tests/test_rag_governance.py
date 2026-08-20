"""Tests for the KB source-governance layer + post-generation validator.

These guard the safety contract added in Phase 8b:
  - Every trust_level in TIER_MAP resolves to a valid tier + allowed_use.
  - Sources with an unmapped trust_level land in T5 with no allowed_use
    (so the validator refuses to cite them).
  - Staleness buckets are computed correctly relative to TTL.
  - The post-gen validator blocks every banned-claim category and lets
    safe educational text through.
"""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.services.kb_source_governance import (
    ALLOWED_USE_VOCABULARY,
    TIER_MAP,
    TIER_ORDER,
    build_kb_source_governance,
    load_kb_source_governance,
)
from backend.services.post_generation_validator import (
    ALL_RULES,
    DEFAULT_REFUSAL,
    validate_reply,
)


# ─── KB governance ───────────────────────────────────────────────────────────


class TierMapInvariants(unittest.TestCase):
    """The tier mapping is the single source of truth for what KB content
    can back what kind of claim.  Drift between tier / allowed_use /
    description would silently weaken the validator."""

    def test_every_tier_map_entry_is_well_formed(self) -> None:
        for trust_level, meta in TIER_MAP.items():
            self.assertIn(meta["tier"], TIER_ORDER, f"{trust_level} has invalid tier")
            self.assertGreaterEqual(meta["rank"], 1)
            self.assertLessEqual(meta["rank"], 5)
            self.assertGreater(len(meta["description"]), 20)
            for use in meta["allowed_use"]:
                self.assertIn(use, ALLOWED_USE_VOCABULARY)

    def test_safety_policy_is_top_tier(self) -> None:
        self.assertEqual(TIER_MAP["clinical_safety_policy"]["tier"], "T1")

    def test_patient_education_is_education_only(self) -> None:
        self.assertEqual(
            TIER_MAP["patient_education"]["allowed_use"], ("education",),
        )


class GovernanceArtifactContract(unittest.TestCase):
    """Build the governance artifact against a controlled KB fixture and
    confirm every required field is populated."""

    def test_builds_against_synthetic_kb_with_mixed_tiers(self) -> None:
        now = datetime.now(timezone.utc)
        fresh = now.isoformat()
        stale = (now - timedelta(days=400)).isoformat()
        kb = {
            "chunks": [
                {"id": "c1", "parent_id": "src_guideline", "title": "NCCN summary",
                 "trust_level": "clinical_guideline_summary", "ingested_at": fresh, "modality": ["text"]},
                {"id": "c2", "parent_id": "src_guideline", "trust_level": "clinical_guideline_summary",
                 "ingested_at": fresh, "modality": ["text"]},
                {"id": "c3", "parent_id": "src_patient", "title": "What is chemotherapy",
                 "trust_level": "patient_education", "ingested_at": fresh, "modality": ["text"]},
                {"id": "c4", "parent_id": "src_old_pub", "title": "Old systematic review",
                 "trust_level": "systematic_review", "ingested_at": stale, "modality": ["text"]},
                # Uncategorised trust_level → T5, allowed_use empty.
                {"id": "c5", "parent_id": "src_mystery", "title": "Mystery source",
                 "trust_level": "unknown_origin", "ingested_at": fresh, "modality": ["text"]},
            ],
        }
        with TemporaryDirectory() as tmp:
            kb_path = Path(tmp) / "kb.json"
            kb_path.write_text(json.dumps(kb))
            out_path = Path(tmp) / "gov.json"
            payload = build_kb_source_governance(
                kb_chunks_path=str(kb_path),
                output_path=str(out_path),
            )
            # Required top-level fields.
            for key in ("schema_version", "status", "source_count", "chunk_count",
                        "tier_distribution", "allowed_use_distribution",
                        "staleness_distribution", "sources", "governance_issues"):
                self.assertIn(key, payload)
            self.assertEqual(payload["source_count"], 4)
            # T1 + T3 + T2 + T5 distribution.
            self.assertEqual(payload["tier_distribution"].get("T1"), 1)
            self.assertEqual(payload["tier_distribution"].get("T2"), 1)
            self.assertEqual(payload["tier_distribution"].get("T3"), 1)
            self.assertEqual(payload["tier_distribution"].get("T5"), 1)
            # Mystery source must be T5 with empty allowed_use.
            mystery = next(s for s in payload["sources"] if s["source_id"] == "src_mystery")
            self.assertEqual(mystery["tier"], "T5")
            self.assertEqual(mystery["allowed_use"], [])
            # Stale source must land in needs_review.
            old = next(s for s in payload["sources"] if s["source_id"] == "src_old_pub")
            self.assertEqual(old["staleness_status"], "needs_review")
            # Governance flags should call out the T5 source.
            codes = {i["code"] for i in payload["governance_issues"]}
            self.assertIn("uncategorised_trust_level", codes)
            self.assertIn("needs_review", codes)

    def test_loader_returns_missing_shell(self) -> None:
        with TemporaryDirectory() as tmp:
            out = load_kb_source_governance(path=str(Path(tmp) / "absent.json"))
            self.assertEqual(out["status"], "missing")
            self.assertEqual(out["sources"], [])

    def test_missing_kb_returns_missing_payload(self) -> None:
        payload = build_kb_source_governance(
            kb_chunks_path="/nonexistent/kb.json",
            output_path="/tmp/should_not_be_written.json",
        )
        self.assertEqual(payload["status"], "missing")


# ─── Post-generation validator ───────────────────────────────────────────────


class ValidatorAllowsSafeReplies(unittest.TestCase):
    SAFE_TEXTS = (
        "WBC stands for white blood cells. They help the body respond to infection. Discuss anything concerning with your care team.",
        "Diagnosis is determined by a clinician; please bring this result to your oncology team.",
        "Reference ranges shown are population defaults. They are not personalised to you.",
    )

    def test_safe_texts_are_allowed(self) -> None:
        for text in self.SAFE_TEXTS:
            d = validate_reply(text)
            self.assertEqual(d.decision, "allowed", f"safe text flagged: {text!r}")
            self.assertEqual(d.triggered_rules, [])

    def test_empty_output_fails_closed(self) -> None:
        decision = validate_reply("")
        self.assertEqual(decision.decision, "blocked")
        self.assertIn("malformed_or_empty_output", decision.triggered_rules)


class ValidatorBlocksEachBannedClaim(unittest.TestCase):
    """One test method per rule code so a regression surfaces the exact
    category that broke."""

    def test_blocks_diagnosis_claim(self) -> None:
        d = validate_reply("Based on your symptoms, you have breast cancer.")
        self.assertEqual(d.decision, "blocked")
        self.assertIn("diagnosis_claim", d.triggered_rules)
        self.assertEqual(d.severity, "high")
        self.assertEqual(d.suggested_response, DEFAULT_REFUSAL)

    def test_blocks_lesion_malignant_phrasing(self) -> None:
        d = validate_reply("The lesion is malignant.")
        self.assertIn("diagnosis_claim", d.triggered_rules)

    def test_blocks_treatment_recommendation(self) -> None:
        d = validate_reply("I recommend you take tamoxifen daily.")
        self.assertIn("treatment_recommendation", d.triggered_rules)

    def test_blocks_dose_change_phrasing(self) -> None:
        d = validate_reply("Stop your chemo this cycle.")
        self.assertIn("treatment_recommendation", d.triggered_rules)

    def test_blocks_prognosis_with_percentage(self) -> None:
        d = validate_reply("Survival rate is 85% for this stage.")
        self.assertIn("prognosis_estimate", d.triggered_rules)

    def test_blocks_prognosis_months_to_live(self) -> None:
        d = validate_reply("You have 6 months left.")
        self.assertIn("prognosis_estimate", d.triggered_rules)

    def test_blocks_dosage_instruction(self) -> None:
        d = validate_reply("Take 200 mg twice a day.")
        self.assertIn("dosage_instruction", d.triggered_rules)

    def test_blocks_vus_overclaim(self) -> None:
        d = validate_reply("A VUS means you are positive for BRCA.")
        self.assertIn("genetic_risk_overclaim", d.triggered_rules)

    def test_blocks_relatives_will_develop_cancer(self) -> None:
        d = validate_reply("Your siblings will develop cancer.")
        self.assertIn("genetic_risk_overclaim", d.triggered_rules)

    def test_blocks_tumor_marker_recurrence_claim(self) -> None:
        d = validate_reply(
            "Your elevated CA 15-3 indicates recurrence and your cancer has come back based on the marker.",
        )
        self.assertIn("tumor_marker_overclaim", d.triggered_rules)


class ValidatorDecisionEnvelope(unittest.TestCase):
    def test_blocked_replies_carry_excerpts_and_severity(self) -> None:
        d = validate_reply("You have breast cancer.")
        self.assertEqual(d.decision, "blocked")
        self.assertEqual(d.severity, "high")
        self.assertGreater(len(d.matched_excerpts), 0)
        self.assertEqual(d.matched_excerpts[0]["rule"], "diagnosis_claim")
        self.assertIn("cancer", d.matched_excerpts[0]["excerpt"].lower())

    def test_to_dict_is_json_serialisable(self) -> None:
        d = validate_reply("Take 200 mg twice a day.")
        as_dict = d.to_dict()
        # round-trip through JSON to assert no surprise types
        round_tripped = json.loads(json.dumps(as_dict))
        self.assertEqual(round_tripped["decision"], "blocked")

    def test_rule_catalog_is_non_empty_and_unique(self) -> None:
        codes = [r.code for r in ALL_RULES]
        self.assertEqual(len(codes), len(set(codes)), "rule codes must be unique")
        self.assertGreaterEqual(len(codes), 6)


if __name__ == "__main__":
    unittest.main()
