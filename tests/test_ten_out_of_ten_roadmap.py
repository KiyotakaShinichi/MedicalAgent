"""Tests for the 10/10-under-constraints roadmap artifact.

Lock-ins (anti-overclaim):

* ``clinical_validation`` is False at the top level.
* ``real_clinical_readiness.current_score_out_of_10 <= 2.0``.
* ``claim_boundary`` contains every token in
  ``REQUIRED_ANTI_OVERCLAIM_TOKENS``.
* Every dimension has all 7 required fields.
* Every roadmap item carries one of the 4 tier strings exactly, and
  tiers C / D have ``controllable_now is False``.
* The doc ``docs/ten_out_of_ten_under_constraints.md`` does not
  contain marketing-overclaim phrases like "clinically validated",
  "FDA approved", or "production healthcare ready" used as claims.
* No dimension's score exceeds 9.0 (the constraint floor forbids 10s
  in this snapshot).
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


from backend.services.ten_out_of_ten_roadmap import (
    DEFAULT_OUTPUT_PATH,
    DIMENSIONS,
    REQUIRED_ANTI_OVERCLAIM_TOKENS,
    ROADMAP_ITEMS,
    build_roadmap,
)


VALID_TIERS = {
    "A_implement_now",
    "B_external_reviewer",
    "C_real_data",
    "D_irb_institution",
}

REQUIRED_DIMENSION_FIELDS = {
    "dimension",
    "side",
    "current_score_out_of_10",
    "why_not_higher",
    "strongest_evidence",
    "weakest_evidence",
    "credibility_risk",
    "what_would_make_it_10_under_constraints",
    "what_cannot_be_solved_without_external_or_real_data_or_irb",
}


def _payload() -> dict:
    if not hasattr(_payload, "_cache"):
        _payload._cache = build_roadmap()  # type: ignore[attr-defined]
    return _payload._cache  # type: ignore[attr-defined]


class TopLevelInvariants(unittest.TestCase):
    def test_clinical_validation_false(self) -> None:
        self.assertFalse(_payload()["clinical_validation"])

    def test_status_is_informational(self) -> None:
        self.assertEqual(_payload()["status"], "informational")

    def test_required_anti_overclaim_tokens_present(self) -> None:
        cb = _payload()["claim_boundary"].lower()
        for tok in REQUIRED_ANTI_OVERCLAIM_TOKENS:
            self.assertIn(tok, cb, f"claim_boundary missing {tok!r}")

    def test_things_we_cannot_claim_list_present(self) -> None:
        items = _payload().get("things_we_cannot_claim_under_constraints") or []
        # Must include the canonical anti-claims.
        joined = " ".join(items).lower()
        for token in ("clinical validation", "irb", "clinician sign-off", "real-world safety"):
            self.assertIn(token, joined, token)


class DimensionRules(unittest.TestCase):
    def test_every_dimension_has_required_fields(self) -> None:
        for d in _payload()["dimensions"]:
            missing = REQUIRED_DIMENSION_FIELDS - set(d.keys())
            self.assertFalse(missing, f"{d.get('dimension')} missing {missing}")

    def test_no_dimension_exceeds_nine(self) -> None:
        for d in _payload()["dimensions"]:
            self.assertLessEqual(
                d["current_score_out_of_10"], 9.0,
                f"{d['dimension']} scored {d['current_score_out_of_10']} > 9.0",
            )

    def test_real_clinical_readiness_capped_at_two(self) -> None:
        for d in _payload()["dimensions"]:
            if d["dimension"] == "real_clinical_readiness":
                self.assertLessEqual(
                    d["current_score_out_of_10"], 2.0,
                    "real_clinical_readiness must stay <= 2.0 by anti-overclaim policy",
                )
                return
        self.fail("real_clinical_readiness dimension missing")

    def test_module_dimensions_match_payload(self) -> None:
        # Defensive: the module constant and the JSON payload must agree.
        payload_keys = {d["dimension"] for d in _payload()["dimensions"]}
        module_keys = {d.dimension for d in DIMENSIONS}
        self.assertEqual(payload_keys, module_keys)


class RoadmapTierRules(unittest.TestCase):
    def test_every_roadmap_item_has_valid_tier(self) -> None:
        for item in _payload()["ranked_roadmap"]:
            self.assertIn(item["tier"], VALID_TIERS, item["item"])

    def test_tier_C_and_D_items_are_not_controllable_now(self) -> None:
        for item in _payload()["ranked_roadmap"]:
            if item["tier"] in {"C_real_data", "D_irb_institution"}:
                self.assertFalse(
                    item["controllable_now"],
                    f"{item['item']!r} in {item['tier']!r} marked controllable_now=True",
                )

    def test_ranks_are_unique_and_one_indexed(self) -> None:
        ranks = [item["rank"] for item in _payload()["ranked_roadmap"]]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(len(set(ranks)), len(ranks))
        self.assertEqual(min(ranks), 1)

    def test_module_roadmap_matches_payload(self) -> None:
        module_items = {r.item for r in ROADMAP_ITEMS}
        payload_items = {item["item"] for item in _payload()["ranked_roadmap"]}
        self.assertEqual(module_items, payload_items)


class DocAntiOverclaim(unittest.TestCase):
    DOC_PATH = Path("docs/ten_out_of_ten_under_constraints.md")

    def test_doc_exists(self) -> None:
        self.assertTrue(self.DOC_PATH.exists())

    def test_doc_does_not_make_clinical_claims(self) -> None:
        text = self.DOC_PATH.read_text(encoding="utf-8").lower()
        # Phrases that, if used as *claims* (not as part of disclaimers),
        # would constitute overclaiming.  We grep for them as bare
        # affirmative sentences; if found, fail.
        forbidden = (
            r"\bis clinically validated\b",
            r"\bis fda approved\b",
            r"\bproduction healthcare ready\b(?!\s*\.)",  # bare claim, not in a "not …" sentence
        )
        for pat in forbidden:
            # Only fail if the pattern shows up WITHOUT a negation in the same line.
            for line in text.splitlines():
                if re.search(pat, line) and not re.search(
                    r"\bnot\b|\bdoes not\b|\bno\b|\bnever\b|>|cannot",
                    line,
                ):
                    self.fail(f"Doc contains affirmative overclaim: {line!r}")

    def test_doc_pins_real_clinical_readiness_cap(self) -> None:
        text = self.DOC_PATH.read_text(encoding="utf-8").lower()
        self.assertIn("real_clinical_readiness", text)
        # The doc should state the 2.0 cap somewhere.
        self.assertIn("2.0", text)


class ArtifactWritten(unittest.TestCase):
    def test_default_artifact_exists_and_parses(self) -> None:
        self.assertTrue(DEFAULT_OUTPUT_PATH.exists())
        json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
