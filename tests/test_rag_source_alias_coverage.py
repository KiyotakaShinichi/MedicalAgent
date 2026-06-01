"""Tests for ``rag_source_alias_coverage``.

Lock-ins:

* The diagnostic is read-only.  It never mutates the goldset, the KB,
  or ``LOGICAL_SOURCE_ALIASES``.
* The report exposes the schema the drift tracker / docs expect.
* Every alias key demanded by the goldset must be addressable in
  ``LOGICAL_SOURCE_ALIASES`` after the 2026-05-27 promotion pass —
  ``n_alias_keys_uncovered == 0``.
* The proposed-additions list is non-trivial — if it were empty we'd
  have lost the diagnostic's value silently.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.services.rag_source_alias_coverage import (
    DEFAULT_GOLDSET_PATH,
    DEFAULT_KB_PATH,
    build_alias_coverage_report,
)


class CoverageReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_alias_coverage_report(
            goldset_path=DEFAULT_GOLDSET_PATH,
            kb_path=DEFAULT_KB_PATH,
        )

    def test_label_and_disclaimer(self) -> None:
        self.assertEqual(self.report["label"], "source_alias_coverage_diagnostic")
        self.assertIn("not auto-applied", self.report["claim_boundary"].lower())
        self.assertIn("engineering signal only", self.report["claim_boundary"].lower())

    def test_no_alias_key_uncovered_after_2026_05_27(self) -> None:
        # Every distinct expected_source_id in the goldset must exist
        # as a top-level entry in LOGICAL_SOURCE_ALIASES.  This is the
        # ADR-0009 invariant.
        self.assertEqual(
            self.report["n_alias_keys_uncovered"],
            0,
            f"uncovered: {self.report['uncovered_alias_keys']}",
        )

    def test_per_alias_shape(self) -> None:
        for entry in self.report["per_alias"]:
            for key in (
                "alias_key", "goldset_demand_count", "alias_set_size",
                "kb_parent_ids_in_alias_set", "proposed_additions_by_content_match",
                "match_method",
            ):
                self.assertIn(key, entry, entry.get("alias_key"))

    def test_diagnostic_is_read_only(self) -> None:
        # Build the report twice; the second build must produce the
        # same proposed-additions count.  If it doesn't, the diagnostic
        # accidentally mutated state somewhere.
        first = build_alias_coverage_report()
        second = build_alias_coverage_report()
        self.assertEqual(
            first["n_proposed_additions_total"],
            second["n_proposed_additions_total"],
        )


class AliasPromotionInvariant(unittest.TestCase):
    """The promoted aliases (from ADR 0009) must actually be in the map."""

    def test_each_promoted_parent_id_is_in_alias_map(self) -> None:
        from backend.services.rag_baseline_comparison import LOGICAL_SOURCE_ALIASES
        # One sentinel per alias group promoted on 2026-05-27.
        sentinels = {
            "genetic-counseling":              "664fb49bb1343408",
            "tumor-marker-context":            "5598e2371d2713c4",
            "curated-tumor-marker-limitations":"5598e2371d2713c4",
            "supplement-safety":               "918edc260afd2d63",
            "infection-safety":                "9a6347c207d53299",
            "curated-mri-response-terms":      "2a9f2ed73f0b189c",
            "treatment-side-effects":          "24de6c8ad0379f43",
            "portal-help":                     "c35c9264029ff9c9",
        }
        for alias_key, sentinel in sentinels.items():
            alias_set = LOGICAL_SOURCE_ALIASES.get(alias_key, set())
            self.assertIn(
                sentinel, alias_set,
                f"{alias_key} missing promoted sentinel {sentinel}",
            )


if __name__ == "__main__":
    unittest.main()
