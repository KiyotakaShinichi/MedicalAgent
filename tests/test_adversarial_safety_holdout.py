"""Tests for the held-out adversarial variants.

Lock-ins:

* The held-out file is exactly 32 cases across 4 hardened categories.
* No held-out query is a substring (or substring-of) of any query in
  the original bank — this is the anti-contamination check.
* The runner produces the schema the eval drift tracker expects.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ORIGINAL_BANK = Path("Data/evals/safety/adversarial_safety_regression_bank.jsonl")
HOLDOUT_BANK = Path("Data/evals/safety/adversarial_safety_holdout_variants.jsonl")
HARDENED_CATEGORIES = {
    "privacy_pii",
    "prompt_injection",
    "genetic_risk_misinterpretation",
    "vus_misinterpretation",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class HoldoutFileShape(unittest.TestCase):
    def setUp(self) -> None:
        self.holdout = _load(HOLDOUT_BANK)

    def test_size_and_categories(self) -> None:
        self.assertEqual(len(self.holdout), 32)
        cats = {c["category"] for c in self.holdout}
        self.assertEqual(cats, HARDENED_CATEGORIES)

    def test_unique_case_ids(self) -> None:
        ids = [c["case_id"] for c in self.holdout]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_holdout_marked_not_used_for_tuning(self) -> None:
        for case in self.holdout:
            self.assertFalse(case["was_used_for_tuning"], case["case_id"])

    def test_holdout_queries_do_not_overlap_original_bank(self) -> None:
        """No held-out query should be a substring of an original-bank query (or vice versa) for the same category."""
        original = _load(ORIGINAL_BANK)
        original_by_cat: dict[str, list[str]] = {}
        for case in original:
            original_by_cat.setdefault(case["category"], []).append(case["query"].lower())

        for case in self.holdout:
            ho_q = case["query"].lower()
            for orig_q in original_by_cat.get(case["category"], []):
                self.assertFalse(
                    ho_q == orig_q or ho_q in orig_q or orig_q in ho_q,
                    f"{case['case_id']} overlaps original case: {orig_q!r}",
                )


class HoldoutSummaryShape(unittest.TestCase):
    """Runs the runner and confirms the summary keys the drift tracker reads exist."""

    def test_runner_summary_keys(self) -> None:
        import os
        os.environ["ONCOTRACK_FAST_MODE"] = "1"
        from scripts import run_adversarial_safety_holdout as runner  # type: ignore
        summary = runner.run(HOLDOUT_BANK, Path("Data/evals/safety/latest_adversarial_safety_holdout.json"))
        for key in ("status", "total_n", "pass_count", "fail_count", "skipped_count",
                    "overall_attack_block_rate", "by_category", "contamination_note"):
            self.assertIn(key, summary, key)
        self.assertEqual(set(summary["by_category"].keys()), HARDENED_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
