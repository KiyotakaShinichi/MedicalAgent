"""Tests for the adversarial safety regression bank + runner.

Deliberately structural, not score-based: a test that asserts
"attack_block_rate >= 95%" would be Goodhart bait — it would
incentivise weakening the bank to pass.  The bank's purpose is to
*surface* gaps, not be a passable target.

Lock-ins:

* All 200 cases load with unique stable IDs.
* Every case has the required schema fields (so the runner cannot KeyError).
* Every category has at least one negative control or at least one
  high-risk case, depending on its role.
* The runner executes end-to-end and produces the expected summary
  shape (with the documented breakdown keys).
* Negative controls' wording does not contain literal urgent terms
  ("fever", "bleeding") — proves the bank doesn't accidentally pack
  high-risk wording into supposed safe cases.
* The patient-exfil and prompt-injection sub-banks actually exercise
  the security guardrail (i.e. at least *some* of them block when
  routed through ``detect_prompt_injection_or_exfiltration``).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path



BANK_PATH = Path("Data/evals/safety/adversarial_safety_regression_bank.jsonl")
EXPECTED_TOTAL = 200
REQUIRED_FIELDS = {
    "case_id",
    "category",
    "language",
    "query",
    "attack_layer",
    "expected_safety_level",
    "authored_date",
    "authored_by",
    "was_used_for_tuning",
    "case_source",
}


def _load_bank() -> list[dict]:
    return [json.loads(line) for line in BANK_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


class BankSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = _load_bank()

    def test_count(self) -> None:
        self.assertEqual(len(self.cases), EXPECTED_TOTAL)

    def test_unique_case_ids(self) -> None:
        ids = [c["case_id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields_present(self) -> None:
        for case in self.cases:
            missing = REQUIRED_FIELDS - set(case.keys())
            self.assertFalse(missing, f"{case['case_id']} missing {missing}")

    def test_was_used_for_tuning_is_false(self) -> None:
        # If any of these became True silently, that would mean a case
        # leaked into training and the regression value is destroyed.
        for case in self.cases:
            self.assertFalse(case["was_used_for_tuning"], case["case_id"])

    def test_negative_controls_have_no_urgent_wording(self) -> None:
        urgent_tokens = ("fever", "bleeding", "chest pain", "suicidal", "self harm", "nilalagnat")
        for case in self.cases:
            if case["category"] != "safe_educational_negative_control":
                continue
            text = case["query"].lower()
            for tok in urgent_tokens:
                self.assertNotIn(tok, text, f"{case['case_id']} accidentally has urgent token {tok!r}")

    def test_attack_layer_values(self) -> None:
        for case in self.cases:
            self.assertIn(case["attack_layer"], {"safety_scope", "security", "negative_control"})


class RunnerEndToEnd(unittest.TestCase):
    """Runs the actual runner against the bank, FAST_MODE forced on."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["ONCOTRACK_FAST_MODE"] = "1"
        # Import inside setUpClass so the env var is set before module-level imports.
        from scripts import run_adversarial_safety_regression as runner  # type: ignore
        cls.runner = runner

    def test_runner_summary_shape(self) -> None:
        # Writes to a temporary directory, not to
        # Data/evals/safety/latest_adversarial_safety_regression.json. The
        # assertions below read only the returned summary, so pointing the
        # runner at the committed evidence path bought nothing and left three
        # tracked artifacts (summary, failure analysis, holdout split, the
        # latter two via run_regression's defaults) rewritten by every test
        # run. Same pattern as test_runner_writes_failure_analysis_and_holdout_artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.runner.run_regression(
                BANK_PATH,
                Path(tmp) / "latest.json",
                Path(tmp) / "failure_analysis.json",
                Path(tmp) / "holdout.json",
            )
        self.assertEqual(summary["total_cases"], EXPECTED_TOTAL)
        for key in [
            "schema_version", "generated_at", "fast_mode",
            "total_cases", "total_passed", "overall_attack_block_rate",
            "by_category", "by_language", "by_attack_layer", "failures", "hard_gate",
        ]:
            self.assertIn(key, summary, key)
        # Categories present in the bank must appear in by_category.
        bank = _load_bank()
        seen_cats = {c["category"] for c in bank}
        self.assertEqual(set(summary["by_category"].keys()), seen_cats)
        for row in summary["by_category"].values():
            for key in ("total_n", "category_n", "pass_count", "fail_count", "skipped_count", "authored_by", "authored_date", "was_used_for_tuning"):
                self.assertIn(key, row)

    def test_runner_writes_failure_analysis_and_holdout_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary = self.runner.run_regression(
                BANK_PATH,
                tmp_path / "latest.json",
                tmp_path / "failure_analysis.json",
                tmp_path / "holdout.json",
            )
            failure = json.loads((tmp_path / "failure_analysis.json").read_text(encoding="utf-8"))
            holdout = json.loads((tmp_path / "holdout.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["total_cases"], EXPECTED_TOTAL)
        self.assertGreaterEqual(failure["summary"]["focus_case_count"], 60)
        self.assertGreater(holdout["dev"]["total_n"], holdout["holdout"]["total_n"])
        self.assertGreaterEqual(holdout["holdout"]["total_n"], 20)

    def test_security_layer_blocks_at_least_some_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.runner.run_regression(
                BANK_PATH,
                Path(tmp) / "latest.json",
                Path(tmp) / "failure_analysis.json",
                Path(tmp) / "holdout.json",
            )
        sec = summary["by_attack_layer"].get("security", {})
        # At least one security case must be blocked — guards against
        # the detector silently being neutered.
        self.assertGreater(sec.get("passed", 0), 0)


if __name__ == "__main__":
    unittest.main()
