from __future__ import annotations

import functools
import json
import tempfile
import unittest
from pathlib import Path


HOLDOUT_V2 = Path("Data/evals/safety/adversarial_holdout_v2.jsonl")
EXPECTED_CATEGORIES = {
    "privacy_pii",
    "prompt_injection",
    "cross_patient_exfil",
    "genetic_risk_misinterpretation",
    "vus_misinterpretation",
    "diagnosis_confirmation",
    "tumor_marker_overclaim",
    "treatment_change",
    "dosage_request",
    "prognosis_estimate",
    "supplement_replacement",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class HoldoutV2ShapeTests(unittest.TestCase):
    def test_v2_has_expected_size_categories_and_metadata(self) -> None:
        rows = _load(HOLDOUT_V2)
        self.assertGreaterEqual(len(rows), 100)
        self.assertEqual({row["category"] for row in rows}, EXPECTED_CATEGORIES)
        for category in EXPECTED_CATEGORIES:
            self.assertGreaterEqual(sum(1 for row in rows if row["category"] == category), 10)
        for row in rows:
            self.assertFalse(row["was_used_for_tuning"], row["case_id"])
            self.assertIn("authored_by", row)
            self.assertIn("authored_date", row)
            self.assertIn("contamination_note", row)
            self.assertIn("expected_route", row)
            self.assertIn("expected_refusal_or_escalation", row)
            self.assertIn("safe_negative", row)

    def test_v2_eval_artifact_shape(self) -> None:
        from scripts import run_adversarial_generalization_v2_eval as runner  # type: ignore

        # `main()` takes no arguments: it writes its own artifact to a
        # module-level OUTPUT_PATH and calls run_regression()/run_holdout_v1()
        # with their default destinations, so an unmodified call rewrote five
        # tracked evidence files under Data/evals/safety/. All three
        # destinations are therefore redirected on the module rather than
        # passed in. `main()` looks each name up as a global at call time, and
        # the two runner defaults are bound at def-time, which is why the
        # functions are wrapped rather than their DEFAULT_* constants patched.
        #
        # This changes where the artifacts land, not what is computed: the
        # assertions below read back the same artifact this test just wrote.
        originals = (runner.OUTPUT_PATH, runner.run_regression, runner.run_holdout_v1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_path = tmp_dir / "latest_adversarial_generalization_v2_eval.json"
            runner.OUTPUT_PATH = out_path
            runner.run_regression = functools.partial(
                originals[1],
                output_path=tmp_dir / "regression.json",
                failure_analysis_path=tmp_dir / "failure_analysis.json",
                holdout_output_path=tmp_dir / "regression_holdout.json",
            )
            runner.run_holdout_v1 = functools.partial(
                originals[2], output_path=tmp_dir / "holdout_v1.json"
            )
            try:
                rc = runner.main()
                self.assertEqual(rc, 0)
                artifact = json.loads(out_path.read_text(encoding="utf-8"))
            finally:
                (
                    runner.OUTPUT_PATH,
                    runner.run_regression,
                    runner.run_holdout_v1,
                ) = originals
        for key in ("original_bank", "heldout_v1", "heldout_v2", "paraphrase_robustness", "safe_negative_controls", "metrics"):
            self.assertIn(key, artifact)
        self.assertGreaterEqual(artifact["heldout_v2"]["total_n"], 100)
        self.assertFalse(any(artifact["heldout_v2"]["was_used_for_tuning"]))


if __name__ == "__main__":
    unittest.main()
