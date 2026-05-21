from __future__ import annotations

import json
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

        rc = runner.main()
        self.assertEqual(rc, 0)
        artifact = json.loads(Path("Data/evals/safety/latest_adversarial_generalization_v2_eval.json").read_text(encoding="utf-8"))
        for key in ("original_bank", "heldout_v1", "heldout_v2", "paraphrase_robustness", "safe_negative_controls", "metrics"):
            self.assertIn(key, artifact)
        self.assertGreaterEqual(artifact["heldout_v2"]["total_n"], 100)
        self.assertFalse(any(artifact["heldout_v2"]["was_used_for_tuning"]))


if __name__ == "__main__":
    unittest.main()
