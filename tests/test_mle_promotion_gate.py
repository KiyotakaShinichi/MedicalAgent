"""Tests for ``scripts.run_mle_promotion_gate``.

The gate aggregates real audit artifacts under ``Data/evals/models/``.
These tests:

  - cover the comparator helpers (``_compare`` / ``_dig``) with edge
    cases (None, non-numeric, missing path).
  - exercise the decision logic by feeding synthetic condition rows
    (no real artifacts touched).
  - exercise an end-to-end run against a temp config + tmp artifact
    directory so the test is hermetic.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


# Load the script as a module under the same import path so tests can
# call its private helpers without spawning a subprocess.
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_mle_promotion_gate.py"
_spec = importlib.util.spec_from_file_location("run_mle_promotion_gate", SCRIPT_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["run_mle_promotion_gate"] = gate
_spec.loader.exec_module(gate)  # type: ignore[union-attr]


class Comparators(unittest.TestCase):
    def test_compare_none_op_is_noop(self) -> None:
        self.assertTrue(gate._compare(0, None, 1))
        self.assertTrue(gate._compare("x", "", "y"))

    def test_compare_actual_none_is_fail(self) -> None:
        self.assertFalse(gate._compare(None, ">=", 0))

    def test_compare_equality(self) -> None:
        self.assertTrue(gate._compare(0, "==", 0))
        self.assertFalse(gate._compare(0, "==", 1))
        self.assertTrue(gate._compare(0, "!=", 1))

    def test_compare_numeric_ops(self) -> None:
        self.assertTrue(gate._compare(2, ">=", 2))
        self.assertTrue(gate._compare(2.1, ">", 2))
        self.assertTrue(gate._compare(1.0, "<=", 2.0))
        self.assertFalse(gate._compare(3, "<", 2))

    def test_compare_non_numeric_falls_to_false(self) -> None:
        self.assertFalse(gate._compare("abc", ">=", 1))

    def test_dig_walks_dict_and_list(self) -> None:
        payload = {"summary": {"items": [{"score": 0.5}, {"score": 0.9}]}}
        self.assertEqual(gate._dig(payload, ["summary", "items", 1, "score"]), 0.9)

    def test_dig_returns_none_for_missing_path(self) -> None:
        self.assertIsNone(gate._dig({"a": 1}, ["a", "b"]))
        self.assertIsNone(gate._dig({"a": [1, 2]}, ["a", 99]))


class DecisionLogic(unittest.TestCase):
    def test_all_passing_is_promote(self) -> None:
        results = [
            {"name": "a", "passed": True, "critical": True, "decision_on_fail": "REJECT", "issues": []},
            {"name": "b", "passed": True, "critical": False, "decision_on_fail": "HOLD",  "issues": []},
        ]
        decision, reasons = gate._decision_from_results(results)
        self.assertEqual(decision, "PROMOTE")
        self.assertEqual(reasons, [])

    def test_one_critical_reject_forces_reject(self) -> None:
        results = [
            {"name": "leak", "passed": False, "critical": True, "decision_on_fail": "REJECT", "issues": ["leak"]},
            {"name": "other", "passed": True, "critical": False, "decision_on_fail": "HOLD", "issues": []},
        ]
        decision, _ = gate._decision_from_results(results)
        self.assertEqual(decision, "REJECT")

    def test_non_critical_failure_holds(self) -> None:
        results = [
            {"name": "shortcut", "passed": False, "critical": False, "decision_on_fail": "HOLD", "issues": ["meh"]},
        ]
        decision, _ = gate._decision_from_results(results)
        self.assertEqual(decision, "HOLD")

    def test_critical_hold_does_not_escalate_to_reject(self) -> None:
        results = [
            {"name": "calibration", "passed": False, "critical": True, "decision_on_fail": "HOLD", "issues": ["x"]},
        ]
        decision, _ = gate._decision_from_results(results)
        self.assertEqual(decision, "HOLD")


class EndToEndAgainstTempArtifacts(unittest.TestCase):
    """Run the gate against synthetic artifacts.  We place artifacts
    INSIDE the repo's ``Data/evals/_tmp`` so the gate's ``ROOT / path``
    resolution stays correct without needing to monkey-patch ROOT."""

    def setUp(self) -> None:
        from pathlib import Path as _P
        self.tmp_root = _P(__file__).resolve().parents[1] / "Data" / "evals" / "_tmp_test"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        import shutil
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(gate.ROOT)).replace("\\", "/")

    def test_passing_artifact_seeds_promote(self) -> None:
        good = self.tmp_root / "good.json"
        good.write_text(json.dumps({"status": "strong"}), encoding="utf-8")
        cfg = self.tmp_root / "cfg.yaml"
        cfg.write_text(
            "conditions:\n"
            f"  - name: c1\n"
            f"    artifact: {self._rel(good)}\n"
            "    required: true\n"
            "    critical: true\n"
            "    accepted_status: [\"strong\"]\n"
            "    decision_on_fail: \"REJECT\"\n"
            "claim_boundary: \"engineering only\"\n",
            encoding="utf-8",
        )
        report = gate.run_mle_promotion_gate(cfg)
        self.assertEqual(report["decision"], "PROMOTE")
        self.assertEqual(report["passing_count"], 1)
        self.assertEqual(report["failing_count"], 0)

    def test_missing_required_artifact_triggers_reject(self) -> None:
        cfg = self.tmp_root / "cfg.yaml"
        missing = self.tmp_root / "does_not_exist.json"
        cfg.write_text(
            "conditions:\n"
            f"  - name: missing\n"
            f"    artifact: {self._rel(missing)}\n"
            "    required: true\n"
            "    critical: true\n"
            "    accepted_status: [\"strong\"]\n"
            "    decision_on_fail: \"REJECT\"\n",
            encoding="utf-8",
        )
        report = gate.run_mle_promotion_gate(cfg)
        self.assertEqual(report["decision"], "REJECT")
        self.assertIn("missing", report["conditions"][0]["issues"])


if __name__ == "__main__":
    unittest.main()
