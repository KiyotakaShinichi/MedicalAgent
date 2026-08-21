"""Tests for the held-out RAG baseline comparison runner.

Lock-ins:

* The template file has exactly 9 rows (one per required category).
* Every template row carries the no-read protocol's required fields
  AND a placeholder marker — so the template can never be mistaken
  for a real held-out file.
* When the holdout file is absent, the runner emits
  ``completed: false`` with ``status == "ready_for_external_authoring"``.
* When the holdout file still contains placeholder rows, the runner
  refuses to complete and reports the offending case_ids.
* When the holdout file has internal-authored or tuning-tainted rows,
  the runner refuses to complete.
* When a tiny clean holdout is provided, the runner runs the real
  comparison and produces ``completed: true`` with the expected keys.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path



TEMPLATE_PATH = Path("Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl")
REQUIRED_FIELDS = {
    "case_id", "category", "query", "user_query",
    "expected_intent", "expected_answerability_status",
    "expected_refusal_or_escalation", "expected_refusal_or_insufficient_evidence",
    "expected_source_ids", "expected_allowed_use",
    "acceptable_source_tiers", "required_source_tiers",
    "contradiction_traps", "pass_criteria", "fail_criteria",
    "authored_by", "authored_date", "internal_vs_external_authored",
    "was_used_for_tuning", "case_source", "clinical_validation",
    "safety_notes",
}
REQUIRED_CATEGORIES = {
    "easy_education", "hard_contradiction", "no_evidence", "taglish",
    "genetics_vus", "tumor_marker", "supplement", "urgent_symptom",
    "source_tier_filtering",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TemplateSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _load(TEMPLATE_PATH)

    def test_nine_rows_one_per_category(self) -> None:
        self.assertEqual(len(self.rows), 9)
        cats = {r["category"] for r in self.rows}
        self.assertEqual(cats, REQUIRED_CATEGORIES)

    def test_every_row_has_required_fields(self) -> None:
        for row in self.rows:
            missing = REQUIRED_FIELDS - set(row.keys())
            self.assertFalse(missing, f"{row['case_id']} missing {missing}")

    def test_every_row_contains_placeholder_marker(self) -> None:
        # Lock-in: the template itself must never be consumable as a
        # real held-out file.  Even if a contributor renames it.
        for row in self.rows:
            text = json.dumps(row, ensure_ascii=False)
            self.assertIn(
                "PLACEHOLDER", text,
                f"{row['case_id']} has no placeholder marker — template is "
                f"dangerously close to a real case",
            )

    def test_every_row_is_marked_external_and_untuned(self) -> None:
        for row in self.rows:
            self.assertEqual(row["internal_vs_external_authored"], "external")
            self.assertFalse(row["was_used_for_tuning"])
            self.assertFalse(row["clinical_validation"])


class ReadinessWhenAbsent(unittest.TestCase):
    """If the holdout file does not exist, the runner emits a readiness artifact."""

    def test_absent_holdout_returns_readiness_false(self) -> None:
        from scripts.run_rag_holdout_baseline_comparison import run  # type: ignore
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            holdout = tmp_path / "missing.jsonl"
            out = tmp_path / "comparison.json"
            failures = tmp_path / "failures.json"
            report = run(
                holdout_path=holdout,
                comparison_output=out,
                failures_output=failures,
            )
            self.assertFalse(report["completed"])
            self.assertFalse(report["external_author_eval_completed"])
            self.assertFalse(report["clinical_validation"])
            self.assertEqual(report["status"], "ready_for_external_authoring")
            self.assertIn("holdout file not found", report["reason"])
            # Both artifacts must exist and reflect the readiness state.
            self.assertTrue(out.exists())
            self.assertTrue(failures.exists())
            failure_payload = json.loads(failures.read_text(encoding="utf-8"))
            self.assertFalse(failure_payload["completed"])


class NoFakeExternalCompletion(unittest.TestCase):
    """If the holdout file still contains placeholder rows, the runner refuses to complete."""

    def test_placeholder_rows_block_completion(self) -> None:
        from scripts.run_rag_holdout_baseline_comparison import run  # type: ignore
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            holdout = tmp_path / "holdout.jsonl"
            holdout.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in _load(TEMPLATE_PATH)
                ) + "\n",
                encoding="utf-8",
            )
            report = run(
                holdout_path=holdout,
                comparison_output=tmp_path / "comparison.json",
                failures_output=tmp_path / "failures.json",
            )
            self.assertFalse(report["completed"])
            self.assertEqual(report["status"], "ready_for_external_authoring")
            self.assertIn("placeholders", report["reason"])
            self.assertEqual(len(report["cases_with_placeholders"]), 9)

    def test_internal_authored_rows_block_completion(self) -> None:
        from scripts.run_rag_holdout_baseline_comparison import run  # type: ignore
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            holdout = tmp_path / "holdout.jsonl"
            row = _make_real_row("holdout_internal_001")
            row["internal_vs_external_authored"] = "internal"
            holdout.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = run(
                holdout_path=holdout,
                comparison_output=tmp_path / "comparison.json",
                failures_output=tmp_path / "failures.json",
            )
            self.assertFalse(report["completed"])
            self.assertIn("not marked external", report["reason"])

    def test_tuning_tainted_rows_block_completion(self) -> None:
        from scripts.run_rag_holdout_baseline_comparison import run  # type: ignore
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            holdout = tmp_path / "holdout.jsonl"
            row = _make_real_row("holdout_tuning_001")
            row["was_used_for_tuning"] = True
            holdout.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = run(
                holdout_path=holdout,
                comparison_output=tmp_path / "comparison.json",
                failures_output=tmp_path / "failures.json",
            )
            self.assertFalse(report["completed"])
            self.assertIn("was_used_for_tuning", report["reason"])


class CleanHoldoutCompletes(unittest.TestCase):
    """A clean, tiny holdout file produces a real ``completed: true`` artifact."""

    def test_tiny_clean_holdout_produces_completed_true(self) -> None:
        # Use pytest's marker mechanism only if needed; here we run
        # the real comparison.  The case content is intentionally
        # education-only and uses a canonical that the alias map
        # already supports, so the run is fast and the result is
        # interpretable.
        from scripts.run_rag_holdout_baseline_comparison import run  # type: ignore
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            holdout = tmp_path / "holdout.jsonl"
            rows = [
                _make_real_row("holdout_smoke_001", category="easy_education"),
                _make_real_row("holdout_smoke_002", category="easy_education"),
            ]
            holdout.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )
            report = run(
                holdout_path=holdout,
                comparison_output=tmp_path / "comparison.json",
                failures_output=tmp_path / "failures.json",
            )
            self.assertTrue(report["completed"])
            self.assertTrue(report["external_author_eval_completed"])
            self.assertFalse(report["clinical_validation"])
            self.assertEqual(report["total_n"], 2)
            # The comparison-derived headline keys must exist.
            for key in (
                "bm25_recall_at_10", "full_stack_recall_at_10",
                "full_stack_mrr", "full_stack_ndcg_at_10",
                "citation_precision", "claim_support_rate",
                "unsupported_context_rate", "refusal_correctness",
                "source_tier_correctness", "latency_p50_ms", "latency_p95_ms",
                "improvement_proven_vs_bm25_holdout",
            ):
                self.assertIn(key, report, key)


def _make_real_row(case_id: str, *, category: str = "easy_education") -> dict:
    """Construct a placeholder-free, external-authored, untuned case for smoke tests."""
    return {
        "case_id": case_id,
        "category": category,
        "query": "What does CBC monitoring mean during chemotherapy in general?",
        "user_query": "What does CBC monitoring mean during chemotherapy in general?",
        "expected_intent": "education",
        "expected_answerability_status": "answerable_with_citations",
        "expected_refusal_or_escalation": False,
        "expected_refusal_or_insufficient_evidence": False,
        "expected_source_ids": ["cbc-monitoring", "curated-wbc-neutropenia"],
        "expected_allowed_use": "general_patient_education",
        "acceptable_source_tiers": ["T1", "T2", "T3"],
        "required_source_tiers": ["T1", "T2", "T3"],
        "contradiction_traps": ["a single low WBC proves infection"],
        "pass_criteria": ["retrieves at least one expected canonical"],
        "fail_criteria": ["retrieval misses all expected source groups"],
        "authored_by": "external_smoke_test_reviewer",
        "authored_date": "2026-05-27",
        "internal_vs_external_authored": "external",
        "was_used_for_tuning": False,
        "case_source": "external_author_no_read_protocol_v2",
        "clinical_validation": False,
        "safety_notes": "Engineering retrieval/grounding test only. Not clinician-reviewed and not clinical validation.",
    }


if __name__ == "__main__":
    unittest.main()
