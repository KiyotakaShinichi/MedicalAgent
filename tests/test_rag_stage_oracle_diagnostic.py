"""Tests for the stage-wise RAG retrieval oracle diagnostic.

Lock-ins:

* Artifact schema (top-level keys + summary keys).
* Every case has a ``final_failure_stage`` from the documented
  vocabulary.
* The diagnostic does NOT mutate the goldset on disk.
* The diagnostic does NOT call any live patient-agent generation
  path — we verify by stubbing the generation surface and checking
  it is never invoked.
* The oracle upper bound is >= the actual full-stack Recall@10.
* ``source_filter_drop`` is distinguishable from
  ``candidate_generation_failure`` (different attribution rules).
* ``query_rewrite_helped`` and ``query_rewrite_hurt`` are present on
  every case row.
* ``clinical_validation`` is false.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path



from backend.services.rag_stage_oracle_diagnostic import (
    DEFAULT_GOLDSET_PATH,
    FAILURE_STAGES,
    build_report,
)


COMPARISON_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")


def _current_full_stack_recall() -> float:
    payload = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    return float(payload["summary"]["full_stack_recall_at_10"])


# ─── Slow harness suite ─────────────────────────────────────────────────


def _live_report() -> dict:
    if not hasattr(_live_report, "_cache"):
        _live_report._cache = build_report(  # type: ignore[attr-defined]
            actual_full_stack_recall_at_10=_current_full_stack_recall()
        )
    return _live_report._cache  # type: ignore[attr-defined]


class ArtifactSchema(unittest.TestCase):
    def test_top_level_keys(self) -> None:
        r = _live_report()
        for key in (
            "schema_version", "status", "label", "clinical_validation",
            "claim_boundary", "generated_at", "goldset_path",
            "wall_time_ms", "summary", "cases", "contamination_note",
            "stage_vocabulary",
        ):
            self.assertIn(key, r, key)

    def test_summary_keys(self) -> None:
        s = _live_report()["summary"]
        for key in (
            "total_n", "corpus_coverage_rate",
            "bm25_candidate_recall_at_50", "dense_candidate_recall_at_50",
            "hybrid_candidate_recall_at_50",
            "source_filter_retention_rate", "citation_window_retention_rate",
            "oracle_recall_at_10_upper_bound", "actual_full_stack_recall_at_10",
            "oracle_gap",
            "failure_stage_counts", "category_failure_counts", "intent_failure_counts",
            "taglish_failure_counts", "genetics_vus_failure_counts",
            "tumor_marker_failure_counts", "supplement_failure_counts",
            "urgent_symptom_failure_counts", "source_tier_filtering_failure_counts",
        ):
            self.assertIn(key, s, key)

    def test_status_and_clinical_validation(self) -> None:
        r = _live_report()
        self.assertEqual(r["status"], "informational")
        self.assertFalse(r["clinical_validation"])
        # The disclaimer must explicitly preserve the no-clinical-validation
        # claim — anti-overclaim invariant.
        self.assertIn(
            "no clinical", r["claim_boundary"].lower(),
        )

    def test_stage_vocabulary_matches_module(self) -> None:
        self.assertEqual(_live_report()["stage_vocabulary"], list(FAILURE_STAGES))


class EveryCaseClassified(unittest.TestCase):
    def test_each_case_has_final_failure_stage_in_vocabulary(self) -> None:
        r = _live_report()
        vocab = set(FAILURE_STAGES)
        for case in r["cases"]:
            self.assertIn(case["final_failure_stage"], vocab, case.get("case_id"))

    def test_each_case_has_query_rewrite_flags(self) -> None:
        for case in _live_report()["cases"]:
            self.assertIn("query_rewrite_helped", case)
            self.assertIn("query_rewrite_hurt", case)
            self.assertIsInstance(case["query_rewrite_helped"], bool)
            self.assertIsInstance(case["query_rewrite_hurt"], bool)

    def test_each_case_has_parent_child_flags(self) -> None:
        for case in _live_report()["cases"]:
            self.assertIn("parent_child_helped", case)
            self.assertIn("parent_child_hurt", case)


class OracleUpperBound(unittest.TestCase):
    def test_oracle_upper_bound_geq_actual_full_stack_recall(self) -> None:
        s = _live_report()["summary"]
        # The oracle is the best possible Recall@10 if the post-filter
        # window could be reranked perfectly.  It must always be at
        # least as large as the actual full-stack number.
        self.assertGreaterEqual(
            s["oracle_recall_at_10_upper_bound"],
            s["actual_full_stack_recall_at_10"],
        )

    def test_oracle_gap_non_negative_when_actual_present(self) -> None:
        s = _live_report()["summary"]
        if s["oracle_gap"] is not None:
            self.assertGreaterEqual(s["oracle_gap"], 0.0)


class DistinguishesSourceFilterFromCandidateMiss(unittest.TestCase):
    def test_source_filter_drop_implies_candidate_pool_had_expected(self) -> None:
        # Every "source_filter_drop" case must have had the expected
        # source in at least one of bm25/dense/hybrid top-50; otherwise
        # the right attribution would be "candidate_generation_failure".
        for case in _live_report()["cases"]:
            if case["final_failure_stage"] != "source_filter_drop":
                continue
            self.assertTrue(
                case["bm25_top_50"] or case["dense_top_50"] or case["hybrid_rrf_top_50"],
                f"{case['case_id']} marked source_filter_drop but no candidate pool had expected",
            )

    def test_candidate_failure_implies_corpus_had_expected(self) -> None:
        for case in _live_report()["cases"]:
            if case["final_failure_stage"] != "candidate_generation_failure":
                continue
            self.assertTrue(
                case["corpus_has_expected_source"],
                f"{case['case_id']} candidate_generation_failure but corpus_has_expected_source is False",
            )


class DoesNotMutateGoldset(unittest.TestCase):
    def test_goldset_file_is_unchanged_after_report(self) -> None:
        before = _digest(DEFAULT_GOLDSET_PATH)
        # Force a fresh build (don't share the cache).
        build_report(actual_full_stack_recall_at_10=_current_full_stack_recall())
        after = _digest(DEFAULT_GOLDSET_PATH)
        self.assertEqual(before, after, "goldset hash changed after diagnostic ran")


class DoesNotInvokeLiveAgent(unittest.TestCase):
    """The diagnostic must not exercise any live-agent generation path.

    We stub the live agent's run_patient_agent_pipeline and confirm
    it is never called during a build_report run.
    """

    def test_run_patient_agent_pipeline_not_called(self) -> None:
        from backend.services import agent_rag

        original = getattr(agent_rag, "run_patient_agent_pipeline", None)
        called = []

        def _tripwire(*args, **kwargs):  # pragma: no cover - reached on failure only
            called.append((args, kwargs))
            raise RuntimeError(
                "diagnostic must not invoke run_patient_agent_pipeline",
            )

        if original is not None:
            agent_rag.run_patient_agent_pipeline = _tripwire
        try:
            build_report(actual_full_stack_recall_at_10=_current_full_stack_recall())
        finally:
            if original is not None:
                agent_rag.run_patient_agent_pipeline = original
        self.assertEqual(called, [])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
