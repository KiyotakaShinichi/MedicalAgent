"""Tests for ``citation_context_pruner.prune``.

Lock-ins:

* Pruner preserves chunk metadata verbatim — no chunk fields are
  mutated, renamed, or stripped.
* Duplicate/sibling chunks (same parent_id) are removed after the
  first strong hit unless they add new lexical coverage.
* For refusal/safety intents, at least one boundary/policy source
  survives even when its lexical score is mediocre.
* Clinician-only / disallowed_use chunks are NOT kept (defensive —
  they should be removed by the tier filter, but the pruner is a
  belt + braces).
* The pruner is registered in the baseline comparison as the
  ``hybrid_rrf_query_rewrite_parent_child_source_tier_pruned``
  configuration.
* No goldset-specific data leaks into the pruner: the only inputs
  are (chunks, query, rewritten_query, intent, keep, refusal_route).
"""
from __future__ import annotations

import unittest

from backend.services.citation_context_pruner import prune


def _chunk(
    chunk_id: str,
    *,
    title: str = "",
    source_name: str | None = None,
    topic: str | None = None,
    parent_id: str | None = None,
    source_tier: str = "T2",
    allowed_use: str = "patient_education",
    retrieval_score: float = 0.5,
    staleness_status: str = "fresh",
    extra: dict | None = None,
) -> dict:
    out = {
        "id": chunk_id,
        "parent_id": parent_id or chunk_id,
        "title": title,
        "source_name": source_name or title,
        "topic": topic,
        "source_tier": source_tier,
        "allowed_use": allowed_use,
        "retrieval_score": retrieval_score,
        "staleness_status": staleness_status,
        "text": title,
    }
    if extra:
        out.update(extra)
    return out


class PreservesMetadata(unittest.TestCase):
    def test_kept_chunks_have_every_input_field(self) -> None:
        chunks = [
            _chunk("a", title="Chemotherapy white blood cell monitoring", retrieval_score=0.9,
                   extra={"custom_field": "do-not-strip-me"}),
            _chunk("b", title="WBC reference ranges during therapy", retrieval_score=0.6),
        ]
        kept = prune(chunks, query="what does low WBC mean during chemotherapy", keep=5)
        self.assertTrue(kept)
        self.assertEqual(kept[0]["custom_field"], "do-not-strip-me")
        for k in ("id", "parent_id", "title", "source_name", "topic",
                  "source_tier", "allowed_use", "retrieval_score",
                  "staleness_status"):
            self.assertIn(k, kept[0], k)


class RemovesDuplicatesAndIrrelevantSiblings(unittest.TestCase):
    def test_two_chunks_same_parent_only_one_survives_when_no_new_coverage(self) -> None:
        chunks = [
            _chunk("a1", parent_id="P", title="WBC monitoring during chemotherapy", retrieval_score=0.9),
            _chunk("a2", parent_id="P", title="WBC monitoring during chemotherapy", retrieval_score=0.6),
            _chunk("b",  parent_id="Q", title="Hemoglobin monitoring during chemotherapy", retrieval_score=0.5),
        ]
        kept = prune(chunks, query="WBC monitoring during chemotherapy", keep=5)
        parents = [c["parent_id"] for c in kept]
        # The duplicate-content sibling must be dropped.
        self.assertEqual(parents.count("P"), 1)

    def test_low_overlap_low_score_chunk_drops_out(self) -> None:
        chunks = [
            _chunk("a", title="WBC monitoring during chemotherapy", retrieval_score=0.9),
            _chunk("b", title="Unrelated cooking tips", retrieval_score=0.1),
        ]
        kept = prune(chunks, query="WBC monitoring during chemotherapy", keep=5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["id"], "a")


class RefusalKeepsBoundarySource(unittest.TestCase):
    def test_refusal_route_keeps_boundary_chunk_even_with_low_lex_overlap(self) -> None:
        chunks = [
            # A higher-lexical-overlap but NON-boundary chunk.
            _chunk("a", title="St John's wort herbal monograph", retrieval_score=0.8),
            # A boundary/policy chunk with mediocre overlap.
            _chunk("b", title="Supplement Safety Boundaries Policy",
                   topic="supplement-safety-boundary", retrieval_score=0.4),
        ]
        kept = prune(
            chunks,
            query="can I take St John's wort with chemo",
            intent="pharmacist_or_clinician_review",
            keep=5,
        )
        kept_titles = [c["title"] for c in kept]
        self.assertTrue(any("Boundaries" in t or "boundary" in t.lower() for t in kept_titles),
                        f"no boundary source survived: {kept_titles}")

    def test_explicit_refusal_route_flag_overrides_intent(self) -> None:
        chunks = [
            _chunk("a", title="random education chunk", retrieval_score=0.9),
            _chunk("b", title="Genetic counseling boundary policy",
                   topic="genetic-counseling", retrieval_score=0.3),
        ]
        kept = prune(
            chunks,
            query="does my VUS mean I have cancer",
            intent="education",  # caller mis-tagged intent
            refusal_route=True,  # but force refusal
            keep=5,
        )
        self.assertTrue(any("counseling" in c["title"].lower() for c in kept))


class BlocksClinicianOnly(unittest.TestCase):
    def test_clinician_only_chunk_not_kept(self) -> None:
        chunks = [
            _chunk("a", title="Patient-facing CBC explainer", retrieval_score=0.8,
                   allowed_use="patient_education"),
            _chunk("b", title="Clinician-only dose adjustment protocol",
                   allowed_use="clinician_only", retrieval_score=0.95),
        ]
        kept = prune(chunks, query="what does my CBC mean", keep=5)
        ids = [c["id"] for c in kept]
        self.assertIn("a", ids)
        self.assertNotIn("b", ids)


class GenericityInvariants(unittest.TestCase):
    def test_pruner_signature_has_no_goldset_arguments(self) -> None:
        import inspect
        sig = inspect.signature(prune)
        allowed = {"chunks", "query", "rewritten_query", "intent", "keep", "refusal_route"}
        actual = set(sig.parameters.keys())
        self.assertEqual(actual, allowed)

    def test_pruner_does_not_inspect_case_id(self) -> None:
        # Two chunks identical except case_id smuggled in.  Pruner
        # must produce the same kept set regardless of case_id values.
        a_with = _chunk("a", title="WBC monitoring", retrieval_score=0.9, extra={"case_id": "retrieval_gold_001"})
        b_with = _chunk("b", title="Hgb monitoring", retrieval_score=0.6, extra={"case_id": "retrieval_gold_001"})
        a_no   = _chunk("a", title="WBC monitoring", retrieval_score=0.9)
        b_no   = _chunk("b", title="Hgb monitoring", retrieval_score=0.6)
        kept_with = [c["id"] for c in prune([a_with, b_with], query="WBC and Hgb monitoring", keep=5)]
        kept_no   = [c["id"] for c in prune([a_no,   b_no],   query="WBC and Hgb monitoring", keep=5)]
        self.assertEqual(kept_with, kept_no)


class IntegrationWithBaselineComparison(unittest.TestCase):
    def test_pruned_configuration_is_registered(self) -> None:
        from backend.services.rag_baseline_comparison import CONFIGURATIONS
        ids = {c["id"] for c in CONFIGURATIONS}
        self.assertIn("hybrid_rrf_query_rewrite_parent_child_source_tier_pruned", ids)

    def test_pruned_configuration_label_mentions_pruner(self) -> None:
        from backend.services.rag_baseline_comparison import CONFIGURATIONS
        pruned = next(
            c for c in CONFIGURATIONS
            if c["id"] == "hybrid_rrf_query_rewrite_parent_child_source_tier_pruned"
        )
        self.assertIn("pruner", pruned["label"].lower() + " " + pruned["description"].lower())
        self.assertIn("eval-path", pruned["description"].lower())


if __name__ == "__main__":
    unittest.main()
