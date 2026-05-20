"""Tests for ``retrieval_confidence.classify_retrieval_uncertainty``.

Lock-ins: each of the 6 ``answerability_status`` values must be
reachable, and the safety branch must dominate the others.
"""
from __future__ import annotations

import unittest

from backend.services.retrieval_confidence import (
    ANSWERABILITY_STATUS_VALUES,
    classify_retrieval_uncertainty,
)


def _chunks(*scores_and_tiers: tuple[float, str]) -> list[dict]:
    return [{"score": s, "source_tier": t, "text": "..."} for s, t in scores_and_tiers]


def _verdicts(supported: int = 0, contradicted: int = 0, unsupported: int = 0) -> dict:
    out: list[dict] = []
    out.extend({"status": "supported"} for _ in range(supported))
    out.extend({"status": "contradicted"} for _ in range(contradicted))
    out.extend({"status": "unsupported"} for _ in range(unsupported))
    return {"verdicts": out}


class StatusBranches(unittest.TestCase):
    def test_status_values_constant(self) -> None:
        self.assertEqual(len(ANSWERABILITY_STATUS_VALUES), 6)
        self.assertIn("answerable_with_citations", ANSWERABILITY_STATUS_VALUES)
        self.assertIn("refuse_due_to_safety", ANSWERABILITY_STATUS_VALUES)

    def test_refuse_due_to_safety_dominates(self) -> None:
        out = classify_retrieval_uncertainty(
            chunks=_chunks((0.95, "T1"), (0.9, "T1")),
            claim_envelope=_verdicts(supported=5),
            safety={"level": "high_risk", "scope": "treatment_decision_request"},
        )
        self.assertEqual(out.answerability_status, "refuse_due_to_safety")

    def test_conflicting_evidence(self) -> None:
        out = classify_retrieval_uncertainty(
            chunks=_chunks((0.8, "T1"), (0.7, "T2")),
            claim_envelope=_verdicts(supported=2, contradicted=1),
            safety={"level": "low_risk", "scope": "education_or_tracking"},
        )
        self.assertEqual(out.answerability_status, "conflicting_evidence")
        self.assertTrue(out.evidence_conflict_flag)

    def test_insufficient_evidence_when_low_retrieval(self) -> None:
        out = classify_retrieval_uncertainty(
            chunks=_chunks((0.05, "T4")),
            claim_envelope=_verdicts(unsupported=3),
            safety={"level": "low_risk", "scope": "education_or_tracking"},
        )
        self.assertEqual(out.answerability_status, "insufficient_evidence")

    def test_answerable_with_limited_context(self) -> None:
        out = classify_retrieval_uncertainty(
            chunks=_chunks((0.3, "T2"), (0.25, "T3"), (0.2, "T4")),
            claim_envelope=_verdicts(supported=2, unsupported=1),
            safety={"level": "low_risk", "scope": "education_or_tracking"},
        )
        self.assertEqual(out.answerability_status, "answerable_with_limited_context")

    def test_answerable_with_citations(self) -> None:
        out = classify_retrieval_uncertainty(
            chunks=_chunks((0.85, "T1"), (0.8, "T1"), (0.75, "T2")),
            claim_envelope=_verdicts(supported=4),
            safety={"level": "low_risk", "scope": "education_or_tracking"},
        )
        self.assertEqual(out.answerability_status, "answerable_with_citations")

    def test_clinician_review_required_branch(self) -> None:
        out = classify_retrieval_uncertainty(
            chunks=_chunks((0.7, "T1"), (0.6, "T2")),
            claim_envelope=_verdicts(supported=1, unsupported=3),
            safety={"level": "low_risk", "scope": "education_or_tracking"},
            intent="record_explanation",
        )
        self.assertEqual(out.answerability_status, "clinician_review_required")

    def test_to_dict_shape(self) -> None:
        out = classify_retrieval_uncertainty(
            chunks=_chunks((0.8, "T1")),
            claim_envelope=_verdicts(supported=2),
            safety={"level": "low_risk", "scope": "education_or_tracking"},
        )
        d = out.to_dict()
        for key in [
            "retrieval_confidence", "source_tier_confidence",
            "citation_support_confidence", "evidence_conflict_flag",
            "answerability_status", "reason",
            "top_score", "top_k_evaluated", "high_trust_chunks",
            "supported_claims", "contradicted_claims", "unsupported_claims",
            "safety_level", "safety_scope",
        ]:
            self.assertIn(key, d, key)


if __name__ == "__main__":
    unittest.main()
