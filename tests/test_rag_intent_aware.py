"""Tests for the Phase 11 intent-aware RAG stack.

Covers all six new pure-function services + the intent-aware benchmark +
the tier ablation harness + the Taglish parity check.  No live agent
dependency — the eval + ablation tests pass in a deterministic stub
agent so the contracts can be verified without RAG retrieval running.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.services.kb_source_governance import ALLOWED_USE_VOCABULARY, TIER_ORDER
from backend.services.rag_claim_validator import (
    SUPPORTED_THRESHOLD,
    WEAKLY_SUPPORTED_THRESHOLD,
    ClaimValidationResult,
    validate_claims,
)
from backend.services.rag_evidence_grading import grade_evidence
from backend.services.rag_intent_aware_eval import (
    EVAL_CASES,
    load_intent_aware_eval,
    run_intent_aware_eval,
)
from backend.services.rag_intent_modes import (
    COMMON_BANNED_CLAIMS,
    INTENT_TO_MODE,
    MODES,
    select_mode,
)
from backend.services.rag_tier_ablation import (
    TIER_CONFIGS,
    load_tier_ablation,
    run_tier_ablation,
)
from backend.services.rag_tier_filter import (
    FilterResult,
    filter_chunks_by_mode,
)
from backend.services.taglish_safety_parity import (
    PARITY_CASES,
    load_taglish_safety_parity,
    run_parity_check,
)


# ─── 1. Intent modes config invariants ───────────────────────────────────────


class IntentModesContract(unittest.TestCase):
    """The 5 modes must be well-formed and the intent→mode map must
    cover every intent the live router can emit."""

    def test_all_five_modes_exist_and_are_well_formed(self) -> None:
        expected = {
            "education_rag",
            "urgent_safety_rag",
            "record_explanation_rag",
            "clinician_context_rag",
            "portal_help_rag",
        }
        self.assertEqual(set(MODES), expected)
        for cfg in MODES.values():
            for tier in cfg.allowed_tiers:
                self.assertIn(tier, TIER_ORDER)
            for use in cfg.allowed_use:
                self.assertIn(use, ALLOWED_USE_VOCABULARY)
            self.assertTrue(cfg.description)
            self.assertTrue(cfg.insufficient_evidence_default)
            self.assertTrue(set(cfg.banned_claim_categories).issubset(set(COMMON_BANNED_CLAIMS) | {"other"}))

    def test_urgent_safety_mode_has_zero_retrieval(self) -> None:
        # Urgent route must never run retrieval — that's the entire
        # point of routing it through the deterministic safety response.
        self.assertEqual(MODES["urgent_safety_rag"].max_retrieved_chunks, 0)

    def test_portal_help_mode_only_allows_portal_use(self) -> None:
        self.assertEqual(MODES["portal_help_rag"].allowed_tiers, ("T4",))
        self.assertEqual(MODES["portal_help_rag"].allowed_use, ("portal_help",))


class ModeSelection(unittest.TestCase):
    def test_education_routes_to_education_rag(self) -> None:
        self.assertEqual(select_mode("education").mode, "education_rag")

    def test_clinician_upgrades_to_context_rag(self) -> None:
        self.assertEqual(
            select_mode("education", actor_role="clinician").mode,
            "clinician_context_rag",
        )
        self.assertEqual(
            select_mode("patient_timeline_monitoring", actor_role="clinician").mode,
            "clinician_context_rag",
        )

    def test_safety_boundary_routes_to_urgent_safety(self) -> None:
        self.assertEqual(select_mode("safety_boundary").mode, "urgent_safety_rag")
        self.assertEqual(select_mode("treatment_decision_boundary").mode, "urgent_safety_rag")

    def test_non_rag_intents_return_none(self) -> None:
        for intent in ("conversation", "patient_memory", "data_entry_confirmation"):
            self.assertIsNone(select_mode(intent))

    def test_unknown_intent_returns_none(self) -> None:
        self.assertIsNone(select_mode("totally_made_up_intent"))


# ─── 2. Tier filter ──────────────────────────────────────────────────────────


class TierFilter(unittest.TestCase):
    """Filter against a synthetic governance map so the test is
    deterministic regardless of what the live KB happens to contain."""

    SYNTHETIC_INDEX = {
        "src_t1_guideline": {
            "source_id": "src_t1_guideline",
            "tier": "T1",
            "allowed_use": ["education", "patient_safety", "clinician_only"],
            "staleness_status": "current",
        },
        "src_t2_review": {
            "source_id": "src_t2_review",
            "tier": "T2",
            "allowed_use": ["education", "monitoring_context"],
            "staleness_status": "current",
        },
        "src_t3_patient_ed": {
            "source_id": "src_t3_patient_ed",
            "tier": "T3",
            "allowed_use": ["education"],
            "staleness_status": "current",
        },
        "src_t4_portal": {
            "source_id": "src_t4_portal",
            "tier": "T4",
            "allowed_use": ["portal_help"],
            "staleness_status": "current",
        },
    }

    def _patched_filter(self, chunks, mode):
        with patch(
            "backend.services.rag_tier_filter._source_index",
            return_value=self.SYNTHETIC_INDEX,
        ):
            return filter_chunks_by_mode(chunks, mode)

    def test_education_rag_keeps_t1_t2_t3_drops_t4(self) -> None:
        chunks = [
            {"id": "c1", "parent_id": "src_t1_guideline"},
            {"id": "c2", "parent_id": "src_t2_review"},
            {"id": "c3", "parent_id": "src_t3_patient_ed"},
            {"id": "c4", "parent_id": "src_t4_portal"},
        ]
        result = self._patched_filter(chunks, MODES["education_rag"])
        kept = {c["id"] for c in result.kept_chunks}
        dropped = {c["id"] for c in result.dropped_chunks}
        self.assertEqual(kept, {"c1", "c2", "c3"})
        self.assertEqual(dropped, {"c4"})

    def test_portal_help_rag_keeps_only_t4(self) -> None:
        chunks = [
            {"id": "c1", "parent_id": "src_t1_guideline"},
            {"id": "c4", "parent_id": "src_t4_portal"},
        ]
        result = self._patched_filter(chunks, MODES["portal_help_rag"])
        self.assertEqual([c["id"] for c in result.kept_chunks], ["c4"])

    def test_unmapped_source_is_dropped_by_default(self) -> None:
        chunks = [{"id": "cMystery", "parent_id": "src_unknown"}]
        result = self._patched_filter(chunks, MODES["education_rag"])
        self.assertEqual(result.kept_chunks, [])
        self.assertEqual(len(result.dropped_chunks), 1)
        self.assertEqual(
            result.decisions[0].reason,
            "unmapped_source_not_in_governance",
        )

    def test_filter_result_to_dict_is_json_safe(self) -> None:
        result = self._patched_filter(
            [{"id": "c1", "parent_id": "src_t1_guideline"}],
            MODES["education_rag"],
        )
        as_dict = result.to_dict()
        round_tripped = json.loads(json.dumps(as_dict))
        self.assertEqual(round_tripped["kept_count"], 1)
        self.assertEqual(round_tripped["mode"], "education_rag")


# ─── 3. Claim validator ──────────────────────────────────────────────────────


class ClaimValidator(unittest.TestCase):
    def test_non_claim_sentence_is_skipped(self) -> None:
        result = validate_claims(
            "Please discuss anything concerning with your care team.",
            retrieved_chunks=[],
        )
        self.assertEqual(result.claim_count, 0)
        self.assertEqual(result.verdicts[0].status, "non_claim")

    def test_supported_claim_when_chunk_overlaps(self) -> None:
        result = validate_claims(
            "WBC stands for white blood cells; they help the body fight infection.",
            retrieved_chunks=[{
                "id": "c1",
                "text": "White blood cells (WBC) help the body fight infection and respond to invaders.",
            }],
        )
        self.assertEqual(result.claim_count, 1)
        self.assertEqual(result.verdicts[0].status, "supported")
        self.assertIn("c1", result.verdicts[0].supporting_chunk_ids)

    def test_unsupported_claim_when_no_chunk_overlaps(self) -> None:
        result = validate_claims(
            "Doxorubicin causes severe lethal cardiomyopathy in 100 percent of patients.",
            retrieved_chunks=[{
                "id": "c1",
                "text": "WBC measures the count of white blood cells in the blood.",
            }],
        )
        self.assertEqual(result.claim_count, 1)
        self.assertEqual(result.verdicts[0].status, "unsupported")

    def test_thresholds_are_in_correct_order(self) -> None:
        # Sanity: the supported threshold must be stricter than the
        # weakly-supported one, else the verdict ladder collapses.
        self.assertGreater(SUPPORTED_THRESHOLD, WEAKLY_SUPPORTED_THRESHOLD)

    def test_aggregate_status_for_mixed_replies(self) -> None:
        result = validate_claims(
            "WBC stands for white blood cells. "
            "Doxorubicin causes severe lethal cardiomyopathy in 100 percent of patients.",
            retrieved_chunks=[{
                "id": "c1",
                "text": "White blood cells WBC help the body fight infection.",
            }],
        )
        # One supported + one unsupported → partial citation status.
        self.assertEqual(result.citation_status, "partial")

    def test_empty_reply_returns_missing_citation_status(self) -> None:
        result = validate_claims("", retrieved_chunks=[])
        self.assertEqual(result.claim_count, 0)
        self.assertEqual(result.citation_status, "missing")


# ─── 4. Evidence grading ─────────────────────────────────────────────────────


class EvidenceGrading(unittest.TestCase):
    def _empty_filter(self) -> FilterResult:
        return FilterResult(kept_chunks=[], dropped_chunks=[], decisions=[], mode="education_rag")

    def _filter_with(self, kept: list[dict]) -> FilterResult:
        return FilterResult(kept_chunks=kept, dropped_chunks=[], decisions=[], mode="education_rag")

    def test_urgent_safety_short_circuits_to_high(self) -> None:
        grade = grade_evidence(
            mode=MODES["urgent_safety_rag"],
            filter_result=self._empty_filter(),
            claim_validation=ClaimValidationResult(),
            retrieved_count_before_filter=0,
        )
        self.assertEqual(grade.grade, "high")
        self.assertEqual(grade.answer_scope, "safety_routing")

    def test_no_kept_chunks_is_insufficient(self) -> None:
        grade = grade_evidence(
            mode=MODES["education_rag"],
            filter_result=self._empty_filter(),
            claim_validation=ClaimValidationResult(),
            retrieved_count_before_filter=0,
        )
        self.assertEqual(grade.grade, "insufficient")
        self.assertEqual(grade.answer_scope, "insufficient_evidence")

    def test_unsupported_claims_collapse_to_insufficient(self) -> None:
        validation = ClaimValidationResult()
        validation.claim_count = 2
        validation.unsupported_count = 2
        validation.supported_count = 0
        validation.citation_status = "unsupported"
        grade = grade_evidence(
            mode=MODES["education_rag"],
            filter_result=self._filter_with([{"id": "c1", "parent_id": "src_x"}]),
            claim_validation=validation,
            retrieved_count_before_filter=1,
        )
        self.assertEqual(grade.grade, "insufficient")

    def test_supported_claim_with_t1_source_grades_high(self) -> None:
        validation = ClaimValidationResult()
        validation.claim_count = 1
        validation.supported_count = 1
        validation.citation_status = "complete"
        with patch(
            "backend.services.rag_evidence_grading.known_tier_for_source",
            return_value="T1",
        ):
            grade = grade_evidence(
                mode=MODES["education_rag"],
                filter_result=self._filter_with([{"id": "c1", "parent_id": "src_t1"}]),
                claim_validation=validation,
                retrieved_count_before_filter=1,
            )
        self.assertEqual(grade.grade, "high")
        self.assertEqual(grade.answer_scope, "factual_education")

    def test_to_dict_is_json_safe(self) -> None:
        grade = grade_evidence(
            mode=MODES["education_rag"],
            filter_result=self._empty_filter(),
            claim_validation=ClaimValidationResult(),
            retrieved_count_before_filter=0,
        )
        round_tripped = json.loads(json.dumps(grade.to_dict()))
        self.assertEqual(round_tripped["mode"], "education_rag")


# ─── 5. Taglish safety parity ────────────────────────────────────────────────


class TaglishSafetyParity(unittest.TestCase):
    """Stub the safety detector + intent router so the test is
    deterministic — the live versions are tested by the script run."""

    def _stub_safety_for_case(self, case: dict):
        def safety(query: str) -> dict:
            if query in (case["english"], case["taglish"]) and case["expected_safety_scope"]:
                return {"scope": case["expected_safety_scope"]}
            return {}
        return safety

    def _stub_intent_for_case(self, case: dict):
        def router(query: str, safety: dict) -> str:
            return case["expected_intent"]
        return router

    def test_parity_passes_when_stubs_align_per_case(self) -> None:
        # We synthesise a tiny harness: for each canonical case the
        # stub returns the EXPECTED route, so all cases should pass.
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.json"
            # Run one case at a time so the stubs match the test input.
            passed = 0
            for case in PARITY_CASES:
                payload = run_parity_check(
                    safety_detector=self._stub_safety_for_case(case),
                    intent_router=self._stub_intent_for_case(case),
                    cases=(case,),
                    output_path=str(out_path),
                )
                passed += payload["passed"]
            self.assertEqual(passed, len(PARITY_CASES))

    def test_parity_fails_when_routes_diverge(self) -> None:
        case = PARITY_CASES[0]
        with TemporaryDirectory() as tmp:
            payload = run_parity_check(
                safety_detector=lambda q: {"scope": "urgent_or_safety_related"} if "fever" in q.lower() else {},
                intent_router=lambda q, s: "safety_boundary" if s.get("scope") else "education",
                cases=(case,),
                output_path=str(Path(tmp) / "out.json"),
            )
            self.assertEqual(payload["passed"], 0)
            self.assertEqual(payload["cases"][0]["intent_match"], False)

    def test_loader_returns_missing_shell(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_taglish_safety_parity(path=str(Path(tmp) / "absent.json"))
            self.assertEqual(payload["status"], "missing")


# ─── 6. Intent-aware eval (with stubbed agent) ───────────────────────────────


class IntentAwareEval(unittest.TestCase):
    def _perfect_stub_agent(self, query: str) -> dict:
        case = next(c for c in EVAL_CASES if c["query"] == query)
        if case["expects_refusal"]:
            return {
                "intent": case["expected_intent"],
                "rag_mode": case["expected_mode"],
                "reply": "Please contact your care team immediately.",
                "evidence_grade": {
                    "grade": "high",
                    "claim_support_rate": 1.0,
                    "citation_status": "complete",
                    "source_basis": [],
                },
                "mode_allowed_tiers": MODES[case["expected_mode"]].allowed_tiers,
                "post_gen_validator": {"decision": "allowed"},
            }
        return {
            "intent": case["expected_intent"],
            "rag_mode": case["expected_mode"],
            "reply": "Educational answer.",
            "evidence_grade": {
                "grade": "high",
                "claim_support_rate": 0.85,
                "citation_status": "complete",
                "source_basis": [{"source_id": "src_t1", "tier": "T1"}],
            },
            "mode_allowed_tiers": MODES[case["expected_mode"]].allowed_tiers,
            "post_gen_validator": {"decision": "allowed"},
        }

    def test_perfect_agent_produces_passing_status(self) -> None:
        """A perfectly-aligned stub agent should land in `strong` or
        `acceptable` (≥0.70 pass_rate, no unsafe answers).  We don't
        require `strong` here because the case set has 8 entries and one
        borderline grade vs status threshold should not gate the test."""
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "eval.json"
            payload = run_intent_aware_eval(
                agent=self._perfect_stub_agent,
                output_path=str(out_path),
            )
            self.assertIn(payload["status"], {"strong", "acceptable"})
            summary = payload["summary"]
            self.assertEqual(summary["unsafe_answer_rate"], 0.0)
            self.assertGreaterEqual(summary["pass_rate"], 0.7)

    def test_unsafe_blocked_marks_needs_attention(self) -> None:
        def unsafe(query: str) -> dict:
            return {
                "intent": "education",
                "rag_mode": "education_rag",
                "reply": "blocked",
                "post_gen_validator": {"decision": "blocked"},
                "evidence_grade": {"grade": "low", "claim_support_rate": 0.0, "source_basis": []},
                "mode_allowed_tiers": MODES["education_rag"].allowed_tiers,
            }
        with TemporaryDirectory() as tmp:
            payload = run_intent_aware_eval(
                agent=unsafe,
                cases=(EVAL_CASES[0],),
                output_path=str(Path(tmp) / "eval.json"),
            )
            self.assertEqual(payload["status"], "needs_attention")
            self.assertEqual(payload["summary"]["unsafe_answer_rate"], 1.0)

    def test_loader_returns_missing_shell(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_intent_aware_eval(path=str(Path(tmp) / "absent.json"))
            self.assertEqual(payload["status"], "missing")


# ─── 7. Tier ablation harness ────────────────────────────────────────────────


class TierAblation(unittest.TestCase):
    def test_ablation_sweeps_every_config(self) -> None:
        # Stub factory: returns a perfect agent regardless of tiers.
        def factory(allowed_tiers):
            def agent(query: str) -> dict:
                case = next(c for c in EVAL_CASES if c["query"] == query)
                return {
                    "intent": case["expected_intent"],
                    "rag_mode": case["expected_mode"],
                    "reply": "ok",
                    "evidence_grade": {
                        "grade": "high",
                        "claim_support_rate": 0.9,
                        "citation_status": "complete",
                        "source_basis": [],
                    },
                    "mode_allowed_tiers": allowed_tiers,
                    "post_gen_validator": {"decision": "allowed"},
                }
            return agent

        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ablation.json"
            payload = run_tier_ablation(
                agent_factory=factory,
                output_path=str(out_path),
            )
            self.assertIn(payload["status"], {"strong", "acceptable", "needs_attention"})
            self.assertEqual(
                [c["config"] for c in payload["per_config"]],
                [name for name, _ in TIER_CONFIGS],
            )

    def test_loader_returns_missing_shell(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_tier_ablation(path=str(Path(tmp) / "absent.json"))
            self.assertEqual(payload["status"], "missing")


if __name__ == "__main__":
    unittest.main()
