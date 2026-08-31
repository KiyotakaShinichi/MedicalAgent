"""Tests for the source-filter-drop goldset adjudication workflow.

Lock-ins:

* The packet contains **only** cases whose stage-wise oracle final
  failure stage is ``source_filter_drop``.
* Building the packet does NOT mutate
  ``Data/evals/rag/retrieval_goldset.jsonl``.
* ``clinical_validation`` is ``False`` at both the packet level and
  every item level.
* A draft item (``reviewer_decision is None``) is accepted by the
  validator.
* An invalid ``reviewer_decision`` value fails validation.
* A decision in ``DECISIONS_REQUIRING_NOTES`` without notes fails
  validation.
* A decision in ``DECISIONS_REQUIRING_REVIEWER_ROLE`` without role
  fails validation.
* The readiness artifact carries ``status: "ready_for_adjudication"``
  and ``completed: False``.
* The clinician-facing placeholder README exists; the JSONL does not.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


from backend.services.rag_goldset_adjudication import (
    ALLOWED_DECISIONS,
    GOLDSET_PATH,
    build_packet,
    build_readiness_report,
    packet_did_not_mutate_goldset,
    validate_packet,
    write_packet,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PacketSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = build_packet()

    def test_packet_contains_only_source_filter_drop_cases(self) -> None:
        # Cross-check against the oracle diagnostic: the packet's
        # case_ids must equal the diagnostic's source_filter_drop set.
        oracle = json.loads(
            Path("Data/evals/rag/latest_rag_stage_oracle_diagnostic.json").read_text(encoding="utf-8")
        )
        oracle_ids = {
            str(c.get("case_id"))
            for c in oracle.get("cases") or []
            if c.get("final_failure_stage") == "source_filter_drop"
        }
        packet_ids = {str(it["case_id"]) for it in self.packet["items"]}
        self.assertEqual(packet_ids, oracle_ids)

    def test_packet_carries_required_per_case_fields(self) -> None:
        required = {
            "case_id", "user_query", "expected_intent", "category",
            "category_tags", "expected_answerability_status",
            "expected_allowed_use", "acceptable_source_tiers",
            "expected_source_ids", "retrieved_pre_filter_source_ids",
            "dropped_expected_source_ids", "kept_post_filter_source_ids",
            "reason_source_was_dropped",
            "current_patient_facing_policy_summary",
            "adjudication_options",
            "reviewer_decision", "reviewer_role", "reviewer_notes",
            "linked_artifact", "clinical_validation",
        }
        for item in self.packet["items"]:
            missing = required - set(item.keys())
            self.assertFalse(missing, f"{item.get('case_id')} missing {missing}")

    def test_clinical_validation_false_everywhere(self) -> None:
        self.assertFalse(self.packet["clinical_validation"])
        for item in self.packet["items"]:
            self.assertFalse(item["clinical_validation"])


class PacketDoesNotMutateGoldset(unittest.TestCase):
    def test_building_packet_does_not_change_goldset_hash(self) -> None:
        before = _hash(GOLDSET_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            write_packet(Path(tmp) / "packet.json")
        after = _hash(GOLDSET_PATH)
        self.assertEqual(before, after)

    def test_packet_lock_in_check_passes_on_fresh_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet_path = write_packet(Path(tmp) / "packet.json")
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
        self.assertTrue(packet_did_not_mutate_goldset(packet))


class ValidatorRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.draft = build_packet()

    def test_draft_packet_has_no_validation_issues(self) -> None:
        issues = validate_packet(self.draft)
        self.assertEqual(issues, [], [i.to_dict() for i in issues])

    def test_invalid_decision_fails(self) -> None:
        bad = copy.deepcopy(self.draft)
        bad["items"][0]["reviewer_decision"] = "delete_source_tier_filter"
        issues = validate_packet(bad)
        self.assertTrue(any("reviewer_decision" in i.issue for i in issues))

    def test_revise_without_notes_fails(self) -> None:
        bad = copy.deepcopy(self.draft)
        bad["items"][0]["reviewer_decision"] = "revise_patient_facing_expected_sources"
        # reviewer_notes intentionally left None
        issues = validate_packet(bad)
        self.assertTrue(any("requires non-empty reviewer_notes" in i.issue for i in issues))

    def test_move_to_clinician_facing_without_role_fails(self) -> None:
        bad = copy.deepcopy(self.draft)
        bad["items"][0]["reviewer_decision"] = "move_to_clinician_facing_goldset"
        # reviewer_role intentionally None
        issues = validate_packet(bad)
        self.assertTrue(any("requires reviewer_role" in i.issue for i in issues))

    def test_split_requires_both_notes_and_role(self) -> None:
        bad = copy.deepcopy(self.draft)
        bad["items"][0]["reviewer_decision"] = "split_patient_and_clinician_cases"
        # Missing both notes and role.
        issues = validate_packet(bad)
        self.assertTrue(any("reviewer_notes" in i.issue for i in issues))
        self.assertTrue(any("reviewer_role" in i.issue for i in issues))

    def test_keep_expected_sources_does_not_require_notes_or_role(self) -> None:
        ok = copy.deepcopy(self.draft)
        ok["items"][0]["reviewer_decision"] = "keep_expected_sources"
        issues = validate_packet(ok)
        # Only this one item changed; no validation issues should fire.
        self.assertEqual(issues, [])

    def test_ambiguous_does_not_require_notes_or_role(self) -> None:
        ok = copy.deepcopy(self.draft)
        ok["items"][0]["reviewer_decision"] = "mark_ambiguous_needs_external_review"
        issues = validate_packet(ok)
        self.assertEqual(issues, [])

    def test_allowed_decisions_constant_is_what_the_docs_promise(self) -> None:
        self.assertEqual(
            ALLOWED_DECISIONS,
            frozenset({
                "keep_expected_sources",
                "revise_patient_facing_expected_sources",
                "move_to_clinician_facing_goldset",
                "split_patient_and_clinician_cases",
                "mark_ambiguous_needs_external_review",
            }),
        )

    def test_packet_clinical_validation_must_be_false(self) -> None:
        bad = copy.deepcopy(self.draft)
        bad["clinical_validation"] = True
        issues = validate_packet(bad)
        self.assertTrue(any("clinical_validation" in i.issue for i in issues))


class ReadinessArtifact(unittest.TestCase):
    def test_readiness_has_expected_state(self) -> None:
        readiness = build_readiness_report()
        self.assertEqual(readiness["status"], "ready_for_adjudication")
        self.assertFalse(readiness["completed"])
        self.assertFalse(readiness["clinical_validation"])
        self.assertIn("packet_path", readiness)
        self.assertIn("next_human_step", readiness)
        # When the packet is present, readiness must confirm the goldset
        # wasn't mutated since packet build.
        if readiness.get("packet_exists"):
            self.assertTrue(readiness["packet_goldset_unmodified"])


class ClinicianFacingPlaceholder(unittest.TestCase):
    """The placeholder README must exist; the JSONL must NOT."""

    def test_readme_present_jsonl_absent(self) -> None:
        readme = Path("Data/evals/rag/clinician_facing_retrieval_goldset.README.md")
        jsonl = Path("Data/evals/rag/clinician_facing_retrieval_goldset.jsonl")
        self.assertTrue(readme.exists(), "placeholder README is missing")
        self.assertFalse(
            jsonl.exists(),
            "clinician-facing JSONL exists but no adjudication moved cases there yet",
        )

    def test_readme_explicit_about_state(self) -> None:
        text = Path(
            "Data/evals/rag/clinician_facing_retrieval_goldset.README.md"
        ).read_text(encoding="utf-8").lower()
        self.assertIn("no clinician-facing goldset exists yet", text)
        self.assertIn("not clinical validation", text)


if __name__ == "__main__":
    unittest.main()
