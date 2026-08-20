from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "Data/evals/safety/dep001b"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DIR / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def test_development_corpus_meets_size_and_split_contract() -> None:
    manifest = json.loads((DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["development_case_n"] >= 3000
    assert manifest["internal_blind_case_n"] >= 300
    assert manifest["split_exact_overlap_count"] == 0
    assert manifest["clinical_validation"] is False
    assert manifest["provenance"]["burned_external_holdout_opened_for_authoring"] is False


def test_all_intent_families_and_languages_are_represented() -> None:
    train = rows("train_safety_bank.jsonl")
    assert len({row["intent_family"] for row in train}) == 13
    assert {row["language"] for row in train} == {"english", "taglish", "filipino_heavy"}
    assert any(row["multi_turn"] for row in train)
    assert any(row["rag_conditioned"] for row in train)


def test_safe_utility_negative_controls_are_present_and_non_actionable() -> None:
    for name in (
        "train_safety_bank.jsonl",
        "validation_safety_bank.jsonl",
        "internal_test_safety_bank.jsonl",
        "internal_blind_safety_bank.jsonl",
    ):
        controls = [
            row for row in rows(name)
            if row.get("evaluation_group") == "safe_utility_negative_control"
        ]
        assert controls
        assert {row["utility_kind"] for row in controls} == {
            "conversation", "emotional_support", "memory", "symptom_capture",
            "portal_tool", "research_education", "out_of_scope",
            "monitoring_information",
        }
        assert all(row["unsafe_expected"] is False for row in controls)
        assert all(row["urgent_expected"] is False for row in controls)


def test_bare_symptom_disclosures_are_separate_safe_negative_controls() -> None:
    for name in (
        "train_safety_bank.jsonl",
        "validation_safety_bank.jsonl",
        "internal_test_safety_bank.jsonl",
        "internal_blind_safety_bank.jsonl",
    ):
        disclosures = [
            row for row in rows(name)
            if row.get("evaluation_group") == "safe_symptom_disclosure_negative_control"
        ]
        assert disclosures
        assert all(row["intent_family"] == "PERSONALIZED_INFORMATION" for row in disclosures)
        assert all(row["expected_policy_action"] == "ALLOW_WITH_BOUNDARY" for row in disclosures)
        assert all(row["unsafe_expected"] is False for row in disclosures)


def test_safe_contrastive_controls_cover_shared_medical_vocabulary() -> None:
    controls = [
        row for row in rows("train_safety_bank.jsonl")
        if row.get("evaluation_group") == "safe_contrastive_negative_control"
    ]
    assert controls
    assert {row["utility_kind"] for row in controls} == {
        "definition", "effect", "interaction", "emotion", "outside"
    }
    assert {row["language"] for row in controls} == {
        "english", "taglish", "filipino_heavy"
    }
    assert all(row["unsafe_expected"] is False for row in controls)


def test_internal_blind_is_not_marked_or_loaded_for_tuning() -> None:
    blind = rows("internal_blind_safety_bank.jsonl")
    manifest = json.loads((DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert all(row["used_for_tuning"] is False for row in blind)
    assert manifest["provenance"]["internal_blind_loaded_by_training"] is False
    training_source = (ROOT / "backend/services/dep001b_semantic_safety_training.py").read_text(encoding="utf-8")
    assert "INTERNAL_BLIND_PATH" not in training_source


def test_exact_text_is_isolated_across_all_splits() -> None:
    split_sets = []
    for name in ("train_safety_bank.jsonl", "validation_safety_bank.jsonl", "internal_test_safety_bank.jsonl", "internal_blind_safety_bank.jsonl"):
        values = {normalized_hash(row["text"]) for row in rows(name)}
        assert len(values) == len(rows(name))
        split_sets.append(values)
    for index, left in enumerate(split_sets):
        for right in split_sets[index + 1:]:
            assert left.isdisjoint(right)
