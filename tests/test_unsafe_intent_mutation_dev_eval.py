from backend.services.unsafe_intent_mutation_dev_eval import (
    COMPOSITIONAL_MUTATIONS,
    DEV_CASES,
    MULTITURN_DEV_CASES,
    SAFE_CASES,
    evaluate_mutation_dev,
)


def test_mutation_eval_is_explicitly_tuning_used(tmp_path):
    result = evaluate_mutation_dev(tmp_path / "mutation.json")
    assert result["was_used_for_tuning"] is True
    assert result["clinical_validation"] is False
    assert len(DEV_CASES) >= 22
    assert len(COMPOSITIONAL_MUTATIONS) >= 63
    assert len(SAFE_CASES) >= 16
    assert len(MULTITURN_DEV_CASES) >= 12
    assert result["multi_turn_pass_rate"] == 1.0


def test_mutation_eval_preserves_case_level_outputs(tmp_path):
    result = evaluate_mutation_dev(tmp_path / "mutation.json")
    assert len(result["cases"]) == result["total_n"]
    assert all("actual" in case and "observed_family" in case for case in result["cases"])


def test_compositional_mutations_are_varied_and_pass_without_weakening_safe_negatives(tmp_path):
    result = evaluate_mutation_dev(tmp_path / "mutation.json")
    assert result["unsafe_mutation_n"] >= 85
    assert len(result["language_styles"]) >= 8
    assert result["pass_rate"] == 1.0
    assert result["safe_negative_pass_rate"] == 1.0
