from backend.services.unsafe_intent_mutation_dev_eval import DEV_CASES, SAFE_CASES, evaluate_mutation_dev


def test_mutation_eval_is_explicitly_tuning_used(tmp_path):
    result = evaluate_mutation_dev(tmp_path / "mutation.json")
    assert result["was_used_for_tuning"] is True
    assert result["clinical_validation"] is False
    assert len(DEV_CASES) >= 22
    assert len(SAFE_CASES) >= 11


def test_mutation_eval_preserves_case_level_outputs(tmp_path):
    result = evaluate_mutation_dev(tmp_path / "mutation.json")
    assert len(result["cases"]) == result["total_n"]
    assert all("actual" in case and "observed_family" in case for case in result["cases"])
