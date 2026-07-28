from backend.services.adversarial_v6_tuning_regression import (
    build_v6_tuning_regression,
)


def test_v6_regression_cannot_be_presented_as_heldout(tmp_path):
    payload = build_v6_tuning_regression(tmp_path / "regression.json")
    assert payload["was_used_for_tuning"] is True
    assert payload["independent_holdout_evidence"] is False
    assert payload["external_author_eval_completed"] is False
    assert payload["clinical_validation"] is False
    assert payload["historical_baseline_preserved"] is True
    assert payload["total_n"] >= 100
