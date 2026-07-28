from backend.services.xai_comprehension_contract_eval import build_xai_comprehension_contract_eval, evaluate_explanation_contract


def test_complete_explanation_contract_passes():
    result = evaluate_explanation_contract(
        "This number means how many recorded items need review. It counts the available flags. "
        "Missing records can change it. It is not a diagnosis. Bring this to your care team for review."
    )
    assert result["valid"] is True


def test_authoritative_treatment_wording_fails():
    result = evaluate_explanation_contract(
        "This number means a result. It is calculated from records. Missing data matters. "
        "It is not a diagnosis. Stop chemotherapy and review with your care team."
    )
    assert result["valid"] is False
    assert result["authority_pattern_hits"]


def test_eval_is_internal_proxy_not_human_evidence(tmp_path):
    report = build_xai_comprehension_contract_eval(tmp_path / "xai.json")
    assert report["n_cases"] == 20
    assert report["passed_n"] == 20
    assert report["human_participant_study_completed"] is False
    assert report["clinical_validation"] is False
