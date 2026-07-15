from backend.services.confirmed_write_contract_eval import build_report


def test_confirmed_write_contract_eval_covers_core_write_invariants():
    report = build_report()
    assert report["case_count"] >= 50
    assert report["pass_rate"] == 1.0
    assert report["failed_n"] == 0
    case_ids = {row["case_id"] for row in report["rows"]}
    assert "patient_isolation" in case_ids
    assert "ambiguous_confirmation" in case_ids
    assert any(case_id.startswith("duplicate_") for case_id in case_ids)
    assert any(case_id.startswith("undo_") for case_id in case_ids)


def test_confirmed_write_contract_eval_is_not_clinical_or_live_generation_evidence():
    report = build_report()
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["external_author_eval_completed"] is False
    assert report["full_live_generation_n"] == 0
    assert "not natural-language coverage" in report["claim_boundary"]
