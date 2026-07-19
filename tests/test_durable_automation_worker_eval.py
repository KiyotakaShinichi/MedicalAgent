from backend.services.durable_automation_worker_eval import build_durable_automation_worker_eval


def test_durable_worker_eval_keeps_delivery_and_clinical_action_separate(tmp_path):
    result = build_durable_automation_worker_eval(tmp_path / "worker.json")
    assert result["status"] == "acceptable"
    assert result["control_pass_rate"] == 1.0
    assert result["delivery_receipt_is_human_acknowledgement"] is False
    assert result["clinical_action_automated"] is False
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False
