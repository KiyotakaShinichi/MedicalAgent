from backend.services.automation_fault_injection_eval import build_automation_fault_injection_eval


def test_fault_injection_executes_all_controlled_scenarios(tmp_path):
    report = build_automation_fault_injection_eval(tmp_path / "faults.json")
    assert report["scenario_count"] == 8
    assert report["passed_count"] == 8
    assert report["status"] == "strong"
    assert report["external_delivery_performed"] is False
    assert report["human_acknowledgement_proven"] is False
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False


def test_crash_retry_contract_has_stable_event_and_receiver_replay_rejection(tmp_path):
    report = build_automation_fault_injection_eval(tmp_path / "faults.json")
    row = next(item for item in report["scenarios"] if item["id"] == "crash_after_side_effect_uses_stable_event_id")
    assert row["passed"] is True
    assert row["evidence"]["event_ids"] == ["durable-job-42", "durable-job-42"]
    assert row["evidence"]["second_receiver_result"] == "replay"
