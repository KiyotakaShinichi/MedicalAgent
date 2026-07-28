from backend.services.agent_execution_policy_eval import (
    build_agent_execution_policy_eval,
)


def test_agent_execution_policy_eval_passes_without_live_writes(tmp_path) -> None:
    result = build_agent_execution_policy_eval(tmp_path / "result.json")
    assert result["status"] == "strong"
    assert result["passed_count"] == result["case_count"] == 6
    assert result["live_patient_write_performed"] is False
    assert result["clinical_authority_allowed"] is False
    assert result["clinical_validation"] is False
    assert any(
        "forbidden_medical_authority_tool_requested" in case["violations"]
        for case in result["cases"]
    )
