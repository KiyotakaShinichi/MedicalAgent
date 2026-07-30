from backend.services.finetune_runtime_preflight import build_finetune_runtime_preflight


def test_preflight_is_pinned_but_cannot_claim_training(tmp_path):
    report = build_finetune_runtime_preflight(tmp_path / "preflight.json", execute_runtime_probe=False)
    assert report["candidate_config_ready"] is True
    assert len(report["candidate"]["revision"]) == 40
    assert report["model_trained"] is False
    assert report["adapter_created"] is False
    assert report["patient_facing_promotion_allowed"] is False
    assert report["clinical_validation"] is False
    assert report["status"] == "blocked_runtime"


def test_preflight_requires_explicit_enable(tmp_path):
    report = build_finetune_runtime_preflight(tmp_path / "preflight.json", execute_runtime_probe=False)
    assert report["explicit_experiment_enable"] is False
    assert report["ready_for_offline_experiment"] is False


def test_preflight_requires_completed_contamination_adjudication(tmp_path):
    adjudication = tmp_path / "adjudication.json"
    adjudication.write_text(
        '{"status":"ready_for_human_adjudication","completed":false,'
        '"unresolved_count":3,"critical_unresolved_count":1}',
        encoding="utf-8",
    )
    report = build_finetune_runtime_preflight(
        tmp_path / "preflight.json",
        execute_runtime_probe=False,
        adjudication_path=adjudication,
    )
    gate = report["contamination_adjudication"]
    assert gate["completed"] is False
    assert gate["unresolved_count"] == 3
    assert gate["critical_unresolved_count"] == 1
    assert gate["cleared_for_runtime"] is False
    assert report["ready_for_offline_experiment"] is False
