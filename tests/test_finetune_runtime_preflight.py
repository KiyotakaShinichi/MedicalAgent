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
