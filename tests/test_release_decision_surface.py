import json

from backend.services import release_decision_surface as surface


def test_missing_hard_blocker_blocks_engineering_release(tmp_path, monkeypatch):
    checks = ({"id": "missing", "tier": "hard_blocker", "owner": "test", "path": str(tmp_path / "missing.json"), "status_path": ("status",), "accepted": {"passed"}},)
    monkeypatch.setattr(surface, "CHECKS", checks)
    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    assert result["engineering_release_decision"] == "BLOCK"
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False
    assert result["schema_version"] == "release_decision_surface_v2"
    assert result["domain_count"] == 7


def test_warning_does_not_hide_clean_hard_blocker(tmp_path, monkeypatch):
    good = tmp_path / "good.json"
    warning = tmp_path / "warning.json"
    good.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    warning.write_text(json.dumps({"status": "needs_attention"}), encoding="utf-8")
    monkeypatch.setattr(surface, "CHECKS", (
        {"id": "good", "tier": "hard_blocker", "owner": "test", "path": str(good), "status_path": ("status",), "accepted": {"passed"}},
        {"id": "warn", "tier": "warning", "owner": "test", "path": str(warning), "status_path": ("status",), "accepted": {"acceptable"}},
    ))
    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    assert result["engineering_release_decision"] == "PROCEED_WITH_WARNINGS"
    assert result["hard_blocker_count"] == 0
    assert result["warning_count"] == 1
    assert any(row["domain"] == "swe" for row in result["domains"])


def test_scaffold_and_external_blocker_are_not_reported_as_verified(tmp_path, monkeypatch):
    finetune = tmp_path / "finetune.json"
    external = tmp_path / "external.json"
    finetune.write_text(json.dumps({"status": "needs_attention", "model_trained": False}), encoding="utf-8")
    external.write_text(json.dumps({"status": "ready", "completed": False}), encoding="utf-8")
    monkeypatch.setattr(surface, "CHECKS", (
        {"id": "finetune_governance", "tier": "informational", "owner": "fine-tuning", "path": str(finetune), "status_path": ("status",), "accepted": set()},
        {"id": "external_review_execution", "tier": "informational", "owner": "external review", "path": str(external), "status_path": ("status",), "accepted": set()},
    ))

    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    states = {row["id"]: row["evidence_state"] for row in result["checks"]}

    assert states == {
        "finetune_governance": "scaffolded",
        "external_review_execution": "external_blocked",
    }
    assert result["engineering_release_decision"] == "PROCEED"
