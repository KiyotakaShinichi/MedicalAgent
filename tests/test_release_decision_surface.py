import json

from backend.services import release_decision_surface as surface


def test_missing_hard_blocker_blocks_engineering_release(tmp_path, monkeypatch):
    checks = ({"id": "missing", "tier": "hard_blocker", "owner": "test", "path": str(tmp_path / "missing.json"), "status_path": ("status",), "accepted": {"passed"}},)
    monkeypatch.setattr(surface, "CHECKS", checks)
    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    assert result["engineering_release_decision"] == "BLOCK"
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False


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
