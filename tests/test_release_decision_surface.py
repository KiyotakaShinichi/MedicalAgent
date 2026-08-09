import json
from datetime import datetime, timezone

from backend.services import release_decision_surface as surface


def test_missing_hard_blocker_blocks_engineering_release(tmp_path, monkeypatch):
    checks = ({"id": "missing", "tier": "hard_blocker", "owner": "test", "path": str(tmp_path / "missing.json"), "status_path": ("status",), "accepted": {"passed"}},)
    monkeypatch.setattr(surface, "CHECKS", checks)
    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    assert result["engineering_release_decision"] == "BLOCK"
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False
    assert result["schema_version"] == "release_decision_surface_v3"
    assert result["domain_count"] == 9


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


def test_primary_surface_is_capped_at_twenty_checks():
    assert len(surface.CHECKS) <= 20
    assert {row["domain"] for row in surface.CHECKS} >= {
        "aie",
        "mle",
        "swe",
        "data_engineering",
        "infrastructure",
        "medical",
        "automation",
        "deployment",
    }


def test_failed_full_ship_manifest_is_a_hard_blocker(tmp_path, monkeypatch):
    ship_path = tmp_path / "ship.json"
    ship_path.write_text(
        json.dumps({"status": "failed", "generated_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    checks = tuple(
        {**check, "path": str(ship_path)} if check["id"] == "full_ship_manifest" else check
        for check in surface.CHECKS
    )
    monkeypatch.setattr(surface, "CHECKS", checks)
    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    row = next(row for row in result["checks"] if row["id"] == "full_ship_manifest")
    assert result["engineering_release_decision"] == "BLOCK"
    assert row["tier"] == "hard_blocker"
    assert row["decision"] == "attention"


def test_failed_fail_closed_rag_assurance_blocks_engineering_release(
    tmp_path, monkeypatch
):
    assurance_path = tmp_path / "assurance.json"
    assurance_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    checks = tuple(
        {**check, "path": str(assurance_path)}
        if check["id"] == "fail_closed_rag_release"
        else check
        for check in surface.CHECKS
    )
    monkeypatch.setattr(surface, "CHECKS", checks)
    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    row = next(
        row for row in result["checks"] if row["id"] == "fail_closed_rag_release"
    )
    assert result["engineering_release_decision"] == "BLOCK"
    assert row["tier"] == "hard_blocker"
    assert row["decision"] == "attention"


def test_failed_restricted_staging_assurance_blocks_engineering_release(
    tmp_path, monkeypatch
):
    assurance_path = tmp_path / "staging-assurance.json"
    assurance_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    checks = tuple(
        {**check, "path": str(assurance_path)}
        if check["id"] == "restricted_synthetic_staging_boundary"
        else check
        for check in surface.CHECKS
    )
    monkeypatch.setattr(surface, "CHECKS", checks)
    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    row = next(
        row
        for row in result["checks"]
        if row["id"] == "restricted_synthetic_staging_boundary"
    )
    assert result["engineering_release_decision"] == "BLOCK"
    assert row["tier"] == "hard_blocker"
    assert row["decision"] == "attention"


def test_stale_warning_is_not_reported_as_verified(tmp_path, monkeypatch):
    warning = tmp_path / "warning.json"
    warning.write_text(
        json.dumps({"status": "acceptable", "generated_at": "2020-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(surface, "CHECKS", ({
        "id": "stale",
        "tier": "warning",
        "domain": "swe",
        "owner": "test",
        "path": str(warning),
        "status_path": ("status",),
        "accepted": {"acceptable"},
        "max_age_days": 1,
    },))

    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    row = result["checks"][0]
    assert row["decision"] == "attention"
    assert row["evidence_state"] == "stale"
    assert row["stale"] is True


def test_container_findings_are_visible_as_release_warning(tmp_path, monkeypatch):
    artifact = tmp_path / "container.json"
    artifact.write_text(
        json.dumps({
            "status": "blocked",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "deployment_decision": "BLOCK_PUBLIC_DEPLOYMENT",
            "summary": {"high_or_critical_count": 2},
            "clinical_validation": False,
            "healthcare_production_ready": False,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(surface, "CHECKS", ({
        "id": "container_security",
        "tier": "warning",
        "domain": "infrastructure",
        "owner": "security",
        "path": str(artifact),
        "status_path": ("status",),
        "accepted": {"acceptable"},
        "max_age_days": 30,
    },))

    result = surface.build_release_decision_surface(tmp_path / "surface.json")
    assert result["engineering_release_decision"] == "PROCEED_WITH_WARNINGS"
    assert result["checks"][0]["observed_status"] == "blocked"
