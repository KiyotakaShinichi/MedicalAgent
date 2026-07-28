import json

from backend.services.constraint_aware_improvement_program import (
    DOMAIN_PLANS,
    build_improvement_program,
)


def test_program_covers_every_requested_engineering_domain(tmp_path):
    checks = []
    domains = []
    for domain in DOMAIN_PLANS:
        checks.append({
            "id": f"{domain}_check",
            "domain": domain,
            "tier": "warning",
            "decision": "pass",
            "evidence_state": "verified_internal",
        })
        domains.append({"domain": domain, "state": "verified_internal_only"})
    surface = tmp_path / "surface.json"
    surface.write_text(json.dumps({
        "engineering_release_decision": "PROCEED",
        "hard_blocker_count": 0,
        "checks": checks,
        "domains": domains,
    }), encoding="utf-8")

    result = build_improvement_program(
        surface_path=surface,
        output_path=tmp_path / "program.json",
        doc_path=tmp_path / "program.md",
    )

    assert result["domain_count"] == 8
    assert set(DOMAIN_PLANS) == {
        "aie",
        "mle",
        "swe",
        "data_engineering",
        "infrastructure",
        "medical",
        "automation",
        "deployment",
    }
    assert all(row["acceptance_criteria"] for row in result["domains"])
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False
    assert result["real_patient_data_used"] is False


def test_program_escalates_hard_blocker_and_keeps_external_work_separate(tmp_path):
    surface = tmp_path / "surface.json"
    surface.write_text(json.dumps({
        "engineering_release_decision": "BLOCK",
        "hard_blocker_count": 1,
        "checks": [{
            "id": "safety",
            "domain": "aie",
            "tier": "hard_blocker",
            "decision": "attention",
            "evidence_state": "needs_attention",
        }],
        "domains": [{"domain": "aie", "state": "blocked"}],
    }), encoding="utf-8")

    result = build_improvement_program(
        surface_path=surface,
        output_path=tmp_path / "program.json",
        doc_path=tmp_path / "program.md",
    )
    aie = next(row for row in result["ranked_priorities"] if row["domain"] == "aie")

    assert result["status"] == "blocked"
    assert aie["priority"] == "P0"
    assert aie["blocked_by_external_evidence"]
    assert "clinical validation" in result["things_internal_engineering_cannot_prove"][0]
