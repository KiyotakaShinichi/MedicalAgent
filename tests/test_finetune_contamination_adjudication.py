import json

from backend.services.finetune_contamination_adjudication import (
    build_finetune_contamination_adjudication,
    validate_adjudication_candidates,
)


def test_builds_prioritized_unresolved_packet(tmp_path):
    source = tmp_path / "source.json"
    packet = tmp_path / "packet.json"
    source.write_text(
        json.dumps(
            {
                "flagged_pairs": [
                    {"pair_id": "review", "severity": "review", "max_similarity": 0.85, "channel": "user"},
                    {"pair_id": "critical", "severity": "critical", "max_similarity": 0.96, "channel": "assistant"},
                ]
            }
        ),
        encoding="utf-8",
    )
    built, readiness = build_finetune_contamination_adjudication(source, packet)
    assert built["candidates"][0]["pair_id"] == "critical"
    assert readiness["unresolved_count"] == 2
    assert readiness["adapter_promotion_allowed"] is False
    assert readiness["clinical_validation"] is False


def test_preserves_existing_human_decision(tmp_path):
    source = tmp_path / "source.json"
    packet = tmp_path / "packet.json"
    source.write_text(
        json.dumps(
            {"flagged_pairs": [{"pair_id": "p1", "severity": "review", "max_similarity": 0.85}]}
        ),
        encoding="utf-8",
    )
    packet.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "pair_id": "p1",
                        "decision": "not_contaminated",
                        "reviewer_role": "independent_engineer",
                        "reviewed_at": "2026-07-30",
                        "reviewer_notes": "Shared boundary wording only.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    built, readiness = build_finetune_contamination_adjudication(source, packet)
    assert built["candidates"][0]["decision"] == "not_contaminated"
    assert readiness["completed"] is True
    assert readiness["adapter_promotion_allowed"] is False


def test_decisions_require_audit_fields():
    issues = validate_adjudication_candidates(
        [{"pair_id": "p1", "decision": "not_contaminated"}]
    )
    assert "missing_reviewer_role:p1" in issues
    assert "missing_reviewed_at:p1" in issues
    assert "missing_reviewer_notes:p1" in issues
