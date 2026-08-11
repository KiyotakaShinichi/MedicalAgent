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
                        "reviewer_id": "reviewer-a",
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
    assert readiness["fully_adjudicated_count"] == 1
    assert built["integrity"]["source_artifact_sha256"]
    assert built["integrity"]["candidate_snapshot_sha256"]
    assert readiness["adapter_promotion_allowed"] is False


def test_decisions_require_audit_fields():
    issues = validate_adjudication_candidates(
        [{"pair_id": "p1", "decision": "not_contaminated"}]
    )
    assert "missing_reviewer_role:p1" in issues
    assert "missing_reviewer_id:p1" in issues
    assert "missing_reviewed_at:p1" in issues
    assert "missing_reviewer_notes:p1" in issues


def test_critical_pair_requires_independent_matching_second_review():
    row = {
        "pair_id": "critical",
        "severity": "critical",
        "decision": "not_contaminated",
        "reviewer_id": "reviewer-a",
        "reviewer_role": "engineer",
        "reviewed_at": "2026-08-11",
        "reviewer_notes": "Shared boundary language.",
        "secondary_decision": "contaminated",
        "secondary_reviewer_id": "reviewer-b",
        "secondary_reviewer_role": "reviewer",
        "secondary_reviewed_at": "2026-08-11",
        "secondary_reviewer_notes": "Potential memorization.",
    }
    issues = validate_adjudication_candidates([row])
    assert "critical_reviewer_disagreement:critical" in issues

    row["secondary_decision"] = "not_contaminated"
    row["secondary_reviewer_id"] = "reviewer-a"
    issues = validate_adjudication_candidates([row])
    assert "critical_reviewers_not_independent:critical" in issues
