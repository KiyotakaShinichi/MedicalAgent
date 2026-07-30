from backend.services.finetune_hardening_assurance import (
    build_finetune_hardening_assurance,
)


def test_hardening_assurance_keeps_current_scaffold_on_hold(tmp_path):
    report = build_finetune_hardening_assurance(
        output_path=tmp_path / "report.json",
        doc_path=tmp_path / "report.md",
    )

    assert report["clinical_validation"] is False
    assert report["patient_facing_promotion_allowed"] is False
    assert report["promotion_decision"] in {"HOLD", None}
    assert report["summary"]["check_count"] >= 10
    assert report["promotion_contract"]["maximum_paired_p_value"] == 0.05
    assert report["promotion_contract"]["generation_manifest_required"] is True
    assert "not medical knowledge tuning" in report["claim_boundary"].lower()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_hardening_assurance_does_not_hide_semantic_contamination_review_gap(tmp_path):
    report = build_finetune_hardening_assurance(
        output_path=tmp_path / "report.json",
        doc_path=tmp_path / "report.md",
    )
    screen = next(
        item
        for item in report["checks"]
        if item["id"] == "semantic_similarity_screen_completed"
    )
    review = next(
        item
        for item in report["checks"]
        if item["id"] == "semantic_flags_cleared_for_candidate"
    )
    assert screen["evidence_artifact"].endswith(
        "latest_finetune_semantic_contamination.json"
    )
    assert report["summary"]["semantic_contamination_absence_proven"] is False
    if report["summary"]["semantic_unresolved_pair_count"]:
        assert review["passed"] is False
