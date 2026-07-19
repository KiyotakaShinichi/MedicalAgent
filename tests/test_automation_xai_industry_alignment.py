from __future__ import annotations

import json
from pathlib import Path

from backend.services.automation_xai_industry_alignment import build_automation_xai_industry_alignment


def test_industry_alignment_preserves_safety_boundaries(tmp_path: Path) -> None:
    report = build_automation_xai_industry_alignment(
        output_path=tmp_path / "latest_automation_xai_industry_alignment.json",
        doc_path=tmp_path / "automation_xai_industry_alignment.md",
    )

    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["automation_live_delivery_enabled"] is False
    assert report["patient_benefit_claim"] is False
    assert report["diagnostic_authority_claim"] is False
    assert report["treatment_recommendation_claim"] is False
    assert report["real_emergency_coverage_claim"] is False
    assert "not clinical validation" in report["claim_boundary"].lower()


def test_industry_alignment_has_automation_and_xai_controls(tmp_path: Path) -> None:
    report = build_automation_xai_industry_alignment(
        output_path=tmp_path / "latest_automation_xai_industry_alignment.json",
        doc_path=tmp_path / "automation_xai_industry_alignment.md",
    )

    automation_ids = {item["id"] for item in report["automation_controls"]}
    xai_ids = {item["id"] for item in report["xai_controls"]}
    assert "outbox_first_source_of_truth" in automation_ids
    assert "delivery_receipt_not_human_acknowledgement" in automation_ids
    assert "test_recipient_only_external_channels" in automation_ids
    assert "explanation_contract_per_surface" in xai_ids
    assert "non_causal_feature_contributions" in xai_ids
    assert "negative_results_visible_to_reviewers" in xai_ids


def test_industry_alignment_backlog_blocks_live_clinical_claims(tmp_path: Path) -> None:
    report = build_automation_xai_industry_alignment(
        output_path=tmp_path / "latest_automation_xai_industry_alignment.json",
        doc_path=tmp_path / "automation_xai_industry_alignment.md",
    )

    assert len(report["ranked_backlog"]) >= 5
    assert all(item["live_clinical_claim_allowed"] is False for item in report["ranked_backlog"])


def test_industry_alignment_writes_artifact_and_doc(tmp_path: Path) -> None:
    artifact = tmp_path / "latest_automation_xai_industry_alignment.json"
    doc = tmp_path / "automation_xai_industry_alignment.md"
    report = build_automation_xai_industry_alignment(output_path=artifact, doc_path=doc)

    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "automation_xai_industry_alignment_v1"
    assert persisted["automation_control_count"] == report["automation_control_count"]
    assert "Still Not Industry Ready" in doc.read_text(encoding="utf-8")
