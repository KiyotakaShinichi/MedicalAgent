from __future__ import annotations

from backend.services.automation_reliability_dossier import (
    AUTOMATION_CENTER_VISIBILITY_REQUIREMENTS,
    CHANNEL_MATRIX,
    CLAIM_BOUNDARY,
    REQUIRED_AUTOMATION_INVARIANTS,
    build_automation_reliability_dossier,
)


def test_automation_reliability_dossier_is_strong_and_nonclinical(tmp_path):
    report = build_automation_reliability_dossier(
        output_path=tmp_path / "automation_dossier.json",
        doc_path=tmp_path / "automation_dossier.md",
    )

    assert report["status"] == "strong"
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["hipaa_compliance_claim"] is False
    assert report["phi_allowed"] is False
    assert report["live_patient_route_enabled"] is False
    assert report["external_delivery_enabled_by_default"] is False
    assert report["real_emergency_coverage_claim"] is False
    assert report["failed_required_count"] == 0
    assert report["passed_count"] == report["check_count"]
    assert "does not prove emergency coverage" in report["claim_boundary"]


def test_required_invariants_cover_alert_delivery_lifecycle():
    invariant_ids = {item["id"] for item in REQUIRED_AUTOMATION_INVARIANTS}

    assert {
        "local_outbox_first",
        "redacted_signed_webhook",
        "inactive_import_templates",
        "test_recipient_only_high_risk_delivery",
        "delivery_receipt_not_acknowledgement",
        "retry_dead_letter_contract",
        "preview_only_schedule_plan",
        "dry_run_control_plane",
    } <= invariant_ids


def test_channel_matrix_has_email_sms_viber_and_dashboard_boundaries(tmp_path):
    report = build_automation_reliability_dossier(
        output_path=tmp_path / "automation_dossier.json",
        doc_path=tmp_path / "automation_dossier.md",
    )
    channels = {item["channel"]: item for item in report["channel_matrix"]}

    assert {"email", "sms", "viber_or_chatops", "admin_dashboard"} <= set(channels)
    assert channels["email"]["live_patient_status"] == "disabled_by_default"
    assert channels["sms"]["live_patient_status"] == "disabled_by_default"
    assert "patient identifiers" in " ".join(channels["viber_or_chatops"]["blocked_payload"]).lower()
    assert channels["admin_dashboard"]["live_patient_status"] == "local_demo_source_of_truth"


def test_blocked_payload_fields_are_carried_into_dossier(tmp_path):
    report = build_automation_reliability_dossier(
        output_path=tmp_path / "automation_dossier.json",
        doc_path=tmp_path / "automation_dossier.md",
    )

    blocked = set(report["blocked_payload_fields"])
    assert {"patient_id", "patient_name", "raw_patient_message", "full_chat_transcript"} <= blocked
    assert {"raw_prompt", "raw_response", "private_chain_of_thought"} <= blocked


def test_doc_is_written_with_negative_claims(tmp_path):
    doc = tmp_path / "automation_dossier.md"
    build_automation_reliability_dossier(
        output_path=tmp_path / "automation_dossier.json",
        doc_path=doc,
    )

    text = doc.read_text(encoding="utf-8")
    assert "# Automation Reliability Dossier" in text
    assert "No clinical validation" in text
    assert "No emergency coverage" in text
    assert CLAIM_BOUNDARY in text


def test_no_channel_allows_raw_chat_or_clinical_instructions():
    for channel in CHANNEL_MATRIX:
        joined = " ".join(channel["blocked_payload"]).lower()
        assert "raw" in joined or "clinical" in joined or "medical" in joined
        assert channel["minimum_controls"]


def test_automation_center_visibility_contract_distinguishes_delivery_from_acknowledgement(tmp_path):
    report = build_automation_reliability_dossier(
        output_path=tmp_path / "automation_dossier.json",
        doc_path=tmp_path / "automation_dossier.md",
    )
    requirements = {item["id"]: item for item in report["automation_center_visibility_requirements"]}

    assert report["automation_center_requirement_count"] == len(AUTOMATION_CENTER_VISIBILITY_REQUIREMENTS)
    assert "delivery_receipt_status" in requirements
    assert "manual_acknowledgement" in requirements["local_outbox_status"]["must_show"]
    assert "delivery_receipt_validated" in requirements["delivery_receipt_status"]["must_show"]
    assert "clinician acknowledgement" in requirements["delivery_receipt_status"]["why"]
    assert "not_emergency_service" in requirements["claim_boundary_visibility"]["must_show"]
