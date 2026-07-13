from __future__ import annotations

import json

from backend.services.n8n_automation_templates import build_n8n_automation_templates


def test_n8n_automation_templates_write_manifest_doc_and_workflows(tmp_path):
    output = tmp_path / "n8n.json"
    doc = tmp_path / "n8n.md"
    template_dir = tmp_path / "templates"

    report = build_n8n_automation_templates(
        output_path=output,
        doc_path=doc,
        template_dir=template_dir,
    )

    assert output.exists()
    assert doc.exists()
    assert report["status"] == "ready_for_optional_import"
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["phi_allowed"] is False
    assert report["live_patient_route_enabled"] is False
    assert report["template_count"] >= 9
    for item in report["templates"]:
        assert (template_dir / f"{item['id']}.json").exists()


def test_n8n_templates_are_inactive_and_marked_nonclinical(tmp_path):
    template_dir = tmp_path / "templates"
    report = build_n8n_automation_templates(
        output_path=tmp_path / "n8n.json",
        doc_path=tmp_path / "n8n.md",
        template_dir=template_dir,
    )

    for item in report["templates"]:
        workflow = json.loads((template_dir / f"{item['id']}.json").read_text(encoding="utf-8"))
        assert workflow["active"] is False
        assert workflow["meta"]["template"] is True
        assert workflow["meta"]["clinical_validation"] is False
        assert workflow["meta"]["phi_allowed"] is False
        assert workflow["meta"]["signature_header_presence_required"] is True
        assert workflow["meta"]["receiver_hmac_verification_requires_operator_configuration"] is True
        assert any(node["type"] == "n8n-nodes-base.webhook" for node in workflow["nodes"])
        assert any(node["type"] == "n8n-nodes-base.respondToWebhook" for node in workflow["nodes"])


def test_n8n_manifest_blocks_phi_payload_fields(tmp_path):
    report = build_n8n_automation_templates(
        output_path=tmp_path / "n8n.json",
        doc_path=tmp_path / "n8n.md",
        template_dir=tmp_path / "templates",
    )

    blocked = set(report["blocked_payload_fields"])
    assert {"patient_name", "patient_id", "raw_patient_message", "full_chat_transcript", "medical_record_number"} <= blocked
    assert {"raw_prompt", "raw_response", "raw_trace", "private_chain_of_thought"} <= blocked
    assert "genetic_variant_details_for_patient_advice" in blocked
    assert "must not send PHI" in report["claim_boundary"]


def test_n8n_templates_have_expected_workflow_ids(tmp_path):
    report = build_n8n_automation_templates(
        output_path=tmp_path / "n8n.json",
        doc_path=tmp_path / "n8n.md",
        template_dir=tmp_path / "templates",
    )

    ids = {item["id"] for item in report["templates"]}
    assert {
        "release_gate_alert",
        "stale_artifact_ticket",
        "reviewer_intake_reminder",
        "eval_refresh_trigger",
        "trace_quality_digest",
        "pinecone_shadow_report",
        "external_red_team_intake",
        "dependency_security_alert",
        "deployment_health_alert",
    } <= ids
    assert any("reviewer" in item["title"].lower() for item in report["templates"])
