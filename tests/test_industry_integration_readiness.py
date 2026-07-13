from __future__ import annotations

from backend.services.industry_integration_readiness import build_industry_integration_readiness


def test_industry_integration_readiness_writes_disabled_by_default_contract(tmp_path):
    output = tmp_path / "industry.json"
    doc = tmp_path / "industry.md"

    report = build_industry_integration_readiness(output_path=output, doc_path=doc)

    assert output.exists()
    assert doc.exists()
    assert report["status"] == "ready_for_optional_scaffold"
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["hipaa_compliance_claim"] is False
    assert report["live_patient_route_enabled"] is False
    assert report["phi_allowed"] is False
    assert "clinically validated" in report["claim_boundary"]


def test_n8n_plan_is_internal_workflow_only(tmp_path):
    report = build_industry_integration_readiness(
        output_path=tmp_path / "industry.json",
        doc_path=tmp_path / "industry.md",
    )
    n8n = report["integrations"]["n8n"]

    assert n8n["role"] == "internal_workflow_automation"
    assert n8n["status"] == "optional_disabled_by_default"
    assert any("release-gate" in item for item in n8n["recommended_uses"])
    assert "patient-facing clinical advice" in n8n["not_allowed_uses"]
    assert "treatment or dosage decisions" in n8n["not_allowed_uses"]
    assert any("HMAC" in item for item in n8n["security_requirements"])


def test_pinecone_plan_preserves_source_governance_and_blocks_phi(tmp_path):
    report = build_industry_integration_readiness(
        output_path=tmp_path / "industry.json",
        doc_path=tmp_path / "industry.md",
    )
    pinecone = report["integrations"]["pinecone"]

    assert pinecone["role"] == "optional_managed_vector_backend_shadow_mode"
    assert pinecone["status"] == "optional_disabled_by_default"
    assert "raw patient chat or PHI storage" in pinecone["not_allowed_uses"]
    assert "replacement of source-tier filtering" in pinecone["not_allowed_uses"]
    assert pinecone["namespace_plan"]["patient_data"] == "disallowed until compliance/security review"
    assert {"source_tier", "allowed_use", "patient_facing", "kb_fingerprint"} <= set(pinecone["metadata_contract"])


def test_industry_integration_acceptance_checks_block_medical_authority(tmp_path):
    report = build_industry_integration_readiness(
        output_path=tmp_path / "industry.json",
        doc_path=tmp_path / "industry.md",
    )

    assert "using Pinecone score as clinical confidence" in report["blocked_workflows"]
    assert "allowing n8n workflow output to bypass NLCare safety validators" in report["blocked_workflows"]
    assert any("No PHI" in item for item in report["acceptance_checks_before_live_use"])
    assert any("same source-tier" in item for item in report["acceptance_checks_before_live_use"])


def test_industry_integration_doc_mentions_official_docs_and_boundaries(tmp_path):
    doc = tmp_path / "industry.md"
    report = build_industry_integration_readiness(output_path=tmp_path / "industry.json", doc_path=doc)
    text = doc.read_text(encoding="utf-8")

    assert "n8n" in text
    assert "Pinecone" in text
    assert "Not allowed uses" in text
    for url in report["source_docs"]:
        assert url in text
