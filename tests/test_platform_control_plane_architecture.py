from __future__ import annotations

from backend.services.platform_control_plane_architecture import build_platform_control_plane_architecture


def _build(tmp_path):
    return build_platform_control_plane_architecture(
        output_path=tmp_path / "platform.json",
        doc_path=tmp_path / "platform.md",
    )


def test_platform_control_plane_writes_artifact_and_doc(tmp_path):
    output = tmp_path / "platform.json"
    doc = tmp_path / "platform.md"

    report = build_platform_control_plane_architecture(output_path=output, doc_path=doc)

    assert output.exists()
    assert doc.exists()
    assert report["status"] == "strong"
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["live_patient_authority_added"] is False
    assert "clinical validation" in report["claim_boundary"]


def test_agent_state_machine_blocks_direct_generation_shortcuts(tmp_path):
    report = _build(tmp_path)
    machine = report["sections"]["agent_state_machine"]

    assert machine["status"] == "contract_ready_not_live_refactor"
    assert machine["terminal_state"] == "trace_persisted"
    assert machine["live_agent_change"] is False
    assert "evidence_sufficiency_checked" in machine["states"]
    assert any(
        item["from"] == "evidence_retrieved" and item["to"] == "evidence_sufficiency_checked"
        for item in machine["transitions"]
    )
    assert any("without post_generation_validated" in item for item in machine["forbidden_transitions"])
    for transition in machine["transitions"]:
        assert {"reason", "safety_level", "policy_decision", "latency_ms", "blocked_alternatives"} <= set(
            transition["required_trace_fields"]
        )


def test_rag_control_plane_preserves_source_governance(tmp_path):
    report = _build(tmp_path)
    rag = report["sections"]["rag_control_plane"]

    assert "source_tier_allowed_use_filter" in rag["controller_steps"]
    assert "evidence_sufficiency_grading" in rag["controller_steps"]
    assert "claim_source_alignment" in rag["controller_steps"]
    assert rag["backends"]["local_faiss_bm25"] == "primary"
    assert rag["backends"]["pinecone"] == "shadow_only_disabled_by_default"
    assert "source_tier_correctness remains 1.0" in rag["promotion_rule"]
    assert rag["clinical_validation"] is False


def test_medical_policy_registry_contains_high_risk_boundaries(tmp_path):
    report = _build(tmp_path)
    registry = report["sections"]["medical_policy_registry"]
    policy_ids = {item["policy_id"] for item in registry["policies"]}

    assert {
        "diagnosis_boundary",
        "treatment_change_boundary",
        "dosage_boundary",
        "prognosis_boundary",
        "genetic_vus_boundary",
        "tumor_marker_boundary",
        "privacy_cross_patient_boundary",
        "prompt_injection_boundary",
    } <= policy_ids
    for policy in registry["policies"]:
        assert "safe_refusal" in policy["allowed_response_types"]
        assert "review_routing" in policy["allowed_response_types"]
        assert "treatment_recommendation" in policy["blocked_response_types"]
        assert "false_reassurance" in policy["blocked_response_types"]
        assert policy["requires_test_cases"] is True


def test_ml_feature_store_schema_keeps_heads_non_promotional(tmp_path):
    report = _build(tmp_path)
    schema = report["sections"]["ml_feature_store_schema"]
    safe_uses = {item["head"]: item["safe_use"] for item in schema["prediction_heads"]}

    assert schema["status"] == "versioned_schema_contract"
    assert "lineage_hash" in schema["feature_groups"]["governance"]
    assert "schema_version" in schema["feature_groups"]["governance"]
    assert safe_uses["response_classification"] == "monitor_only"
    assert safe_uses["toxicity_review_signal"] == "review_hint_only"
    assert safe_uses["tumor_marker_context"] == "context_only"
    assert "No head may influence treatment decisions" in schema["promotion_block"]
    assert schema["clinical_validation"] is False


def test_eval_ops_registry_separates_blockers_from_informational_scaffolds(tmp_path):
    report = _build(tmp_path)
    registry = report["sections"]["eval_ops_registry"]

    assert "medical claim-boundary regression" in registry["artifact_tiers"]["hard_blocker"]
    assert "retrieval improvement not proven" in registry["artifact_tiers"]["warning"]
    assert "external dataset readiness map" in registry["artifact_tiers"]["supporting"]
    assert "not-completed holdouts" in registry["artifact_tiers"]["informational"]
    assert "contamination_status" in registry["required_metadata"]
    assert "strongest_allowed_reading" in registry["required_metadata"]
    assert registry["clinical_validation"] is False


def test_trace_envelope_blocks_private_or_unredacted_fields(tmp_path):
    report = _build(tmp_path)
    trace = report["sections"]["trace_envelope_v2"]

    assert "correlation_id" in trace["required_fields"]
    assert "claim_validation" in trace["required_fields"]
    assert "post_generation_decision" in trace["required_fields"]
    assert "private_chain_of_thought" in trace["forbidden_fields"]
    assert "unredacted_phi" in trace["forbidden_fields"]
    assert "raw_patient_identifier" in trace["forbidden_fields"]
    assert trace["clinical_validation"] is False


def test_background_eval_worker_cannot_execute_clinical_actions(tmp_path):
    report = _build(tmp_path)
    worker = report["sections"]["background_eval_worker"]

    assert "run_release_gate" in worker["allowed_job_types"]
    assert "run_pinecone_shadow_dry_run" in worker["allowed_job_types"]
    assert "diagnosis" in worker["blocked_job_types"]
    assert "treatment_recommendation" in worker["blocked_job_types"]
    assert "send_phi_to_external_service" in worker["blocked_job_types"]
    assert worker["queue_contract"]["payload_redacted"] is True
    assert worker["clinical_validation"] is False


def test_integration_boundaries_keep_n8n_and_pinecone_optional(tmp_path):
    report = _build(tmp_path)
    boundaries = report["sections"]["integration_boundaries"]

    assert boundaries["n8n"]["status"] == "internal_automation_only"
    assert "patient-facing clinical advice" in boundaries["n8n"]["blocked"]
    assert boundaries["pinecone"]["status"] == "shadow_retrieval_only"
    assert "raw patient chat storage" in boundaries["pinecone"]["blocked"]
    assert boundaries["clinical_validation"] is False


def test_blocked_claims_include_healthcare_overclaims(tmp_path):
    report = _build(tmp_path)
    blocked = set(report["blocked_claims"])

    assert "clinical validation" in blocked
    assert "production healthcare readiness" in blocked
    assert "HIPAA compliance" in blocked
    assert "FHIR interoperability" in blocked
    assert "diagnostic authority" in blocked
    assert "treatment recommendation" in blocked
    assert "genetic-risk interpretation" in blocked
    assert "tumor-marker interpretation" in blocked
