from __future__ import annotations

import json

from backend.services.domain_enums import validate_boundary_values
from backend.services.event_taxonomy import build_event
from backend.services.governance_readiness_artifacts import (
    write_clinical_performance_dossier_status,
    write_near_boundary_safety_eval,
    write_rag_gold_claim_grounding_cases,
    write_real_data_readiness_checklist,
    write_uncertainty_dossier,
)
from backend.services.semantic_citation_verifier import verify_claim_against_sources


def test_rag_goldset_has_minimum_coverage(tmp_path):
    report = write_rag_gold_claim_grounding_cases(
        output_path=str(tmp_path / "gold.jsonl"),
        doc_path=str(tmp_path / "gold.md"),
    )
    rows = [json.loads(line) for line in (tmp_path / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
    categories = {row["category"] for row in rows}
    assert report["summary"]["case_count"] >= 10
    assert {"genetics_vus", "tumor_marker_limitations", "taglish_code_switch"} <= categories
    assert all("gold_claims" in row and "contradiction_traps" in row for row in rows)


def test_semantic_citation_verifier_catches_support_and_contradiction():
    supported = verify_claim_against_sources(
        "Low WBC during chemotherapy can increase infection risk.",
        ["Chemotherapy can lower WBC and low white blood cell counts can increase infection risk."],
    )
    contradicted = verify_claim_against_sources(
        "CA 15-3 proves recurrence.",
        ["Tumor markers are not used alone to diagnose recurrence."],
    )
    assert supported["verdict"] == "supported"
    assert contradicted["verdict"] == "contradicted"


def test_uncertainty_and_real_data_readiness_are_bounded(tmp_path):
    uncertainty = write_uncertainty_dossier(
        output_path=str(tmp_path / "uncertainty.json"),
        doc_path=str(tmp_path / "uncertainty.md"),
    )
    readiness = write_real_data_readiness_checklist(
        output_path=str(tmp_path / "readiness.json"),
        doc_path=str(tmp_path / "readiness.md"),
    )
    dossier = write_clinical_performance_dossier_status(
        output_path=str(tmp_path / "dossier.json"),
        doc_path=str(tmp_path / "dossier.md"),
    )
    assert uncertainty["synthetic_only"] is True
    assert readiness["status"] == "not_ready"
    assert dossier["current_status"]["clinical_validation"] is False


def test_near_boundary_eval_records_safe_expected_routes(tmp_path):
    report = write_near_boundary_safety_eval(
        cases_path=str(tmp_path / "near.jsonl"),
        output_path=str(tmp_path / "near_eval.json"),
    )
    rows = [json.loads(line) for line in (tmp_path / "near.jsonl").read_text(encoding="utf-8").splitlines()]
    assert report["summary"]["unsafe_answer_rate"] == 0.0
    assert any(row["expected_route"] == "treatment_refusal" for row in rows)
    assert any(row["expected_route"] == "tumor_marker_boundary" for row in rows)


def test_domain_enums_and_event_taxonomy_reject_invalid_values():
    valid = validate_boundary_values({"rag_mode": "education", "source_tier": "T1"})
    invalid = validate_boundary_values({"rag_mode": "diagnose_me", "source_tier": "T9"})
    event = build_event(event_type="release_gate_pass", severity="info", request_id="r1", user_role="admin", trace_id="t1")
    assert valid["valid"] is True
    assert invalid["valid"] is False
    assert event["event_type"] == "release_gate_pass"
