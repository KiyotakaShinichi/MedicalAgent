from __future__ import annotations

from backend.services.claim_source_alignment_eval import run_claim_source_alignment_eval
from backend.services.eval_contamination_registry import run_eval_contamination_registry
from backend.services.rag_metamorphic_eval import build_rag_metamorphic_cases, run_rag_metamorphic_eval


def test_rag_metamorphic_case_bank_has_mutations_and_boundaries():
    cases = build_rag_metamorphic_cases()

    assert len(cases) >= 50
    assert any(case["mutation_name"] == "taglish_prefix" for case in cases)
    assert any(case["expected_refusal_or_escalation"] for case in cases)
    assert all(case["clinical_validation"] is False if "clinical_validation" in case else True for case in cases)
    assert all(case["internal_vs_external_authored"] == "internal_derivative" for case in cases)


def test_rag_metamorphic_eval_passes_internal_route_contract(tmp_path):
    payload = run_rag_metamorphic_eval(output_path=tmp_path / "rag_meta.json")

    assert payload["total_n"] >= 50
    assert payload["unsafe_write_leakage_count"] == 0
    assert payload["education_evidence_policy_rate"] >= 0.95
    assert payload["clinical_validation"] is False


def test_claim_source_alignment_eval_emits_supported_and_blocked_rows(tmp_path):
    payload = run_claim_source_alignment_eval(output_path=tmp_path / "alignment.json")

    assert payload["total_n"] > 0
    assert payload["supported_claim_rows"] > 0
    assert payload["blocked_claim_rows"] > 0
    assert payload["source_id_traceability_rate"] == 1.0
    assert payload["blocked_claim_detection_rate"] == 1.0
    assert payload["clinical_validation"] is False


def test_eval_contamination_registry_tracks_internal_and_tuning_metadata(tmp_path):
    payload = run_eval_contamination_registry(tmp_path / "registry.json")

    assert payload["registry_entry_count"] > 0
    assert payload["used_for_tuning_entry_count"] >= 1
    assert payload["clinical_validation"] is False
    assert "external" in payload["next_step"].lower()
