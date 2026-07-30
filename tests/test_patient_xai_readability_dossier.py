from __future__ import annotations

from backend.services.patient_xai_readability_dossier import (
    CLAIM_BOUNDARY,
    REQUIRED_PATIENT_EXPLANATION_SURFACES,
    build_patient_xai_readability_dossier,
)


def test_patient_xai_dossier_is_nonclinical_and_strong(tmp_path):
    report = build_patient_xai_readability_dossier(
        output_path=tmp_path / "xai.json",
        doc_path=tmp_path / "xai.md",
    )

    assert report["status"] == "strong"
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["patient_benefit_claim"] is False
    assert report["diagnostic_authority_claim"] is False
    assert report["treatment_recommendation_claim"] is False
    assert report["failed_check_count"] == 0
    assert report["implementation_evidence"]["source_exists"] is True
    assert report["implementation_evidence"]["all_passed"] is True
    assert "does not explain clinical causality" in report["claim_boundary"]


def test_patient_xai_dossier_fails_when_ui_source_is_only_a_spec(tmp_path):
    missing = tmp_path / "missing.tsx"
    report = build_patient_xai_readability_dossier(
        output_path=tmp_path / "xai.json",
        doc_path=tmp_path / "xai.md",
        patient_kpi_source=missing,
    )

    assert report["status"] == "needs_attention"
    assert report["implementation_evidence"]["source_exists"] is False
    assert report["implementation_evidence"]["all_passed"] is False


def test_required_surfaces_cover_patient_numbers_and_removed_score():
    surface_ids = {surface["id"] for surface in REQUIRED_PATIENT_EXPLANATION_SURFACES}

    assert {
        "review_queue",
        "synthetic_model_pattern",
        "latest_cbc",
        "record_coverage",
        "old_monitoring_score_boundary",
    } <= surface_ids


def test_synthetic_model_surface_requires_missingness_and_abstention_explanation(tmp_path):
    report = build_patient_xai_readability_dossier(
        output_path=tmp_path / "xai.json",
        doc_path=tmp_path / "xai.md",
    )
    surfaces = {surface["id"]: surface for surface in report["surfaces"]}
    model_surface = surfaces["synthetic_model_pattern"]

    assert "modalities_used_and_missing" in model_surface["must_explain"]
    assert "abstention_or_low_confidence_reason" in model_surface["must_explain"]
    assert "confidence_is_not_patient_outcome_probability" in model_surface["must_explain"]


def test_weakness_visibility_carries_rag_ml_and_automation_boundaries(tmp_path):
    report = build_patient_xai_readability_dossier(
        output_path=tmp_path / "xai.json",
        doc_path=tmp_path / "xai.md",
    )
    weakness = report["weakness_visibility"]

    assert set(weakness) == {"rag", "ml", "automation"}
    assert "citation_precision" in weakness["rag"]
    assert "unsupported_context_rate" in weakness["rag"]
    assert "known_attention_items" in weakness["ml"]
    assert weakness["automation"]["real_emergency_coverage_claim"] is False


def test_doc_is_written_with_copy_rules_and_boundaries(tmp_path):
    doc = tmp_path / "xai.md"
    build_patient_xai_readability_dossier(
        output_path=tmp_path / "xai.json",
        doc_path=doc,
    )

    text = doc.read_text(encoding="utf-8")
    assert "# Patient XAI Readability Dossier" in text
    assert CLAIM_BOUNDARY in text
    assert "No clinical validation" in text
    assert "not clinical validation" in text.lower() or "no clinical validation" in text.lower()
    assert "RAG citation precision" in text
