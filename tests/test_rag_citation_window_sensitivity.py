from __future__ import annotations

import json
from pathlib import Path

from backend.services.rag_citation_window_sensitivity import build_citation_window_sensitivity


def test_citation_window_sensitivity_is_eval_only(tmp_path: Path) -> None:
    report = build_citation_window_sensitivity(
        output_path=tmp_path / "latest_citation_window_sensitivity.json",
        doc_path=tmp_path / "citation_window_sensitivity.md",
    )

    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["live_patient_route_changed"] is False
    assert report["retrieval_ranking_changed"] is False
    assert "not clinical validation" in report["claim_boundary"].lower()


def test_citation_window_rows_cover_expected_k_values(tmp_path: Path) -> None:
    report = build_citation_window_sensitivity(
        output_path=tmp_path / "latest_citation_window_sensitivity.json",
        doc_path=tmp_path / "citation_window_sensitivity.md",
    )

    rows = report["rows"]
    assert {row["cited_context_k"] for row in rows} == {1, 2, 3, 5}
    for row in rows:
        assert 0.0 <= row["citation_precision"] <= 1.0
        assert 0.0 <= row["cited_window_support_rate"] <= 1.0
        assert "claim_support_rate" not in row


def test_citation_window_recommendation_is_not_live_promotion(tmp_path: Path) -> None:
    report = build_citation_window_sensitivity(
        output_path=tmp_path / "latest_citation_window_sensitivity.json",
        doc_path=tmp_path / "citation_window_sensitivity.md",
    )

    assert report["promotion_recommendation"] in {
        "candidate_for_live_ab_test",
        "do_not_promote_without_more_evidence",
    }
    assert report["live_patient_route_changed"] is False
    assert report["recommended_cited_context_k"] in {1, 2, 3, 5}


def test_citation_window_writes_artifact_and_doc(tmp_path: Path) -> None:
    artifact = tmp_path / "latest_citation_window_sensitivity.json"
    doc = tmp_path / "citation_window_sensitivity.md"
    report = build_citation_window_sensitivity(output_path=artifact, doc_path=doc)

    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "citation_window_sensitivity_v1"
    assert persisted["recommended_cited_context_k"] == report["recommended_cited_context_k"]
    assert "Do not present this as clinical validation" in doc.read_text(encoding="utf-8")
