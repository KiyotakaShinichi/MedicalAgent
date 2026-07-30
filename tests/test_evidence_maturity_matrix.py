from backend.services.evidence_maturity_matrix import (
    build_evidence_maturity_matrix,
)


def test_matrix_refuses_misleading_aggregate_score(tmp_path):
    result = build_evidence_maturity_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=tmp_path / "matrix.md",
    )
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False
    assert result["scoring_policy"]["aggregate_score_emitted"] is False
    assert "overall_score" not in result
    assert len(result["dimensions"]) >= 10
    assert any(
        row["current_evidence_tier"] <= 1 for row in result["dimensions"]
    )
    assert (tmp_path / "matrix.json").exists()
    assert (tmp_path / "matrix.md").exists()


def test_architecture_budget_surfaces_hotspots_without_claiming_failure(tmp_path):
    result = build_evidence_maturity_matrix(
        output_path=tmp_path / "matrix.json",
        doc_path=tmp_path / "matrix.md",
    )
    architecture = result["architecture_maintainability"]
    assert architecture["backend_service_file_count"] > 0
    assert architecture["test_file_count"] > 0
    assert architecture["oversized_file_count"] >= 1
    assert "Do not add a new service" in architecture["policy"]
