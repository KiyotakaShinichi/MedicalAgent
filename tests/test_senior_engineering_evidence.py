import pytest

from backend.services.senior_engineering_evidence import (
    EVIDENCE_TRIANGLES,
    build_senior_engineering_evidence,
)


@pytest.fixture(scope="module")
def senior_evidence_report(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("senior-evidence")
    return build_senior_engineering_evidence(
        output_path=directory / "evidence.json",
        doc_path=directory / "evidence.md",
    )


def test_senior_evidence_is_falsifiable_and_does_not_award_seniority(
    senior_evidence_report,
):
    report = senior_evidence_report

    assert report["status"] in {
        "strong_internal_engineering_evidence",
        "provisional_pending_current_ship",
    }
    assert report["evidence_maturity"] in {
        "advanced_internal_candidate",
        "provisional_internal_candidate",
    }
    assert report["senior_title_awarded_by_artifact"] is False
    assert report["independent_reproduction_completed"] is False
    assert report["external_reviewer_completed"] is False
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert len(report["falsification_criteria"]) >= 5
    assert all(
        row["passed"]
        for row in report["architecture_fitness"]
        if row["mandatory"]
    )
    assert report["architecture_fitness_pass_rate"] == 1.0


def test_senior_evidence_has_source_test_artifact_triangles(
    senior_evidence_report,
):
    report = senior_evidence_report

    assert len(report["evidence_triangles"]) == len(EVIDENCE_TRIANGLES)
    for row in report["evidence_triangles"]:
        assert row["complete"] is True
        assert row["source"]["sha256"]
        assert row["test"]["sha256"]
        assert row["artifact"]["sha256"]
        assert row["artifact_clinical_validation"] is False


def test_senior_evidence_preserves_negative_findings(senior_evidence_report):
    report = senior_evidence_report
    negative = report["negative_results"]

    assert negative["rag_improvement_proven_vs_bm25"] is False
    assert negative["synthetic_ml_promotion_decision"] == "hold_synthetic_only"
    assert negative["cloud_deployment_completed"] is False
    assert negative["independent_review_completed"] is False
    assert negative["frozen_adversarial_v6_pass_rate"] < 1.0
