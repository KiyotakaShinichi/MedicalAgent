import pytest

from backend.services.cross_domain_assurance_eval import (
    build_cross_domain_assurance_eval,
)


@pytest.fixture(scope="module")
def assurance_report(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("cross-domain-assurance")
    return build_cross_domain_assurance_eval(
        output_path=directory / "assurance.json"
    )


def test_cross_domain_assurance_composes_controls_without_side_effects(
    assurance_report,
):
    report = assurance_report
    assert report["status"] == "strong_internal_assurance"
    assert report["failed_count"] == 0
    assert report["scenario_count"] >= 10
    assert report["patient_write_performed"] is False
    assert report["external_network_request_performed"] is False
    assert report["managed_cloud_operation_performed"] is False
    assert report["clinical_action_automated"] is False
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["independent_reviewer_completed"] is False


def test_cross_domain_assurance_keeps_negative_results_binding(assurance_report):
    report = assurance_report
    by_id = {row["scenario_id"]: row for row in report["scenarios"]}

    promotion = by_id["weak_evidence_cannot_promote_rag_or_synthetic_ml"]
    assert promotion["passed"] is True
    assert (
        promotion["evidence"]["rag_improvement_proven_vs_bm25"] is False
    )
    assert promotion["evidence"]["ml_promotion_decision"] == "hold_synthetic_only"

    visibility = by_id[
        "negative_results_remain_visible_on_release_surface"
    ]
    assert visibility["passed"] is True
    assert visibility["evidence"]["frozen_adversarial_warning_visible"] is True


def test_cross_domain_assurance_blocks_confirmation_tampering_and_replay(
    assurance_report,
):
    report = assurance_report
    by_id = {row["scenario_id"]: row for row in report["scenarios"]}

    substitution = by_id["confirmation_payload_substitution_fails_closed"]
    replay = by_id["consumed_confirmation_replay_fails_closed"]
    assert substitution["passed"] is True
    assert substitution["evidence"]["decision"] == "block"
    assert "confirmation_payload_mismatch" in substitution["evidence"]["issues"]
    assert replay["passed"] is True
    assert "confirmation_replayed" in replay["evidence"]["issues"]
