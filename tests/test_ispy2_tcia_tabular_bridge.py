from pathlib import Path

from backend.services.ispy2_tcia_tabular_bridge import (
    CLINICAL_PATH,
    EXPECTED_SHA256,
    MRI_PATH,
    _sha256,
    build_canonical_frame,
    run_ispy2_tcia_external_stress,
)


def test_official_input_files_match_locked_hashes():
    assert CLINICAL_PATH.exists() and MRI_PATH.exists()
    assert _sha256(CLINICAL_PATH) == EXPECTED_SHA256[CLINICAL_PATH.name]
    assert _sha256(MRI_PATH) == EXPECTED_SHA256[MRI_PATH.name]


def test_canonical_bridge_removes_raw_id_and_treatment_arm():
    frame, manifest = build_canonical_frame()
    assert manifest["clinical_row_count"] == 985
    assert manifest["mri_row_count"] == 384
    assert manifest["joined_row_count"] == 384
    assert "Patient_ID" not in frame.columns
    assert "CLINICAL-TRIAL-SUBJECT-ID" not in frame.columns
    assert "Arm" not in frame.columns
    assert frame["external_case_key"].str.fullmatch(r"[0-9a-f]{16}").all()


def test_external_stress_is_isolated_and_non_promotional(tmp_path: Path):
    report = run_ispy2_tcia_external_stress(
        canonical_path=tmp_path / "canonical.csv",
        output_path=tmp_path / "report.json",
    )
    assert report["clinical_validation"] is False
    assert report["used_for_nlcare_training"] is False
    assert report["patient_facing_allowed"] is False
    assert report["promotion_allowed"] is False
    assert report["task_boundary"]["nlcare_target_match"] is False
    assert report["source"]["treatment_arm_exported_or_used_as_feature"] is False
    assert len(report["per_seed_results"]) == 30
    assert len(report["paired_feature_deltas"]) == 6
