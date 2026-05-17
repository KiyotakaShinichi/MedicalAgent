from pathlib import Path

from backend.services.breast_cancer_journey import infer_journey_phase
from backend.services.medical_report_parser import classify_report_type, parse_report
from backend.services.response_conformal_calibration import (
    build_response_conformal_calibration,
    conformal_adjustment,
    load_response_conformal_calibration,
)
from backend.services.robustness_stress import run_robustness_stress_suite


def test_response_conformal_calibration_artifact(tmp_path):
    output = tmp_path / "conformal.json"
    payload = build_response_conformal_calibration(output_path=str(output))
    assert payload["status"] in {"strong", "acceptable"}
    assert payload["adjusted_coverage"] >= payload["raw_coverage"]
    assert payload["qhat_normalized"] >= 0
    loaded = load_response_conformal_calibration(str(output))
    assert loaded["schema_version"] == "response_conformal_calibration_v1"


def test_conformal_adjustment_missing_is_zero(tmp_path):
    assert conformal_adjustment(str(tmp_path / "missing.json")) == 0.0


def test_robustness_stress_suite_writes_cases(tmp_path):
    output = tmp_path / "stress.json"
    payload = run_robustness_stress_suite(output_path=str(output))
    assert payload["status"] in {"strong", "acceptable"}
    assert payload["summary"]["case_count"] >= 8
    assert payload["summary"]["pass_rate"] >= 0.75
    assert output.exists()


def test_report_parser_extracts_explicit_cbc_fields():
    parsed = parse_report("CBC: WBC 3.1 Hemoglobin 10.2 Platelets 145")
    assert parsed["report_type"] == "cbc_report"
    assert parsed["extracted"]["wbc"] == 3.1
    assert "diagnosis" in parsed["not_inferred"]


def test_report_parser_extracts_biomarker_fields_without_inference():
    parsed = parse_report("Pathology: ER positive PR negative HER2 equivocal Ki-67 22%")
    assert parsed["report_type"] == "pathology_biomarker_report"
    assert parsed["extracted"]["er_status"] == "positive"
    assert parsed["extracted"]["her2_status"] == "equivocal"


def test_report_type_classification_for_genetics_and_imaging():
    assert classify_report_type("BRCA2 VUS germline panel") == "genetic_test_report"
    assert classify_report_type("CT Impression: small ascites noted") == "imaging_report"


def test_journey_phase_model_routes_imaging_and_genetics():
    imaging = infer_journey_phase("CT impression mentions ascites and lesion")
    genetics = infer_journey_phase("Family history and BRCA VUS question")
    assert imaging["phase"] == "imaging_response_review"
    assert genetics["phase"] == "genetic_counseling_readiness"
    assert "does not diagnose" in imaging["claim_boundary"].lower()
