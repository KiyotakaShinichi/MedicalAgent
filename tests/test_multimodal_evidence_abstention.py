from backend.services.multimodal_fusion import build_multimodal_assessment
from backend.services.patient_timeline_summary import build_patient_timeline_risk_summary


def test_empty_record_abstains_instead_of_assigning_neutral_score(tmp_path) -> None:
    assessment = build_multimodal_assessment(
        "EMPTY-SYNTHETIC",
        {
            "latest_labs": None,
            "lab_history": [],
            "risks": [],
            "symptoms": [],
            "radiology_summary": None,
            "synthetic_model_prediction": None,
        },
        model_predictions_csv_path=str(tmp_path / "missing.csv"),
        shap_explanations_json_path=str(tmp_path / "missing.json"),
    )

    assert assessment["overall_status"] == "insufficient_evidence"
    assert assessment["treatment_monitoring_score"] is None
    assert assessment["score_breakdown"]["abstained"] is True
    assert assessment["score_interpretation"]["scale"] == "not_displayed"
    assert "not enough" in assessment["overall_message"].lower()


def test_empty_record_timeline_copy_does_not_imply_no_warning_pattern(tmp_path) -> None:
    report = {
        "latest_labs": None,
        "lab_history": [],
        "risks": [],
        "symptoms": [],
        "treatment_effects": [],
        "timeline": [],
        "treatment_outcome": None,
        "synthetic_model_prediction": None,
    }
    report["multimodal_assessment"] = build_multimodal_assessment(
        "EMPTY-SYNTHETIC",
        report,
        model_predictions_csv_path=str(tmp_path / "missing.csv"),
        shap_explanations_json_path=str(tmp_path / "missing.json"),
    )

    summary = build_patient_timeline_risk_summary(report)

    assert "unavailable" in summary["headline"].lower()
    assert "not have enough" in summary["patient_summary"].lower()
    assert "no major combined warning pattern" not in summary["patient_summary"].lower()


def test_available_signal_still_produces_synthetic_index(tmp_path) -> None:
    assessment = build_multimodal_assessment(
        "LAB-SYNTHETIC",
        {
            "latest_labs": {"wbc": 5.0, "hemoglobin": 12.0, "platelets": 200},
            "lab_history": [{"wbc": 5.0, "hemoglobin": 12.0, "platelets": 200}],
            "risks": [],
            "symptoms": [],
            "radiology_summary": None,
            "synthetic_model_prediction": None,
        },
        model_predictions_csv_path=str(tmp_path / "missing.csv"),
        shap_explanations_json_path=str(tmp_path / "missing.json"),
    )

    assert assessment["treatment_monitoring_score"] == 50
    assert assessment["score_breakdown"]["abstained"] is False
    assert assessment["signals"]["clinical_monitoring"]["has_labs"] is True
