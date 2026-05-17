import base64
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import MedicationLog, Patient
from backend.services.ctcae_mapping import map_symptom_to_ctcae_review_hint
from backend.services.full_feature_group_ablation import run_full_feature_group_ablation
from backend.services.lab_reference_context import build_cbc_reference_context
from backend.services.medication_interactions import check_medication_interactions
from backend.services.patient_uploads import save_patient_upload
from backend.services.toxicity_model_metadata import build_toxicity_model_metadata


def test_full_feature_group_ablation_generates_required_groups(tmp_path):
    report = run_full_feature_group_ablation(output_path=str(tmp_path / "ablation.json"))

    assert report["schema_version"] == "full_feature_group_ablation_v1"
    assert report["leakage_audit"]["status"] == "passed"
    assert report["recommendation"]["recommended_use"] in {"monitor_only", "candidate_for_external_validation"}
    assert "labs_only" in report["feature_groups"]
    assert "symptoms_only" in report["feature_groups"]
    assert "imaging_only" in report["feature_groups"]
    assert "clinical_labs_imaging_biomarkers_genetics_tumor_markers" in report["feature_groups"]

    full = report["feature_groups"]["clinical_labs_imaging_biomarkers_genetics_tumor_markers"]
    assert "patient_level_auroc" in full["classification"]
    assert "ece" in full["classification"]
    assert "mae" in full["regression"]
    assert "synthetic" in report["claim_boundary"].lower()


def test_upload_text_report_returns_safe_parsed_metadata():
    db_session, patient = _temp_patient_session()
    text = "WBC 2.4 K/uL\nHemoglobin 9.8 g/dL\nPlatelets 88 K/uL\n"
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")

    upload = save_patient_upload(
        db=db_session,
        patient_id=patient.id,
        upload_type="cbc",
        file_name="cbc_result.txt",
        content_type="text/plain",
        content_base64=payload,
        notes="CBC report uploaded by patient.",
    )

    parsed = upload["parsed_report"]
    assert parsed["report_type"] == "cbc_report"
    assert parsed["extracted"]["wbc"] == 2.4
    assert parsed["needs_clinician_review"] is True
    assert "treatment_plan" in parsed["not_inferred"]
    assert upload["journey_phase"]["phase"] in {"monitoring_side_effects", "diagnosis_baseline"}


def test_upload_binary_image_requires_manual_review():
    db_session, patient = _temp_patient_session()
    payload = base64.b64encode(b"\x89PNG\r\n\x1a\nnot really an image").decode("ascii")

    upload = save_patient_upload(
        db=db_session,
        patient_id=patient.id,
        upload_type="ultrasound",
        file_name="ultrasound.png",
        content_type="image/png",
        content_base64=payload,
        notes=None,
    )

    assert upload["parsed_report"]["report_type"] == "non_text_upload"
    assert upload["parsed_report"]["requires_review"] is True
    assert "manual clinician review" in upload["parsed_report"]["claim_boundary"]


def test_medication_interaction_checker_routes_supplement_for_review():
    result = check_medication_interactions(
        "St. John's wort",
        current_medications=["Tamoxifen"],
    )

    assert result["status"] == "review_needed"
    assert result["flags"][0]["rule_id"] == "st_johns_wort_cyp_interaction"
    assert "pharmacist" in result["flags"][0]["message"].lower()
    assert "does not determine" in result["claim_boundary"]


def test_medication_interaction_checker_is_conservative_no_hit():
    result = check_medication_interactions(
        "ondansetron",
        current_medications=["paclitaxel"],
    )

    assert result["status"] == "no_specific_rule_hit"
    assert result["flags"] == []


def test_medication_endpoint_style_check_sees_existing_meds():
    db, patient = _temp_patient_session()
    db.add(MedicationLog(
        patient_id=patient.id,
        date=date(2026, 1, 1),
        medication="Tamoxifen",
        dose="20 mg",
        frequency="daily",
    ))
    db.commit()
    current = [row.medication for row in db.query(MedicationLog).filter(MedicationLog.patient_id == patient.id).all()]

    result = check_medication_interactions("St Johns wort supplement", current_medications=current)

    assert result["status"] == "review_needed"
    assert result["flags"][0]["matched_context_terms"] == ["tamoxifen"]


def test_cbc_reference_context_uses_population_defaults_without_demographics():
    context = build_cbc_reference_context(wbc=2.8, hemoglobin=11.5, platelets=130)

    assert context["schema_version"] == "cbc_reference_context_v1"
    assert context["demographics_used"]["sex"] == "not_available"
    assert context["labs"]["wbc"]["status"] == "low"
    assert context["labs"]["hemoglobin"]["range_source"] == "broad_population_default_no_demographics"
    assert "not a diagnosis" in context["claim_boundary"]


def test_cbc_reference_context_can_use_sex_adjusted_hemoglobin_range():
    female = build_cbc_reference_context(wbc=5.0, hemoglobin=12.4, platelets=220, sex="female")
    male = build_cbc_reference_context(wbc=5.0, hemoglobin=12.4, platelets=220, sex="male")

    assert female["labs"]["hemoglobin"]["status"] in {"within_population_range", "borderline"}
    assert male["labs"]["hemoglobin"]["status"] == "low"
    assert male["labs"]["hemoglobin"]["reference_range"]["low"] == 13.5


def test_ctcae_review_hint_routes_severe_or_red_flag_symptoms():
    severe = map_symptom_to_ctcae_review_hint(symptom="neuropathy", severity=8)
    fever = map_symptom_to_ctcae_review_hint(symptom="fever after chemo", severity=4)

    assert severe["urgent_review"] is True
    assert severe["ctcae_hint"] == "grade_3_or_higher_review_hint"
    assert fever["urgent_review"] is True
    assert fever["red_flag_terms"] == ["fever"]
    assert "not a clinician-assigned CTCAE grade" in fever["claim_boundary"]


def test_toxicity_model_metadata_documents_shortcut_risk(tmp_path):
    metadata = build_toxicity_model_metadata(
        model_path="Data/complete_synthetic_training/gradient_boosting_toxicity_risk_binary.joblib",
        output_path=str(tmp_path / "toxicity.metadata.json"),
    )

    assert metadata["schema_version"] == "toxicity_model_metadata_v1"
    assert metadata["task"] == "toxicity_risk_binary"
    assert metadata["recommended_use"]["current"] == "review_flag_or_deterministic_monitoring_rule"
    assert "learned clinical toxicity prediction" in metadata["recommended_use"]["not_supported"]


def _temp_patient_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    patient = Patient(id="PTEST", name="Test Patient", diagnosis="Synthetic breast cancer demo")
    db.add(patient)
    db.commit()
    return db, patient
