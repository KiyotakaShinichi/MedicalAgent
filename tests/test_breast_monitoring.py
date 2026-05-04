import unittest
import tempfile
import uuid
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import ClinicalIntervention, ClinicalSummaryReview, LabResult, MedicationLog, Patient, PatientUpload, SymptomReport, Treatment, TreatmentOutcome
from backend.processing.risk_engine import detect_clinical_rule_risks
from backend.processing.radiology_analysis import (
    analyze_breast_imaging_reports,
    detect_possible_metastatic_indicators,
    summarize_report,
)
from backend.services.mri_series_indexer import classify_mri_series_role
from backend.services.mri_manifest import select_model_input_series
from backend.services.mri_preprocessing import normalize_pixels
from backend.services.auth import create_demo_session, get_context_from_authorization
from backend.services.clinician_feedback import create_clinical_summary_review
from backend.services import admin_analytics
from backend.services.complete_synthetic_dataset import generate_complete_synthetic_breast_dataset
from backend.services.patient_uploads import save_patient_upload
from backend.services.synthetic_journey import generate_temporal_breast_cancer_journeys, infer_synthetic_subtype
from backend.services.breastdcedl_inspector import build_breastdcedl_manifest, inspect_breastdcedl_dataset


class FakeSeries:
    def __init__(self, role, description, instances):
        self.candidate_role = role
        self.series_description = description
        self.instance_count = instances


class BreastMonitoringNLPTests(unittest.TestCase):
    def test_negated_metastatic_text_is_not_flagged(self):
        text = "No evidence of liver, lung, or bone metastatic disease."
        self.assertEqual(detect_possible_metastatic_indicators(text), [])

    def test_new_liver_lesion_is_flagged(self):
        text = "New liver lesion concerning for metastatic disease."
        indicators = detect_possible_metastatic_indicators(text)
        self.assertTrue(any(item["site"] == "liver" for item in indicators))

    def test_breast_mri_size_decrease_is_detected(self):
        df = pd.DataFrame([
            {
                "date": "2026-01-01",
                "modality": "Breast MRI",
                "report_type": "Baseline",
                "body_site": "Breast",
                "findings": "Right breast upper outer mass measuring 4.2 cm. BI-RADS 6.",
                "impression": "Known malignancy.",
            },
            {
                "date": "2026-02-01",
                "modality": "Breast MRI",
                "report_type": "Follow-up",
                "body_site": "Breast",
                "findings": "Right breast upper outer mass decreased to 3.1 cm.",
                "impression": "Interval decrease.",
            },
        ])

        summary = analyze_breast_imaging_reports(df)

        self.assertEqual(summary["size_status"], "decreased")
        self.assertEqual(summary["latest_modality"], "Breast MRI")
        self.assertEqual(summary["latest_breast_side"], "right")

    def test_later_staging_ct_does_not_override_mri_size_trend(self):
        df = pd.DataFrame([
            {
                "date": "2026-01-01",
                "modality": "Breast MRI",
                "report_type": "Baseline",
                "body_site": "Breast",
                "findings": "Left breast mass measuring 5.0 cm.",
                "impression": "Baseline MRI.",
            },
            {
                "date": "2026-02-01",
                "modality": "Breast MRI",
                "report_type": "Follow-up",
                "body_site": "Breast",
                "findings": "Left breast mass decreased to 3.0 cm.",
                "impression": "Interval decrease.",
            },
            {
                "date": "2026-03-01",
                "modality": "CT chest/abdomen/pelvis",
                "report_type": "Staging CT",
                "body_site": "Chest abdomen pelvis",
                "findings": "No evidence of metastatic disease.",
                "impression": "No definite metastatic disease.",
            },
        ])

        summary = analyze_breast_imaging_reports(df)

        self.assertEqual(summary["size_status"], "decreased")
        self.assertEqual(summary["latest_modality"], "Breast MRI")
        self.assertEqual(summary["latest_report_modality"], "CT chest/abdomen/pelvis")

    def test_breast_terms_are_extracted(self):
        report = summarize_report({
            "date": "2026-01-01",
            "modality": "Breast MRI",
            "report_type": "Baseline",
            "body_site": "Breast",
            "findings": "Left breast upper inner quadrant multifocal enhancing mass measuring 2.4 cm. BI-RADS 6.",
            "impression": "Known malignancy.",
        })

        self.assertEqual(report["breast_side"], "left")
        self.assertIn("upper_inner", report["breast_locations"])
        self.assertIn("multifocal", report["disease_extent_terms"])
        self.assertEqual(report["bi_rads"], 6)

    def test_mri_series_roles_are_classified(self):
        self.assertEqual(classify_mri_series_role("DCE Dynamic Breast"), "dce")
        self.assertEqual(classify_mri_series_role("DWI_EPI_b0200800"), "dwi")
        self.assertEqual(classify_mri_series_role("THRIVE SENSE"), "t1w")
        self.assertEqual(classify_mri_series_role("Bloch-Siegert_FA300"), "b1")

    def test_manifest_selects_best_core_mri_series(self):
        selected = select_model_input_series([
            FakeSeries("dce", "early dynamic", 80),
            FakeSeries("dce", "DCE Dynamic", 100),
            FakeSeries("dwi", "DWI_EPI_b0200800", 30),
            FakeSeries("t1w", "THRIVE SENSE", 100),
        ])

        self.assertEqual(selected["dce"].series_description, "DCE Dynamic")
        self.assertEqual(selected["dwi"].series_description, "DWI_EPI_b0200800")
        self.assertEqual(selected["t1w"].series_description, "THRIVE SENSE")

    def test_mri_preview_normalization_returns_uint8(self):
        pixels = normalize_pixels(pd.Series([10, 20, 30]).to_numpy().reshape(1, 3))

        self.assertEqual(str(pixels.dtype), "uint8")
        self.assertEqual(int(pixels.min()), 0)
        self.assertEqual(int(pixels.max()), 255)

    def test_synthetic_subtype_inference(self):
        self.assertEqual(
            infer_synthetic_subtype("Positive", "Negative", "Not amplified"),
            "HR-positive / HER2-negative",
        )
        self.assertEqual(
            infer_synthetic_subtype("Negative", "Negative", "Amplified"),
            "HER2-positive",
        )
        self.assertEqual(
            infer_synthetic_subtype("Negative", "Negative", "Not amplified"),
            "triple-negative",
        )

    def test_breastdcedl_zip_inspector_detects_images_and_metadata(self):
        test_root = _temp_root()
        temp_dir = _make_temp_dir(test_root)
        zip_path = Path(temp_dir) / "BreastDCEDL_spy1.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("ISPY1/patient_001/image.nii.gz", "fake")
            archive.writestr("ISPY1/metadata.csv", "patient_id,pcr\n1,0\n")

        result = inspect_breastdcedl_dataset(zip_path)

        self.assertEqual(result["source_type"], "zip")
        self.assertEqual(result["image_file_count"], 1)
        self.assertEqual(result["metadata_file_count"], 1)
        self.assertEqual(result["training_readiness"], "ready_for_manifest_mapping")

    def test_breastdcedl_manifest_maps_dce_and_mask_paths(self):
        test_root = _temp_root()
        temp_dir = _make_temp_dir(test_root)
        root = Path(temp_dir) / "BreastDCEDL_spy1"
        (root / "spt1_dce").mkdir(parents=True)
        (root / "spy1_mask").mkdir(parents=True)
        pd.DataFrame([{
            "pid": "ISPY1_1001",
            "age": 40,
            "ER": 1,
            "PR": 0,
            "HR": 1,
            "HER2": 0,
            "HR_HER2_STATUS": "HRposHER2neg",
            "MRI_LD_Baseline": 50,
            "pCR": 0,
            "rcb_class": 2,
        }]).to_csv(root / "BreastDCEDL_spy1_metadata.csv", index=False)
        for acq in ("acq0", "acq1", "acq2"):
            (root / "spt1_dce" / f"ISPY1_1001_spy1_vis1_{acq}.nii.gz").write_text("fake")
        (root / "spy1_mask" / "ISPY1_1001_spy1_vis1_mask.nii.gz").write_text("fake")

        result = build_breastdcedl_manifest(
            root_path=str(root),
            output_csv_path=str(root / "manifest.csv"),
        )

        self.assertEqual(result["manifest_rows"], 1)
        self.assertEqual(result["patients_with_acq0_acq1_acq2"], 1)
        self.assertEqual(result["patients_with_masks"], 1)

    def test_temporal_synthetic_generator_creates_longitudinal_rows(self):
        db = _temp_db_session()
        try:
            result = generate_temporal_breast_cancer_journeys(db, count=1, seed=10, cycles=3)

            self.assertEqual(result["patients_created"], 1)
            self.assertEqual(db.query(Treatment).count(), 3)
            self.assertGreaterEqual(db.query(LabResult).count(), 10)
            self.assertGreaterEqual(db.query(MedicationLog).count(), 6)
        finally:
            db.close()
            db.bind.dispose()

    def test_demo_patient_session_and_upload_are_patient_scoped(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="TEST-P001", name="Test Patient", diagnosis="Breast cancer demo"))
            db.commit()

            session = create_demo_session(db, role="patient", patient_id="TEST-P001")
            context = get_context_from_authorization(db, f"Bearer {session['access_token']}")
            self.assertEqual(context.patient_id, "TEST-P001")

            upload = save_patient_upload(
                db=db,
                patient_id="TEST-P001",
                upload_type="lab_report",
                file_name="cbc.txt",
                content_type="text/plain",
                content_base64="V0JDPTUuMQ==",
                notes="test upload",
            )

            self.assertEqual(upload["patient_id"], "TEST-P001")
            self.assertEqual(db.query(PatientUpload).count(), 1)
        finally:
            db.close()
            db.bind.dispose()

    def test_complete_synthetic_dataset_exports_training_tables(self):
        db = _temp_db_session()
        output_dir = _make_temp_dir(_temp_root()) / "complete_bundle"
        try:
            result = generate_complete_synthetic_breast_dataset(
                db=db,
                count=3,
                seed=1,
                cycles=5,
                output_dir=str(output_dir),
                write_db=True,
            )

            self.assertEqual(result["patients_created"], 3)
            self.assertEqual(result["table_counts"]["treatment_sessions"], 15)
            self.assertEqual(result["table_counts"]["mri_reports"], 18)
            self.assertTrue((output_dir / "temporal_ml_rows.csv").exists())
            self.assertTrue((output_dir / "outcomes.csv").exists())
            self.assertEqual(db.query(TreatmentOutcome).count(), 3)
            self.assertGreater(db.query(ClinicalIntervention).count(), 0)
        finally:
            db.close()
            db.bind.dispose()

    def test_clinical_rules_flag_fever_after_treatment_and_low_wbc(self):
        labs = pd.DataFrame([
            {"date": pd.Timestamp("2026-01-01").date(), "wbc": 5.0, "hemoglobin": 12.0, "platelets": 210},
            {"date": pd.Timestamp("2026-01-08").date(), "wbc": 1.8, "hemoglobin": 9.5, "platelets": 90},
        ])
        symptoms = pd.DataFrame([
            {"date": pd.Timestamp("2026-01-09").date(), "symptom": "fever", "severity": 8, "notes": "reported fever"},
        ])
        treatments = pd.DataFrame([
            {"date": pd.Timestamp("2026-01-02").date(), "cycle": 1, "drug": "paclitaxel"},
        ])

        risks = detect_clinical_rule_risks(labs, symptoms, treatments)
        risk_types = {risk["type"] for risk in risks}

        self.assertIn("critical_wbc_suppression", risk_types)
        self.assertIn("fever_after_recent_chemotherapy", risk_types)
        self.assertIn("fever_with_low_wbc", risk_types)

    def test_clinician_summary_review_is_audited(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="TEST-P002", name="Review Patient", diagnosis="Breast cancer demo"))
            db.commit()

            review = create_clinical_summary_review(
                db=db,
                patient_id="TEST-P002",
                reviewer_role="clinician",
                decision="approved",
                summary_snapshot={"headline": "Review recommended"},
                clinician_notes="Agree with review flag.",
                explanation_quality_score=4,
                model_usefulness_score=3,
            )

            self.assertEqual(review["decision"], "approved")
            self.assertEqual(db.query(ClinicalSummaryReview).count(), 1)
        finally:
            db.close()
            db.bind.dispose()

    def test_admin_advanced_metrics_cover_calibration_failures_and_coverage(self):
        labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).to_numpy()
        probabilities = pd.Series([0.92, 0.08, 0.40, 0.22, 0.81, 0.15, 0.73, 0.45, 0.67, 0.30, 0.88, 0.12]).to_numpy()
        frame = pd.DataFrame({
            "patient_id": [f"TEST-{index:03d}" for index in range(len(labels))],
            "actual_label": labels,
            "probability": probabilities,
            "predicted_label": (probabilities >= 0.5).astype(int),
            "stage": ["II", "II", "III", "III"] * 3,
            "molecular_subtype": ["HER2-positive", "triple-negative"] * 6,
            "latest_mri_percent_change": [-55, -12, -8, 5, -44, -20, -35, 12, -63, -5, -24, 3],
            "latest_mri_tumor_size_cm": [1.1, 3.0, 2.9, 3.4, 1.8, 2.2, 2.0, 3.7, 0.9, 2.8, 2.1, 3.3],
            "max_symptom_severity": [2, 4, 3, 8, 2, 6, 3, 7, 2, 5, 3, 4],
            "nadir_wbc": [3.0, 2.4, 2.8, 1.7, 3.1, 2.2, 2.9, 1.8, 3.3, 2.0, 2.7, 2.5],
            "nadir_anc": [1.8, 1.3, 1.5, 0.8, 1.9, 1.2, 1.6, 0.7, 2.1, 1.1, 1.4, 1.3],
            "intervention_count": [0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 0],
        })
        training_rows = pd.DataFrame({
            "patient_id": ["A", "A", "B", "B"],
            "cycle": [1, 2, 1, 2],
            "pre_wbc": [5.0, 4.5, 6.0, 5.5],
            "pre_anc": [3.0, 2.7, 3.5, 3.1],
            "pre_hemoglobin": [12.1, 11.8, 13.0, 12.7],
            "pre_platelets": [220, 210, 250, 240],
            "nadir_wbc": [3.2, 2.8, 4.1, 3.9],
            "nadir_anc": [1.8, 1.4, 2.5, 2.2],
            "nadir_hemoglobin": [11.7, 11.2, 12.5, 12.0],
            "nadir_platelets": [170, 150, 210, 205],
            "mri_tumor_size_cm": [4.0, 3.2, 2.5, 2.1],
            "mri_percent_change_from_baseline": [0.0, -20.0, 0.0, -16.0],
            "treatment_date": ["2026-01-01", "2026-01-21", "2026-01-01", "2026-01-21"],
            "regimen": ["AC-T", "AC-T", "TCHP", "TCHP"],
            "max_symptom_severity": [3, 4, 2, 3],
            "symptom_count": [1, 2, 1, 1],
            "intervention_count": [0, 1, 0, 0],
        })

        calibration = admin_analytics._calibration_metrics(labels, probabilities)
        confidence = admin_analytics._bootstrap_confidence_intervals(labels, probabilities, resamples=25, seed=1)
        false_negatives = admin_analytics._false_negative_review(frame)
        subgroups = admin_analytics._subgroup_performance(frame)
        coverage = admin_analytics._data_coverage(training_rows)
        thresholds = admin_analytics._threshold_operating_points(labels, probabilities)
        cost_sensitive = admin_analytics._cost_sensitive_thresholds(labels, probabilities)
        decision_impact = admin_analytics._decision_impact_simulation(frame)
        mri_summary = admin_analytics._mri_derived_feature_summary(frame)

        self.assertIn("expected_calibration_error", calibration)
        self.assertGreater(len(confidence["metrics"]), 0)
        self.assertEqual(false_negatives["count"], 1)
        self.assertGreater(len(subgroups["rows"]), 0)
        self.assertIn(coverage["status"], {"failed", "unideal", "acceptable", "passed"})
        self.assertGreater(len(thresholds["rows"]), 0)
        self.assertEqual(len(cost_sensitive["policies"]), 3)
        self.assertGreater(len(decision_impact["categories"]), 0)
        self.assertEqual(mri_summary["status"], "acceptable")

def _temp_db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session()


def _temp_root():
    test_root = Path("Data/test_tmp")
    test_root.mkdir(parents=True, exist_ok=True)
    return test_root


def _make_temp_dir(root):
    path = Path(root) / f"unit_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    unittest.main()
