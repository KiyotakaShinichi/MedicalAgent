import unittest
import tempfile
import uuid
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import ClinicalIntervention, LabResult, MedicationLog, Patient, PatientUpload, Treatment, TreatmentOutcome
from backend.processing.radiology_analysis import (
    analyze_breast_imaging_reports,
    detect_possible_metastatic_indicators,
    summarize_report,
)
from backend.services.mri_series_indexer import classify_mri_series_role
from backend.services.mri_manifest import select_model_input_series
from backend.services.mri_preprocessing import normalize_pixels
from backend.services.auth import create_demo_session, get_context_from_authorization
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
