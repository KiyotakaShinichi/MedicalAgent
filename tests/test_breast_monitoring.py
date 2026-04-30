import unittest

import pandas as pd

from backend.processing.radiology_analysis import (
    analyze_breast_imaging_reports,
    detect_possible_metastatic_indicators,
    summarize_report,
)
from backend.services.mri_series_indexer import classify_mri_series_role
from backend.services.mri_manifest import select_model_input_series
from backend.services.mri_preprocessing import normalize_pixels


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


if __name__ == "__main__":
    unittest.main()
