import unittest

import pandas as pd

from backend.processing.radiology_analysis import (
    analyze_breast_imaging_reports,
    detect_possible_metastatic_indicators,
    summarize_report,
)


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


if __name__ == "__main__":
    unittest.main()
