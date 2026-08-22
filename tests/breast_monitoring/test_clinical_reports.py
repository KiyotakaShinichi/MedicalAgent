"""Clinical reporting, monitoring, data-quality, and evaluation contracts."""

import json
from datetime import date
from pathlib import Path
import pandas as pd
from backend.models import AppEventLog, ClinicalSummaryReview, ImagingReport, LabResult, Patient, PredictionAuditLog, Treatment
from backend.processing.risk_engine import detect_clinical_rule_risks
from backend.services.clinician_feedback import create_clinical_summary_review
from backend.services import admin_analytics
from backend.services.detailed_training_report import generate_detailed_training_report
from backend.services.evaluation_reports import generate_versioned_evaluation_report
from backend.services.app_logging import build_app_monitoring_summary, log_app_event
from backend.services.data_availability import build_data_availability
from backend.services.demo_patient_sync import sync_demo_patient_journey
from backend.services.input_validation import validate_cbc_values, validate_symptom_payload
from backend.services.model_artifacts import register_complete_synthetic_champion
from backend.services.complete_synthetic_xai import load_complete_synthetic_patient_prediction
from backend.services.mri_derived_features import build_mri_derived_feature_summary
from backend.services.patient_data_quality import audit_patient_data_coherence
from backend.services.qin_treatment_sync import sync_qin_treatment_cycles

from tests.breast_monitoring.support import (
    _make_temp_dir,
    _temp_db_session,
    _temp_root,
)


class ClinicalReportsTestsMixin:
    def test_hybrid_mle_signal_combines_classification_and_regression(self):
        test_dir = _make_temp_dir(_temp_root())
        predictions_path = test_dir / "classification.csv"
        regression_path = test_dir / "regression.csv"
        metrics_path = test_dir / "metrics.json"
        pd.DataFrame([{
            "patient_id": "HYB-P001",
            "actual_label": 1,
            "gradient_boosting_probability": 0.8,
        }]).to_csv(predictions_path, index=False)
        pd.DataFrame([{
            "patient_id": "HYB-P001",
            "actual_response_score_percent": 34.0,
            "random_forest_regressor_response_score_percent": 35.0,
        }]).to_csv(regression_path, index=False)
        metrics_path.write_text(json.dumps({
            "best_model_by_patient_level_roc_auc": "gradient_boosting",
            "response_regression": {
                "best_model_by_patient_level_mae": "random_forest_regressor"
            },
        }), encoding="utf-8")

        prediction = load_complete_synthetic_patient_prediction(
            "HYB-P001",
            predictions_csv_path=str(predictions_path),
            response_regression_predictions_csv_path=str(regression_path),
            metrics_json_path=str(metrics_path),
        )

        hybrid = prediction["hybrid_mle_signal"]
        self.assertEqual(hybrid["status"], "favorable_response_signal")
        self.assertEqual(hybrid["classification_model"], "gradient_boosting")
        self.assertEqual(hybrid["regression_model"], "random_forest_regressor")
        self.assertAlmostEqual(hybrid["hybrid_score"], 81.8)

    def test_detailed_training_report_exports_hybrid_rules_and_residuals(self):
        test_dir = _make_temp_dir(_temp_root())
        rows = []
        for idx in range(6):
            patient_id = f"RPT-P{idx:03d}"
            for cycle in [1, 2]:
                rows.append({
                    "patient_id": patient_id,
                    "cycle": cycle,
                    "age": 48 + idx,
                    "stage": "IIB",
                    "molecular_subtype": "HR+/HER2-",
                    "regimen": "AC-T",
                    "mri_percent_change_from_baseline": -20.0 - idx,
                    "mri_tumor_size_cm": 2.0,
                    "max_symptom_severity": 4 + (idx % 2),
                    "symptom_count": 2,
                    "nadir_wbc": 2.2,
                    "nadir_anc": 1.1,
                    "nadir_hemoglobin": 10.5,
                    "nadir_platelets": 140,
                    "intervention_count": 0,
                    "dose_delayed": 0,
                    "dose_reduced": 0,
                    "final_cancer_status": "minimal_residual_disease",
                    "final_response_category": "partial_response",
                    "treatment_success_binary": 1 if idx >= 3 else 0,
                    "response_score_percent": 20.0 + idx,
                })
        training_path = test_dir / "rows.csv"
        classification_path = test_dir / "classification.csv"
        regression_path = test_dir / "regression.csv"
        metrics_path = test_dir / "metrics.json"
        pd.DataFrame(rows).to_csv(training_path, index=False)
        pd.DataFrame({
            "patient_id": [f"RPT-P{idx:03d}" for idx in range(6)],
            "actual_label": [0, 0, 0, 1, 1, 1],
            "gradient_boosting_calibrated_probability": [0.1, 0.2, 0.3, 0.8, 0.9, 0.95],
            "gradient_boosting_probability": [0.12, 0.25, 0.35, 0.76, 0.87, 0.92],
        }).to_csv(classification_path, index=False)
        pd.DataFrame({
            "patient_id": [f"RPT-P{idx:03d}" for idx in range(6)],
            "actual_response_score_percent": [20, 21, 22, 23, 24, 25],
            "random_forest_regressor_response_score_percent": [19, 20, 23, 23, 25, 26],
        }).to_csv(regression_path, index=False)
        metrics_path.write_text(json.dumps({
            "best_model_by_patient_level_roc_auc": "gradient_boosting",
            "best_response_regressor_by_patient_level_mae": "random_forest_regressor",
            "models": {"gradient_boosting": {"patient_level_roc_auc": 0.9}},
            "response_regression": {"models": {"random_forest_regressor": {"patient_level_mae": 1.0}}},
        }), encoding="utf-8")

        report = generate_detailed_training_report(
            training_rows_path=str(training_path),
            classification_predictions_path=str(classification_path),
            regression_predictions_path=str(regression_path),
            metrics_path=str(metrics_path),
            output_dir=str(test_dir / "report"),
        )

        self.assertTrue(Path(report["files"]["test_set_predictions_detailed_csv"]).exists())
        self.assertTrue(Path(report["files"]["regression_residual_review_csv"]).exists())
        self.assertEqual(report["best_classifier"], "gradient_boosting")
        self.assertEqual(report["best_regressor"], "random_forest_regressor")

    def test_qin_cycle_sync_merges_agents_and_restores_cbc_density(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="QIN-BREAST-02-UNIT", name="QIN Unit", diagnosis="Breast cancer demo"))
            db.add_all([
                Treatment(patient_id="QIN-BREAST-02-UNIT", date=date(2026, 1, 1), cycle=1, drug="Doxorubicin"),
                Treatment(patient_id="QIN-BREAST-02-UNIT", date=date(2026, 1, 1), cycle=2, drug="Cyclophosphamide"),
                Treatment(patient_id="QIN-BREAST-02-UNIT", date=date(2026, 1, 4), cycle=3, drug="Paclitaxel"),
                Treatment(patient_id="QIN-BREAST-02-UNIT", date=date(2026, 2, 1), cycle=4, drug="Trastuzumab"),
            ])
            db.commit()

            before = audit_patient_data_coherence(db, output_path=None)
            unit_before = next(row for row in before["patients"] if row["patient_id"] == "QIN-BREAST-02-UNIT")
            self.assertTrue(any(issue["code"] == "same_date_treatment_components" for issue in unit_before["issues"]))

            sync_result = sync_qin_treatment_cycles(db)

            treatments = db.query(Treatment).filter(Treatment.patient_id == "QIN-BREAST-02-UNIT").order_by(Treatment.cycle).all()
            self.assertEqual(sync_result["patients_updated"], 1)
            self.assertEqual(len(treatments), 2)
            self.assertEqual(treatments[0].cycle, 1)
            self.assertIn("Doxorubicin", treatments[0].drug)
            self.assertIn("Cyclophosphamide", treatments[0].drug)
            self.assertIn("Paclitaxel", treatments[0].drug)
            self.assertEqual(db.query(LabResult).filter(LabResult.patient_id == "QIN-BREAST-02-UNIT").count(), 5)

            after = audit_patient_data_coherence(db, output_path=None)
            unit_after = next(row for row in after["patients"] if row["patient_id"] == "QIN-BREAST-02-UNIT")
            self.assertEqual(unit_after["status"], "passed")
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

    def test_calibration_and_subgroup_diagnostics_are_claim_aware(self):
        labels = pd.Series(([1, 0] * 20)).to_numpy()
        probabilities = pd.Series(
            [0.90, 0.10, 0.82, 0.15, 0.74, 0.18, 0.68, 0.24] * 5
        ).to_numpy()

        calibration = admin_analytics._calibration_metrics(labels, probabilities, bins=5)

        self.assertEqual(calibration["posthoc_calibration"]["method"], "heldout_posthoc_calibration")
        self.assertGreaterEqual(len(calibration["posthoc_calibration"]["candidates"]), 3)
        self.assertIn("best_validation_ece", calibration["posthoc_calibration"])

        subgroup_frame = pd.DataFrame({
            "patient_id": [f"SG-{index:03d}" for index in range(14)],
            "actual_label": ([1, 0] * 5) + [1, 0, 1, 0],
            "probability": ([0.92, 0.08] * 5) + [0.70, 0.30, 0.60, 0.40],
            "stage": (["II"] * 10) + (["IV"] * 4),
        })
        subgroups = admin_analytics._subgroup_performance(subgroup_frame)

        self.assertEqual(subgroups["low_support_group_count"], 1)
        self.assertEqual(subgroups["powered_group_status"], "passed")
        self.assertEqual(subgroups["status"], "acceptable")
        self.assertIn("low_support", {row["status"] for row in subgroups["rows"]})

    def test_mri_derived_feature_service_documents_report_pipeline(self):
        frame = pd.DataFrame({
            "patient_id": ["P1", "P2"],
            "latest_mri_percent_change": [-52.0, -12.0],
            "latest_mri_tumor_size_cm": [1.2, 3.1],
        })
        reports = pd.DataFrame([
            {"patient_id": "P1", "date": "2026-01-01", "cycle": 0, "timepoint": "baseline", "tumor_size_cm": 3.0, "percent_change_from_baseline": 0.0},
            {"patient_id": "P1", "date": "2026-02-01", "cycle": 1, "timepoint": "cycle_1", "tumor_size_cm": 1.2, "percent_change_from_baseline": -60.0},
            {"patient_id": "P2", "date": "2026-01-01", "cycle": 0, "timepoint": "baseline", "tumor_size_cm": 3.5, "percent_change_from_baseline": 0.0},
            {"patient_id": "P2", "date": "2026-02-01", "cycle": 1, "timepoint": "cycle_1", "tumor_size_cm": 3.1, "percent_change_from_baseline": -11.4},
        ])

        summary = build_mri_derived_feature_summary(frame, reports)

        self.assertEqual(summary["status"], "acceptable")
        self.assertEqual(summary["report_pipeline"]["status"], "passed")
        self.assertGreater(len(summary["report_pipeline"]["steps"]), 0)

    def test_evaluation_report_and_registry_artifacts_are_versioned(self):
        db = _temp_db_session()
        output_dir = _make_temp_dir(_temp_root()) / "eval_reports"
        try:
            registered = register_complete_synthetic_champion(
                db=db,
                version="unit-test",
                promotion_status="candidate",
                promotion_reason="unit test registration",
            )
            report = generate_versioned_evaluation_report(
                db=db,
                output_root=str(output_dir),
                run_id="unit-test-run",
            )

            self.assertEqual(registered["metadata"]["promotion_status"], "candidate")
            self.assertTrue(Path(report["files"]["evaluation_report_json"]).exists())
            self.assertTrue(Path(report["files"]["threshold_operating_points_csv"]).exists())
            self.assertTrue(Path(report["latest_manifest_path"]).exists())
        finally:
            db.close()
            db.bind.dispose()

    def test_validation_rejects_impossible_cbc_and_warns_on_extreme_values(self):
        with self.assertRaises(ValueError):
            validate_cbc_values(wbc=-1, hemoglobin=12.0, platelets=200)

        warnings = validate_cbc_values(wbc=1.4, hemoglobin=6.5, platelets=48)

        self.assertEqual(len(warnings), 3)
        self.assertTrue(all(item["level"] == "clinician_review" for item in warnings))

    def test_validation_rejects_bad_symptom_severity(self):
        with self.assertRaises(ValueError):
            validate_symptom_payload("fatigue", 11)

    def test_data_availability_reports_missing_model_and_insufficient_timeline(self):
        report = {
            "lab_history": [{"date": "2026-01-01", "wbc": 5.0, "hemoglobin": 12.0, "platelets": 200}],
            "symptoms": [],
            "timeline": [{"date": "2026-01-01", "title": "Baseline", "summary": "one event"}],
            "treatment_effects": [],
            "radiology_summary": None,
            "mri_registry": [],
            "synthetic_model_prediction": None,
            "multimodal_assessment": {
                "signals": {
                    "mri_response": {"status": "unavailable", "source": "none"},
                }
            },
        }

        availability = build_data_availability(report)
        statuses = {item["name"]: item["status"] for item in availability["items"]}

        self.assertEqual(statuses["CBC trend"], "insufficient_data")
        self.assertEqual(statuses["Model signal"], "model_unavailable")
        self.assertIn("Interpret with limitations", availability["clinician_style_summary"])

    def test_demo_patient_sync_creates_coherent_cycle_lab_timeline(self):
        db = _temp_db_session()
        try:
            result = sync_demo_patient_journey(db)

            treatments = db.query(Treatment).filter(Treatment.patient_id == "P001").order_by(Treatment.cycle).all()
            labs = db.query(LabResult).filter(LabResult.patient_id == "P001").all()
            imaging = db.query(ImagingReport).filter(ImagingReport.patient_id == "P001").all()

            self.assertEqual(result["treatments"], 6)
            self.assertEqual([row.cycle for row in treatments], [1, 2, 3, 4, 5, 6])
            self.assertGreaterEqual(len(labs), 12)
            self.assertEqual(len(imaging), 3)
        finally:
            db.close()
            db.bind.dispose()

    def test_app_monitoring_counts_failures_and_prediction_confidence(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="LOG-P001", name="Log Patient", diagnosis="Breast cancer demo"))
            db.add(PredictionAuditLog(
                patient_id="LOG-P001",
                model_name="demo_model",
                model_version="v1",
                input_reference="{}",
                prediction_json='{"response_probability": 0.73}',
            ))
            db.commit()
            log_app_event(db, event_type="prediction", patient_id="LOG-P001", status="ok")
            log_app_event(db, event_type="validation_error", patient_id="LOG-P001", status="error", error_message="bad input")

            summary = build_app_monitoring_summary(db)

            self.assertEqual(summary["prediction_count"], 1)
            self.assertEqual(summary["failed_event_count"], 1)
            self.assertEqual(summary["confidence_distribution"]["sample_count"], 1)
            self.assertEqual(db.query(AppEventLog).count(), 2)
        finally:
            db.close()
            db.bind.dispose()
