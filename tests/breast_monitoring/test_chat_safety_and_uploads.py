"""High-risk follow-up, treatment-boundary, and imaging-upload contracts."""

from backend.models import ImagingReport, Patient, SymptomReport
from backend.services import support_chat_agent
from backend.services.support_chat_agent import handle_patient_chat

from tests.breast_monitoring.support import (
    _temp_db_session,
)


class ChatSafetyAndUploadsTestsMixin:
    def test_immediate_danger_statement_uses_complete_deterministic_safety_reply(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-SAFETY-P001", name="Safety Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-SAFETY-P001",
                message="i think im dying",
            )

            self.assertEqual(result["agent_pipeline"]["intent"], "safety_boundary")
            self.assertEqual(result["agent_pipeline"]["safety"]["level"], "high_risk")
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertIn("nearest emergency department", result["reply"].lower())
            self.assertFalse(result["reply"].lower().rstrip().endswith("or go to"))
            self.assertEqual(db.query(SymptomReport).count(), 0)
        finally:
            db.close()
            db.bind.dispose()

    def test_safety_location_followup_keeps_previous_turn_context(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-SAFETY-P002", name="Follow-up Patient", diagnosis="Breast cancer demo"))
            db.commit()

            first = handle_patient_chat(
                db=db,
                patient_id="CHAT-SAFETY-P002",
                message="i think im dying",
            )
            self.assertIn("emergency", first["reply"].lower())

            second = handle_patient_chat(
                db=db,
                patient_id="CHAT-SAFETY-P002",
                message="go to where?",
            )

            self.assertEqual(second["agent_pipeline"]["intent"], "safety_boundary")
            self.assertEqual(second["agent_pipeline"]["safety"]["level"], "high_risk")
            self.assertEqual(second["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertIn("nearest emergency department", second["reply"].lower())
            self.assertNotIn("which part of the portal", second["reply"].lower())
        finally:
            db.close()
            db.bind.dispose()

    def test_treatment_boundary_carries_into_referential_followup(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-SAFETY-P003", name="Context Safety Patient", diagnosis="Breast cancer demo"))
            db.commit()

            first = handle_patient_chat(
                db=db,
                patient_id="CHAT-SAFETY-P003",
                message="Should I stop my treatment?",
            )
            self.assertEqual(first["agent_pipeline"]["intent"], "treatment_decision_boundary")

            second = handle_patient_chat(
                db=db,
                patient_id="CHAT-SAFETY-P003",
                message="what if just tonight?",
            )

            self.assertEqual(second["agent_pipeline"]["intent"], "treatment_decision_boundary")
            self.assertTrue(second["agent_pipeline"]["safety"].get("context_reused"))
            self.assertIn(
                second["agent_pipeline"]["safety"].get("safety_source"),
                {"contextual_boundary_carryover", "contextual_composition"},
            )
            self.assertEqual(db.query(SymptomReport).count(), 0)
        finally:
            db.close()
            db.bind.dispose()

    def test_existing_record_wording_does_not_claim_patient_authorship(self):
        original = "You've logged nausea severity 6/10 today and recent labs are in your record."
        neutral = support_chat_agent._enforce_record_provenance(original, [])
        self.assertIn("portal record currently shows", neutral.lower())
        self.assertNotIn("you've logged", neutral.lower())

        verified = support_chat_agent._enforce_record_provenance(
            original,
            [{"type": "saved_symptom", "symptom": "nausea", "severity": 6}],
        )
        self.assertEqual(verified, original)

    def test_high_risk_dangling_location_clause_is_completed(self):
        repaired = support_chat_agent._ensure_complete_safety_reply(
            "Please call local emergency services or go to",
            {"level": "high_risk", "scope": "urgent_or_safety_related"},
        )
        self.assertEqual(
            repaired,
            "Please call local emergency services or go to the nearest emergency department now.",
        )

    def test_chat_short_mri_hint_requests_details_without_saving(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P007", name="Short MRI Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P007",
                message="mri",
            )

            self.assertEqual(db.query(ImagingReport).count(), 0)
            self.assertTrue(any(action["type"] == "partial_imaging_detected" for action in result["saved_actions"]))
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertEqual(result["agent_pipeline"]["citations"], [])
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_confirms_mri_report_text_before_saving(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P003", name="MRI Patient", diagnosis="Breast cancer demo"))
            db.commit()

            preview = handle_patient_chat(
                db=db,
                patient_id="CHAT-P003",
                message="MRI report on 2026-02-01 impression: right breast mass decreased to 2.1 cm.",
            )
            self.assertEqual(db.query(ImagingReport).count(), 0)
            self.assertTrue(any(action["type"] == "pending_record_confirmation" for action in preview["saved_actions"]))

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P003",
                message="Confirm save",
            )

            saved_imaging = [action for action in result["saved_actions"] if action["type"] == "saved_imaging_report"]
            self.assertEqual(len(saved_imaging), 1)
            self.assertEqual(db.query(ImagingReport).count(), 1)
            report = db.query(ImagingReport).first()
            self.assertEqual(report.modality, "Breast MRI")
            self.assertEqual(str(report.date), "2026-02-01")
            self.assertIn("decreased", report.impression.lower())
            self.assertIn("mri report", result["reply"].lower())
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertEqual(result["agent_pipeline"]["citations"], [])
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_confirms_ct_report_before_saving_and_flagging(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P008", name="CT Patient", diagnosis="Breast cancer demo"))
            db.commit()

            preview = handle_patient_chat(
                db=db,
                patient_id="CHAT-P008",
                message=(
                    "CT abdomen/pelvis report on 2026-03-01 impression: "
                    "new ascites and peritoneal nodularity concerning for metastatic disease."
                ),
            )
            self.assertEqual(db.query(ImagingReport).count(), 0)
            self.assertTrue(any(action["type"] == "pending_record_confirmation" for action in preview["saved_actions"]))

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P008",
                message="Confirm save",
            )

            saved_imaging = [action for action in result["saved_actions"] if action["type"] == "saved_imaging_report"]
            indicator_actions = [action for action in result["saved_actions"] if action["type"] == "possible_metastatic_indicator"]
            self.assertEqual(len(saved_imaging), 1)
            self.assertEqual(len(indicator_actions), 1)
            self.assertEqual(db.query(ImagingReport).count(), 1)
            report = db.query(ImagingReport).first()
            self.assertEqual(report.modality, "CT abdomen/pelvis")
            self.assertEqual(report.body_site, "Abdomen/pelvis")
            self.assertEqual(str(report.date), "2026-03-01")
            self.assertIn("ascites", report.impression.lower())
            self.assertIn("clinician review", result["reply"].lower())
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_confirms_abdominal_ultrasound_report_before_saving(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P009", name="Ultrasound Patient", diagnosis="Breast cancer demo"))
            db.commit()

            preview = handle_patient_chat(
                db=db,
                patient_id="CHAT-P009",
                message="Ultrasound abdomen report on 2026-03-04 impression: new hepatic lesion.",
            )
            self.assertEqual(db.query(ImagingReport).count(), 0)
            self.assertTrue(any(action["type"] == "pending_record_confirmation" for action in preview["saved_actions"]))

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P009",
                message="Confirm save",
            )

            saved_imaging = [action for action in result["saved_actions"] if action["type"] == "saved_imaging_report"]
            self.assertEqual(len(saved_imaging), 1)
            report = db.query(ImagingReport).first()
            self.assertEqual(report.modality, "Abdominal ultrasound")
            self.assertEqual(report.body_site, "Abdomen")
        finally:
            db.close()
            db.bind.dispose()
