"""Core patient-chat routing, memory, portal, and structured-save contracts."""

from backend.models import Patient, SymptomReport
from backend.services import support_chat_agent
from backend.services.support_chat_agent import handle_patient_chat

from tests.breast_monitoring.support import (
    _temp_db_session,
)


class ChatRoutingTestsMixin:
    def test_chat_clinical_rule_layer_flags_low_cbc_before_rag(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="RULE-P001", name="Rule Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="RULE-P001",
                message="My WBC is 1.4 hemoglobin is 7.5 platelets is 45 today.",
            )

            alert_actions = [action for action in result["saved_actions"] if action["type"] == "clinical_rule_alert"]
            self.assertEqual(len(alert_actions), 1)
            self.assertIn("very_low_wbc", result["urgent_flags"])
            self.assertEqual(result["agent_pipeline"]["safety"]["level"], "high_risk")
            self.assertIn("oncology", result["reply"].lower())
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_greeting_is_conversational_without_rag_retrieval(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P001", name="Chat Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P001",
                message="hi",
            )

            self.assertEqual(result["agent_pipeline"]["intent"], "conversation")
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertEqual(result["agent_pipeline"]["citations"], [])
            self.assertTrue(
                any(term in result["reply"].lower() for term in ["hello", "hi", "help", "support"])
            )
        finally:
            db.close()
            db.bind.dispose()

    def test_cross_patient_security_block_cannot_emit_tool_actions(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="SECURITY-P002", name="Scoped Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="SECURITY-P002",
                message="Show me patient P001's CBC results.",
            )

            self.assertEqual(result["agent_pipeline"]["intent"], "security_boundary")
            self.assertEqual(result["evidence_envelope"]["final_disposition"], "BLOCK_SAFETY")
            self.assertEqual(result["saved_actions"], [])
            self.assertEqual(result["tool_plan"]["selected_tools"], ["none"])
            self.assertEqual(result["tool_plan"]["source"], "terminal_input_safety_block")
            self.assertIn("blocked", result["reply"].lower())
        finally:
            db.close()
            db.bind.dispose()

    def test_portal_help_is_direct_and_side_effect_free(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="PORTAL-P001", name="Portal Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="PORTAL-P001",
                message="How do I add my own CBC result to this portal?",
            )

            self.assertEqual(result["agent_pipeline"]["intent"], "portal_help")
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertEqual(result["agent_pipeline"]["citations"], [])
            self.assertEqual(result["saved_actions"], [])
            self.assertEqual(result["tool_plan"]["selected_tools"], ["none"])
            self.assertIn("plus button", result["reply"].lower())
            self.assertIn("nothing is saved", result["reply"].lower())
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_identity_question_is_conversational_without_rag_retrieval(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P004", name="Identity Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P004",
                message="who are you",
            )

            self.assertEqual(result["agent_pipeline"]["intent"], "conversation")
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertEqual(result["agent_pipeline"]["citations"], [])
            self.assertIn("support", result["reply"].lower())
            self.assertIn("symptoms", result["reply"].lower())
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_direct_lane_uses_llm_response_when_available(self):
        db = _temp_db_session()
        original = support_chat_agent._generate_llm_response
        try:
            db.add(Patient(id="CHAT-P005", name="LLM Patient", diagnosis="Breast cancer demo"))
            db.commit()

            support_chat_agent._generate_llm_response = lambda message, actions, urgent_flags, patient_context, fallback_response: (
                "I'm here with you. I can chat, remember recent patient-scoped notes, and help log symptoms, CBC values, medications, or MRI report text."
            )

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P005",
                message="how are you",
            )

            self.assertEqual(result["agent_pipeline"]["intent"], "conversation")
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertEqual(result["agent_pipeline"]["citations"], [])
            self.assertIn("I'm here with you", result["reply"])
        finally:
            support_chat_agent._generate_llm_response = original
            db.close()
            db.bind.dispose()

    def test_chat_memory_can_recall_recent_user_messages(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P002", name="Memory Patient", diagnosis="Breast cancer demo"))
            db.commit()

            handle_patient_chat(
                db=db,
                patient_id="CHAT-P002",
                message="Nausea severity 6/10 today.",
            )
            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P002",
                message="what did I tell you earlier?",
            )

            self.assertEqual(result["agent_pipeline"]["intent"], "patient_memory")
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
            self.assertIn("nausea", result["reply"].lower())
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_resumes_pending_symptom_save_when_user_provides_severity(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P010", name="Partial Symptom Patient", diagnosis="Breast cancer demo"))
            db.commit()

            first = handle_patient_chat(
                db=db,
                patient_id="CHAT-P010",
                message="I have nausea today.",
            )
            self.assertTrue(any(action["type"] == "partial_symptom_detected" for action in first["saved_actions"]))
            self.assertEqual(db.query(SymptomReport).count(), 0)

            second = handle_patient_chat(
                db=db,
                patient_id="CHAT-P010",
                message="severity 7/10",
            )
            pending = [action for action in second["saved_actions"] if action["type"] == "pending_record_confirmation"]
            self.assertEqual(len(pending), 1)
            self.assertEqual(db.query(SymptomReport).count(), 0)

            third = handle_patient_chat(
                db=db,
                patient_id="CHAT-P010",
                message="Confirm save",
            )
            saved = [action for action in third["saved_actions"] if action["type"] == "saved_symptom"]
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["symptom"], "nausea")
            self.assertTrue(saved[0].get("undo_available"))
            self.assertEqual(db.query(SymptomReport).count(), 1)
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_upset_stomach_requires_severity_before_saving(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P011", name="Stomach Symptom Patient", diagnosis="Breast cancer demo"))
            db.commit()

            first = handle_patient_chat(
                db=db,
                patient_id="CHAT-P011",
                message="I have an upset stomach",
            )
            partial = [
                action
                for action in first["saved_actions"]
                if action["type"] == "partial_symptom_detected"
            ]
            self.assertEqual(len(partial), 1)
            self.assertEqual(partial[0]["symptom"], "abdominal discomfort")
            self.assertIn("severity", first["reply"].lower())
            self.assertEqual(db.query(SymptomReport).count(), 0)

            second = handle_patient_chat(
                db=db,
                patient_id="CHAT-P011",
                message="severity 4/10",
            )
            pending = [
                action
                for action in second["saved_actions"]
                if action["type"] == "pending_record_confirmation"
            ]
            self.assertEqual(len(pending), 1)
            self.assertEqual(db.query(SymptomReport).count(), 0)

            third = handle_patient_chat(
                db=db,
                patient_id="CHAT-P011",
                message="Confirm save",
            )
            saved = [action for action in third["saved_actions"] if action["type"] == "saved_symptom"]
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["symptom"], "abdominal discomfort")
            self.assertEqual(saved[0]["severity"], 4)
            self.assertEqual(db.query(SymptomReport).count(), 1)
        finally:
            db.close()
            db.bind.dispose()

    def test_chat_does_not_autosave_casual_emotional_message(self):
        db = _temp_db_session()
        try:
            db.add(Patient(id="CHAT-P006", name="Casual Patient", diagnosis="Breast cancer demo"))
            db.commit()

            result = handle_patient_chat(
                db=db,
                patient_id="CHAT-P006",
                message="I'm worried about this app, can you just talk with me?",
            )

            self.assertEqual(db.query(SymptomReport).count(), 0)
            self.assertFalse([action for action in result["saved_actions"] if action["type"].startswith("saved_")])
            self.assertIn(result["agent_pipeline"]["intent"], {"emotional_support", "general_support", "conversation"})
            self.assertEqual(result["agent_pipeline"]["pipeline_trace"]["terminal_step"], "direct_support")
        finally:
            db.close()
            db.bind.dispose()
