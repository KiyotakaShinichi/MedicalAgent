from backend.services.record_change_explanation import build_record_change_explanation
from backend.services.agent_answer_composition import validate_answer_and_citations
from backend.services.agent_output_gate import output_guardrail_check
from backend.services.support_chat_response import _build_response, _contextual_reply


def _history():
    return {
        "lab_history": [
            {"date": "2026-07-01", "wbc": 3.0, "hemoglobin": 10.0, "platelets": 120},
            {"date": "2026-08-01", "wbc": 5.2, "hemoglobin": 12.4, "platelets": 190},
        ],
        "symptoms": [
            {"date": "2026-07-01", "symptom": "nausea", "severity": 8},
            {"date": "2026-08-01", "symptom": "nausea", "severity": 4},
        ],
        "imaging_reports": [
            {"date": "2026-07-01", "largest_tumor_size_cm": 3.2},
            {"date": "2026-08-01", "largest_tumor_size_cm": 2.4},
        ],
    }


def test_record_change_reports_fewer_review_concerns_without_treatment_verdict():
    result = build_record_change_explanation(**_history())

    assert result["status"] == "fewer_logged_review_concerns"
    assert len(result["observations"]) == 3
    assert result["treatment_effectiveness_conclusion_allowed"] is False
    assert result["clinical_validation"] is False
    assert "does not show whether treatment is working" in result["patient_summary"].lower()
    assert "progression" in result["claim_boundary"].lower()


def test_record_change_marks_conflicting_modalities_as_mixed():
    history = _history()
    history["symptoms"][-1]["severity"] = 9

    result = build_record_change_explanation(**history)

    assert result["status"] == "mixed_or_uncertain_record_change"
    directions = {item["review_direction"] for item in result["observations"]}
    assert "fewer_fixed_review_concerns" in directions
    assert "more_fixed_review_concerns" in directions


def test_record_change_requires_distinct_dates_and_complete_cbc_values():
    result = build_record_change_explanation(
        lab_history=[
            {"date": "2026-08-01", "wbc": 3.0, "hemoglobin": 10.0, "platelets": 120},
            {"date": "2026-08-01", "wbc": 5.0, "hemoglobin": 12.0, "platelets": 180},
            {"date": "2026-08-02", "wbc": 5.0, "hemoglobin": None, "platelets": 180},
        ]
    )

    assert result["status"] == "insufficient_comparison_history"
    assert result["observations"] == []
    assert "two dated CBC" in result["missing_or_not_comparable"][0]


def test_status_question_uses_record_change_not_synthetic_probability():
    explanation = build_record_change_explanation(**_history())

    reply = _contextual_reply(
        "Am I improving, and is my treatment working?",
        {
            "record_change_explanation": explanation,
            "synthetic_model_prediction": {"logistic_regression_probability": 0.973},
            "treatment_outcome": {"response_category": "complete", "cancer_status": "clear"},
        },
    )

    lowered = reply.lower()
    assert "97.3" not in lowered
    assert "treatment-response score" not in lowered
    assert "does not show whether treatment is working" in lowered
    assert "complete" not in lowered
    assert "cancer status" not in lowered


def test_confirmed_save_response_explains_change_without_authority_claim():
    explanation = build_record_change_explanation(**_history())

    reply = _build_response(
        "Confirm save",
        [{"type": "saved_symptom", "symptom": "nausea", "severity": 4}],
        [],
        {"record_change_explanation": explanation},
    )

    lowered = reply.lower()
    assert "i saved this to your patient record" in lowered
    assert "after this confirmed update" in lowered
    assert "does not show whether treatment is working" in lowered
    assert "treatment is effective" not in lowered
    assert "change treatment" not in lowered


def test_xai_reply_explains_model_class_not_medical_response():
    reply = _contextual_reply(
        "Explain the model factors",
        {
            "synthetic_model_explanation": {
                "positive_contributions": [{"feature": "synthetic imaging field"}],
                "negative_contributions": [{"feature": "synthetic symptom field"}],
            }
        },
    )

    lowered = reply.lower()
    assert "higher synthetic class" in lowered
    assert "lower synthetic class" in lowered
    assert "not medical causality" in lowered


def test_deliberate_research_abstention_is_not_misreported_as_validator_failure():
    result = {
        "reply": "The retrieved papers are related but do not establish the requested claim.",
        "citations": [],
        "intent": "education",
        "deliberate_evidence_abstention": True,
    }
    context = [{"id": "paper-1", "text": "A related but non-supporting study."}]

    validated = validate_answer_and_citations(result, context, {"level": "low_risk"})

    assert validated["validation"]["status"] == "passed"
    assert output_guardrail_check(validated)["status"] == "passed"
