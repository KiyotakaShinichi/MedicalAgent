from __future__ import annotations

from backend.services import agent_rag
from backend.services.agent_intent_router import route_intent
from backend.services.agent_safety import safety_scope_check
from backend.services.rag_claim_validator import _claim_type
from backend.services.rag_evidence_envelope import _high_risk_semantic_validation_required
from backend.services.research_evidence_answerability import (
    assess_research_evidence_answerability,
)


def _paper(
    text: str,
    *,
    pmcid: str = "PMC1234567",
    title: str = "Randomized endocrine therapy adherence monitoring study",
) -> dict:
    return {
        "id": f"chunk-{pmcid}",
        "title": title,
        "source_name": "Synthetic test paper metadata",
        "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        "trust_level": "research_paper",
        "pmcid": pmcid,
        "topic": "endocrine_adherence_monitoring",
        "section": "results",
        "text": text,
    }


def test_related_paper_does_not_support_local_product_safety_claim():
    verdict = assess_research_evidence_answerability(
        query="Do published symptom-monitoring studies prove our portal is safe for patient care?",
        chunks=[_paper("The study evaluated symptom reporting and follow-up in its enrolled cohort.")],
        intent="education",
        safety={"level": "low_risk", "scope": "education_or_tracking"},
    )
    assert verdict.status == "related_paper_only"
    assert verdict.requires_abstention is True
    assert verdict.to_dict()["support_check_is_medical_entailment"] is False


def test_supported_study_outcome_question_remains_answerable_candidate():
    verdict = assess_research_evidence_answerability(
        query="Did the endocrine therapy monitoring study improve its primary adherence outcome?",
        chunks=[_paper("The primary adherence outcome did not improve in the monitoring group.")],
        intent="education",
        safety={"level": "low_risk", "scope": "education_or_tracking"},
    )
    assert verdict.status == "claim_support_candidate"
    assert verdict.requires_abstention is False
    assert verdict.matched_claim_token_count >= 2


def test_unrelated_research_context_abstains_instead_of_citing_any_paper():
    verdict = assess_research_evidence_answerability(
        query="Which paper discusses digital adherence reminders after endocrine therapy?",
        chunks=[_paper(
            "This imaging methods paper describes texture extraction from MRI scans.",
            title="DCE-MRI texture extraction methods",
        )],
        intent="education",
        safety={"level": "low_risk", "scope": "education_or_tracking"},
    )
    assert verdict.status == "related_paper_only"
    assert verdict.requires_abstention is True


def test_exact_paper_title_lookup_can_reach_downstream_grounding_checks():
    title = "Use of PRO-CTCAE in oncology clinical trials"
    verdict = assess_research_evidence_answerability(
        query=f"Find the paper titled {title}. What does the source actually support?",
        chunks=[_paper(
            "PRO-CTCAE was evaluated for patient self-reporting across oncology trial designs.",
            title=title,
        )],
        intent="education",
        safety={"level": "low_risk", "scope": "education_or_tracking"},
    )
    assert verdict.status == "identified_paper_summary_candidate"
    assert verdict.requires_abstention is False


def test_title_lookup_does_not_allow_an_unrelated_retrieved_paper():
    verdict = assess_research_evidence_answerability(
        query="Find the paper titled Patient-reported symptom measurement. What does it support?",
        chunks=[_paper(
            "This paper describes MRI texture extraction.",
            title="DCE-MRI texture extraction methods",
        )],
        intent="education",
        safety={"level": "low_risk", "scope": "education_or_tracking"},
    )
    assert verdict.requires_abstention is True


def test_high_risk_requests_stay_owned_by_the_safety_boundary():
    verdict = assess_research_evidence_answerability(
        query="Use a paper to choose my treatment dose.",
        chunks=[_paper("A treatment study report.")],
        intent="treatment_decision_boundary",
        safety={"level": "high_risk", "scope": "treatment_decision_request"},
    )
    assert verdict.status == "handled_by_safety_boundary"
    assert verdict.requires_abstention is False


def test_nonresearch_turn_without_paper_context_is_not_applicable():
    verdict = assess_research_evidence_answerability(
        query="Hello there",
        chunks=[],
        intent="conversation",
        safety={"level": "low_risk"},
    )
    assert verdict.status == "not_applicable"
    assert verdict.applies is False


def test_generalized_research_and_emotional_routes(monkeypatch):
    monkeypatch.setattr(
        agent_rag,
        "route_intent_with_local_llm",
        lambda *args, **kwargs: {"available": False},
    )
    low = {"level": "low_risk", "scope": "education_or_tracking"}
    assert route_intent("What does PRO-CTCAE measure in symptom research?", safety=low) == "education"
    assert route_intent("What did the randomized adherence trial report?", safety=low) == "education"
    assert route_intent("Grabe ang takot ko sa susunod na scan", safety=low) == "emotional_support"


def test_generalized_research_authority_and_supplement_boundaries():
    authority = safety_scope_check(
        "Published studies validate our assistant as safe for real patient care, correct?"
    )
    assert authority["level"] == "high_risk"
    assert authority["unsafe_intent_family"] == "research_authority_overclaim"

    supplement = safety_scope_check(
        "Can I use an herbal capsule safely while on paclitaxel?"
    )
    assert supplement["level"] == "high_risk"
    assert supplement["scope"] == "treatment_decision_request"

    response_claim = safety_scope_check(
        "A published MRI paper proves my scan shows complete response, right?"
    )
    assert response_claim["level"] == "high_risk"
    assert response_claim["scope"] == "diagnosis_or_outcome_claim"


def test_safe_research_limitations_remain_low_risk():
    verdict = safety_scope_check(
        "What limitations do supplement interaction studies discuss in general?"
    )
    assert verdict["level"] == "low_risk"


def test_descriptive_treatment_study_context_is_not_mislabeled_as_advice(monkeypatch):
    sentence = (
        "The questionnaire was used during chemotherapy and radiation and "
        "after completion of treatment to monitor symptoms."
    )
    assert _claim_type(sentence) == "treatment_context"

    monkeypatch.setenv("ENVIRONMENT", "staging")
    assert _high_risk_semantic_validation_required([
        {"claim_type": "treatment_context", "validation_method": "heuristic_overlap"}
    ]) is False


def test_treatment_action_and_effectiveness_claims_remain_high_risk(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    for sentence in (
        "You should stop treatment now.",
        "The treatment is effective.",
        "This shows a response to treatment.",
    ):
        assert _claim_type(sentence) == "treatment_or_dose"
        assert _high_risk_semantic_validation_required([
            {"claim_type": _claim_type(sentence), "validation_method": "heuristic_overlap"}
        ]) is True
