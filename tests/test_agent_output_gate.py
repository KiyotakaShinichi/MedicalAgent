from backend.services.agent_output_gate import output_guardrail_check
from backend.services.agent_knowledge_snippets import KNOWLEDGE_SNIPPETS
from backend.services.rag_intent_modes import MODES


def _result(*, intent: str, rag_mode: str) -> dict:
    return {
        "reply": "Use the Labs section in the portal to review saved entries.",
        "intent": intent,
        "rag_mode": rag_mode,
        "retrieval_context": [{"id": "portal-guide"}],
        "citations": [],
        "validation": {"status": "passed", "issues": []},
        "safety": {"level": "low_risk"},
    }


def test_portal_help_does_not_require_medical_evidence_citations() -> None:
    verdict = output_guardrail_check(
        _result(intent="portal_help", rag_mode="portal_help_rag")
    )

    assert verdict == {"status": "passed", "issues": []}


def test_medical_education_with_retrieval_still_requires_citations() -> None:
    verdict = output_guardrail_check(
        _result(intent="education", rag_mode="education_rag")
    )

    assert verdict["status"] == "failed"
    assert "missing_citations" in verdict["issues"]


def test_portal_intent_cannot_bypass_citations_in_an_evidence_mode() -> None:
    verdict = output_guardrail_check(
        _result(intent="portal_help", rag_mode="education_rag")
    )

    assert verdict["status"] == "failed"
    assert "missing_citations" in verdict["issues"]


def test_portal_help_source_and_fallback_explain_real_navigation() -> None:
    portal_source = next(
        item for item in KNOWLEDGE_SNIPPETS if item["id"] == "portal-upload-guide"
    )
    fallback = MODES["portal_help_rag"].insufficient_evidence_default

    assert "open Labs" in portal_source["text"]
    assert "plus button" in portal_source["text"]
    assert "saved only after" in portal_source["text"]
    assert "left navigation" in fallback
    assert "contact info in the footer" not in fallback
