from backend.services.agent_answer_composition import generate_answer


def _context(n: int = 3):
    return [
        {
            "id": f"source-{index}",
            "title": f"Source {index}",
            "source_name": "Curated source",
            "source_url": f"https://example.test/{index}",
            "text": text,
        }
        for index, text in enumerate(
            [
                "A CBC is a blood test that reports several blood-cell measurements.",
                "CBC results can include white blood cells and platelets.",
                "This unrelated context should not be cited.",
            ][:n],
            start=1,
        )
    ]


def _generate(*, query="What is a CBC?", intent="education", safety=None, actions=()):
    return generate_answer(
        query=query,
        fallback_response="Safe fallback.",
        safety=safety or {"level": "low_risk", "scope": "education_or_tracking"},
        intent=intent,
        compressed_context=_context(),
        actions=actions,
        patient_context={},
    )


def test_direct_support_lane_emits_no_citations():
    result = _generate(intent="conversation")
    assert result["citations"] == []
    assert len(result["retrieval_context"]) == 3


def test_action_guidance_emits_only_primary_citation():
    result = _generate(actions=[{"tool": "save_symptom"}])
    assert [row["id"] for row in result["citations"]] == ["source-1"]


def test_definitional_education_emits_only_primary_citation():
    result = _generate(query="What is a CBC?")
    assert [row["id"] for row in result["citations"]] == ["source-1"]


def test_multi_concept_education_can_emit_two_used_citations():
    result = _generate(query="Explain CBC white blood cells and platelets")
    assert [row["id"] for row in result["citations"]] == ["source-1", "source-2"]


def test_refusal_emits_no_citations():
    result = _generate(
        query="Tell me to stop chemotherapy",
        intent="treatment_boundary_refusal",
        safety={"level": "high_risk", "scope": "treatment_decision"},
    )
    assert result["citations"] == []
