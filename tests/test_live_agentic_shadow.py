from backend.services.live_agentic_shadow import build_live_agentic_shadow


def test_shadow_is_side_effect_free_and_redacted():
    result = build_live_agentic_shadow(
        "I have nausea severity 6/10 today",
        patient_context={},
        live_intent="data_entry_confirmation",
        live_safety={"level": "low_risk"},
        live_tools=["save_symptom"],
    )

    assert result["shadow_only"] is True
    assert result["changed_live_behavior"] is False
    assert result["simulated_records_written"] == []
    assert "message" not in result
    assert "patient_context" not in result


def test_shadow_compares_route_families_without_claiming_validation():
    result = build_live_agentic_shadow(
        "What is a VUS?",
        patient_context={},
        live_intent="education",
        live_safety={"level": "low_risk"},
        live_tools=["none"],
    )

    assert result["status"] == "observed"
    assert isinstance(result["route_family_aligned"], bool)
    assert "does not" in result["claim_boundary"].lower()
    assert "clinical" in result["claim_boundary"].lower()
