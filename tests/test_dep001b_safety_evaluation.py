from __future__ import annotations

from backend.services.dep001b_semantic_safety import DEP001BSafetyPrediction
from backend.services.dep001b_safety_evaluation import evaluate_dep001b_rows
from backend.services.safety_policy_action import PolicyAction


def _prediction(action: str) -> DEP001BSafetyPrediction:
    return DEP001BSafetyPrediction(
        unsafe_probability=0.9 if action not in {PolicyAction.ALLOW_EDUCATIONAL.value, PolicyAction.ALLOW_WITH_BOUNDARY.value} else 0.1,
        urgent_probability=0.95 if action == PolicyAction.URGENT_ESCALATION.value else 0.01,
        intent_family="URGENT_PRESENTATION" if action == PolicyAction.URGENT_ESCALATION.value else "EDUCATIONAL_GENERAL",
        intent_family_confidence=0.9,
        uncertainty=0.1,
        policy_action=action,
        policy_reason="test",
        model_version="test",
        unsafe_route_threshold=0.4,
        urgent_route_threshold=0.7,
        urgent_independent_threshold=0.95,
        selected_turn_offset=0,
        context_turn_count=0,
    )


def test_metric_contract_separates_unsafe_urgent_and_safe_utility() -> None:
    rows = [
        {"case_id": "u", "text": "u", "language": "english", "intent_family": "TREATMENT_MODIFICATION", "unsafe_expected": True, "urgent_expected": False},
        {"case_id": "e", "text": "e", "language": "taglish", "intent_family": "URGENT_PRESENTATION", "unsafe_expected": True, "urgent_expected": True, "multi_turn": True, "rag_conditioned": True},
        {"case_id": "s", "text": "s", "language": "filipino_heavy", "intent_family": "EDUCATIONAL_GENERAL", "unsafe_expected": False, "urgent_expected": False},
    ]
    actions = {
        "u": _prediction(PolicyAction.REFUSE_ACTIONABLE.value),
        "e": _prediction(PolicyAction.URGENT_ESCALATION.value),
        "s": _prediction(PolicyAction.ALLOW_EDUCATIONAL.value),
    }
    report = evaluate_dep001b_rows(rows, classifier=lambda text, **_kwargs: actions[text])
    assert report["metrics"]["unsafe_released_outputs"] == 0
    assert report["metrics"]["unsafe_intent_recall"] == 1.0
    assert report["metrics"]["urgent_escalation_recall"] == 1.0
    assert report["metrics"]["safe_educational_acceptance_rate"] == 1.0
    assert report["metrics"]["over_refusal_rate"] == 0.0
    assert report["metrics"]["multi_turn_unsafe_recall"] == 1.0
    assert report["metrics"]["rag_conditioned_unsafe_recall"] == 1.0


def test_route_miss_is_visible_even_when_postgen_blocks_canary() -> None:
    rows = [{"case_id": "miss", "text": "miss", "language": "english", "intent_family": "TREATMENT_MODIFICATION", "unsafe_expected": True, "urgent_expected": False}]
    report = evaluate_dep001b_rows(
        rows,
        classifier=lambda *_args, **_kwargs: _prediction(PolicyAction.ALLOW_WITH_BOUNDARY.value),
    )
    assert report["metrics"]["unsafe_released_outputs"] == 0
    assert report["metrics"]["unsafe_intent_recall"] == 0.0
    assert report["targets_passed"] is False
    assert report["failed_case_ids"] == ["miss"]


def test_fail_closed_counts_as_contained_not_accepted() -> None:
    row = {"case_id": "failure", "text": "failure", "language": "english", "intent_family": "UNKNOWN_HIGH_RISK", "unsafe_expected": True, "urgent_expected": False}
    report = evaluate_dep001b_rows(
        [row],
        classifier=lambda *_args, **_kwargs: _prediction(PolicyAction.FAIL_CLOSED.value),
    )
    assert report["metrics"]["unsafe_intent_recall"] == 1.0
    assert report["cases"][0]["safe_accepted"] is False
