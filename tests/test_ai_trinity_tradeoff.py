from __future__ import annotations

import json
from pathlib import Path

from backend.services.ai_trinity_tradeoff import (
    build_ai_trinity_tradeoff,
    write_ai_trinity_tradeoff,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_artifacts_fail_closed_on_missing_provider_cost() -> None:
    payload = build_ai_trinity_tradeoff()

    assert payload["clinical_validation"] is False
    assert payload["healthcare_production_ready"] is False
    assert payload["audited_billing"] is False
    assert payload["axes"]["unit_cost"]["missing_cost_is_zero"] is False
    assert payload["axes"]["unit_cost"]["status"] == "blocked_evidence"
    assert payload["promotion_allowed"] is False


def test_source_governance_is_a_non_compensatory_floor() -> None:
    payload = build_ai_trinity_tradeoff()
    rows = {row["configuration"]: row for row in payload["scenarios"]}

    assert rows["bm25_only"]["latency_budget_pass"] is True
    assert rows["bm25_only"]["governance_floor_pass"] is False
    assert rows["bm25_only"]["promotion_eligible"] is False
    assert "safety_or_source_governance_floor" in rows["bm25_only"]["promotion_blockers"]


def test_current_source_governed_stack_retained_but_not_promoted() -> None:
    payload = build_ai_trinity_tradeoff()
    rows = {row["configuration"]: row for row in payload["scenarios"]}
    current = rows[payload["current_operating_configuration"]]

    assert current["governance_floor_pass"] is True
    assert current["promotion_eligible"] is False
    assert payload["current_policy"]["dense_or_complex_retrieval_promoted"] is False
    assert payload["summary"]["improvement_proven_vs_bm25"] is False


def test_planning_cost_scenarios_never_become_observed_cost_evidence() -> None:
    payload = build_ai_trinity_tradeoff()

    assert payload["planning_only_route_cost_scenarios"]
    assert all(
        row["provider_billing_observed"] is False
        and row["promotion_eligible"] is False
        for row in payload["planning_only_route_cost_scenarios"]
    )


def test_writer_emits_machine_readable_artifact(tmp_path: Path) -> None:
    target = tmp_path / "trinity.json"
    payload = write_ai_trinity_tradeoff(target)
    stored = json.loads(target.read_text(encoding="utf-8"))

    assert stored["schema_version"] == "ai_trinity_tradeoff_v1"
    assert stored["decision"] == payload["decision"]
    assert stored["production_slo"] is False


def test_policy_weights_sum_to_one() -> None:
    policy = json.loads(
        (ROOT / "config/ai_trinity_policy.json").read_text(encoding="utf-8")
    )

    assert round(sum(policy["retrieval_quality_weights"].values()), 8) == 1.0
