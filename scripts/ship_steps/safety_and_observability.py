"""Adversarial and XAI safety regressions plus latency, cost, and cache
observability refreshes.

Extracted from ``scripts/ship.py`` as part of splitting a 477-line
``_build_steps``. Step definitions are relocated verbatim: the command, working
directory, environment, and timeout of every step are unchanged, and the order
within and across these modules reproduces the original list exactly.
"""

from __future__ import annotations

import sys

from scripts.ship_steps.common import Step

__all__ = ["safety_and_observability_steps"]


def safety_and_observability_steps() -> list[Step]:
    return [
                Step(
                    name="Development unsafe-intent mutation regression",
                    command=[sys.executable, "scripts/run_unsafe_intent_mutation_dev_eval.py"],
                ),
                Step(
                    name="Tuning-informed adversarial v6 regression",
                    command=[sys.executable, "scripts/run_adversarial_v6_tuning_regression.py"],
                ),
                Step(
                    name="Adversarial v6 contamination retrospective",
                    command=[sys.executable, "scripts/run_adversarial_v6_retrospective.py"],
                ),
                Step(
                    name="Synthetic XAI rank-stability audit",
                    command=[sys.executable, "scripts/run_xai_rank_stability_audit.py"],
                ),
                Step(
                    name="Synthetic XAI retraining-stability audit",
                    command=[sys.executable, "scripts/run_xai_retraining_stability_audit.py"],
                ),
                Step(
                    name="Synthetic XAI mechanical fidelity audit",
                    command=[sys.executable, "scripts/run_xai_fidelity_audit.py"],
                ),
                Step(
                    name="Fail-closed synthetic XAI presentation policy",
                    command=[sys.executable, "scripts/run_xai_reliability_gate.py"],
                ),
                Step(
                    name="Bounded agent execution-policy eval",
                    command=[sys.executable, "scripts/run_agent_execution_policy_eval.py"],
                ),
                Step(
                    name="Local RAG degradation resilience drill",
                    command=[sys.executable, "scripts/run_rag_degradation_resilience_eval.py"],
                ),
                Step(
                    name="Credible local route-latency sample",
                    command=[sys.executable, "scripts/run_credible_route_latency_sample.py"],
                ),
                Step(
                    name="Route-latency budget refresh",
                    command=[sys.executable, "scripts/run_route_latency_budget.py"],
                ),
                Step(
                    name="Token, cost, and stage-latency observability refresh",
                    command=[sys.executable, "scripts/run_cost_latency_report.py"],
                ),
                Step(
                    name="Retrieval runtime-cache regression evidence",
                    command=[sys.executable, "scripts/run_retrieval_runtime_cache_eval.py"],
                ),
                Step(
                    name="Guarded normal-API provider usage probe",
                    command=[sys.executable, "scripts/run_provider_api_path_capture.py"],
                ),
                Step(
                    name="Provider-token reconciliation",
                    command=[sys.executable, "scripts/run_provider_usage_reconciliation.py"],
                ),
                Step(
                    name="Provider-usage capture readiness",
                    command=[
                        sys.executable,
                        "scripts/run_provider_usage_capture_readiness.py",
                    ],
                ),
    ]
