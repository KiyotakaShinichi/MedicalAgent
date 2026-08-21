"""Source-tier retrieval ablation.

Runs the intent-aware benchmark four times under different tier filters
— **T1 only**, **T1+T2**, **T1+T2+T3**, **all** — and reports per-tier-config
metrics so a reviewer can see the coverage/quality trade-off.

Sample question this surfaces: "If I restrict retrieval to T1 guidelines
only, do I get higher citation precision at the cost of more refusals?"

This is engineering instrumentation, not a clinical claim — it tells you
how the *retrieval policy* changes downstream metrics on the curated
case set; it does not establish clinical preference for one tier over
another.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.services.rag_intent_aware_eval import EVAL_CASES, run_intent_aware_eval


DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_rag_tier_ablation.json"


# The four tier configurations we sweep.  Lower-tier inclusion is
# monotone: each step ADDS a tier.
TIER_CONFIGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("t1_only",        ("T1",)),
    ("t1_t2",          ("T1", "T2")),
    ("t1_t2_t3",       ("T1", "T2", "T3")),
    ("all_tiers",      ("T1", "T2", "T3", "T4")),
)


# Caller supplies a factory that, given the allowed-tier tuple, returns
# an agent callable scoped to those tiers.  Decoupling the ablation
# harness from the live agent stack keeps it testable + avoids hard
# coupling to retrieval implementation.

AgentFactory = Callable[[tuple[str, ...]], Callable[[str], dict]]


def run_tier_ablation(
    *,
    agent_factory: AgentFactory,
    cases=EVAL_CASES,
    tier_configs=TIER_CONFIGS,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Sweep every tier config, run the eval, collect per-config metrics."""
    per_config: list[dict[str, Any]] = []
    for config_name, allowed_tiers in tier_configs:
        agent = agent_factory(allowed_tiers)
        # Use a per-config output path so the harness doesn't clobber
        # the main intent-aware eval artifact.
        sub_path = Path(output_path).parent / f"_tier_{config_name}_eval.json"
        eval_payload = run_intent_aware_eval(
            agent=agent,
            cases=cases,
            output_path=str(sub_path),
        )
        summary = eval_payload.get("summary") or {}
        per_config.append({
            "config": config_name,
            "allowed_tiers": list(allowed_tiers),
            "pass_rate":                summary.get("pass_rate"),
            "claim_support_rate":       summary.get("claim_support_rate"),
            "citation_precision":       summary.get("citation_precision"),
            "source_tier_correctness":  summary.get("source_tier_correctness"),
            "refusal_correctness":      summary.get("refusal_correctness"),
            "unsafe_answer_rate":       summary.get("unsafe_answer_rate"),
            "post_gen_validator_trigger_rate": summary.get("post_gen_validator_trigger_rate"),
            "latency_p50_ms":           summary.get("latency_p50_ms"),
            "grade_distribution":       eval_payload.get("grade_distribution") or {},
        })

    payload = {
        "schema_version": "rag_tier_ablation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status(per_config),
        "tier_configs_evaluated": [c[0] for c in tier_configs],
        "per_config": per_config,
        "interpretation": (
            "Each row shows the eval metrics under a different tier "
            "filter. Higher refusal_correctness with lower pass_rate at "
            "the T1-only end is expected — narrower retrieval refuses "
            "more often. The right config depends on which metric the "
            "deployment prioritises (precision vs coverage)."
        ),
        "claim_boundary": (
            "Engineering instrumentation only. Does not establish "
            "clinical preference for any tier configuration."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _status(per_config: list[dict[str, Any]]) -> str:
    """Status reflects whether *any* config achieves an acceptable
    pass_rate with zero unsafe answers — the ablation succeeds when at
    least one tier policy is viable."""
    viable = [
        c for c in per_config
        if (c.get("unsafe_answer_rate") or 0) == 0
        and (c.get("pass_rate") or 0) >= 0.70
    ]
    if any((c.get("pass_rate") or 0) >= 0.90 for c in viable):
        return "strong"
    if viable:
        return "acceptable"
    return "needs_attention"


def load_tier_ablation(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "rag_tier_ablation_v1",
            "status": "missing",
            "message": (
                "Tier ablation has not been generated yet. "
                "Run `scripts/run_rag_tier_ablation.py`."
            ),
            "per_config": [],
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "TIER_CONFIGS",
    "load_tier_ablation",
    "run_tier_ablation",
]
