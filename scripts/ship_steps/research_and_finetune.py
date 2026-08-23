"""Research-paper retrieval evidence, provider usage, and fine-tune promotion
and contamination gates.

Extracted from ``scripts/ship.py`` as part of splitting a 477-line
``_build_steps``. Step definitions are relocated verbatim: the command, working
directory, environment, and timeout of every step are unchanged, and the order
within and across these modules reproduces the original list exactly.
"""

from __future__ import annotations

import sys

from scripts.ship_steps.common import Step

__all__ = ["research_and_finetune_steps"]


def research_and_finetune_steps() -> list[Step]:
    return [
                Step(
                    name="Paired RAG statistical comparison",
                    command=[sys.executable, "scripts/run_rag_paired_statistical_comparison.py"],
                ),
                Step(
                    name="Research-paper KB provenance and retrieval evaluation",
                    command=[sys.executable, "scripts/run_research_paper_kb_eval.py"],
                    timeout_seconds=600,
                ),
                Step(
                    name="Research-paper per-query token and latency telemetry",
                    command=[sys.executable, "scripts/run_research_paper_query_telemetry.py"],
                    timeout_seconds=600,
                ),
                Step(
                    name="Claim-conditioned citation selector offline evaluation",
                    command=[
                        sys.executable,
                        "scripts/run_claim_conditioned_citation_selector_eval.py",
                    ],
                ),
                Step(
                    name="Frozen claim-conditioned selector holdout",
                    command=[
                        sys.executable,
                        "scripts/run_claim_conditioned_citation_selector_holdout.py",
                    ],
                ),
                Step(
                    name="Accuracy-latency-unit-cost tradeoff refresh",
                    command=[sys.executable, "scripts/run_ai_trinity_tradeoff.py"],
                ),
                Step(
                    name="Signed localhost automation channel drill",
                    command=[sys.executable, "scripts/run_automation_channel_drill.py"],
                ),
                Step(
                    name="Synthetic n8n and MailHog staging readiness",
                    command=[sys.executable, "scripts/run_synthetic_automation_staging_readiness.py"],
                ),
                Step(
                    name="Fine-tune promotion evidence gate",
                    command=[sys.executable, "scripts/run_finetune_promotion_gate.py"],
                ),
                Step(
                    name="Fine-tune semantic contamination screen",
                    command=[sys.executable, "scripts/run_finetune_semantic_contamination.py"],
                ),
                Step(
                    name="Fine-tune contamination adjudication readiness",
                    command=[
                        sys.executable,
                        "scripts/run_finetune_contamination_adjudication.py",
                    ],
                ),
    ]
