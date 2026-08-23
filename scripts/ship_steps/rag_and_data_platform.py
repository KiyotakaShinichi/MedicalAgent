"""Knowledge-base materialisation, RAG index/baselines, and the non-patient
data platform, including managed-vector and Azure readiness probes.

Extracted from ``scripts/ship.py`` as part of splitting a 477-line
``_build_steps``. Step definitions are relocated verbatim: the command, working
directory, environment, and timeout of every step are unchanged, and the order
within and across these modules reproduces the original list exactly.
"""

from __future__ import annotations

import sys

from scripts.ship_steps.common import Step

__all__ = ["rag_and_data_platform_steps"]


def rag_and_data_platform_steps() -> list[Step]:
    return [
                Step(
                    name="Reproducible knowledge-base chunk materialization",
                    command=[sys.executable, "scripts/ingest_knowledge_base.py", "--skip-index"],
                ),
                Step(
                    name="Fingerprint-matched local RAG index rebuild",
                    command=[sys.executable, "scripts/build_rag_index.py"],
                    timeout_seconds=600,
                ),
                Step(
                    name="Frozen RAG baseline comparison refresh",
                    command=[sys.executable, "scripts/run_rag_baseline_comparison.py"],
                    timeout_seconds=900,
                ),
                Step(
                    name="Canonical RAG governance tradeoff refresh",
                    command=[sys.executable, "scripts/run_rag_governance_tradeoff.py"],
                ),
                Step(
                    name="Non-patient data-platform pipeline",
                    command=[sys.executable, "scripts/run_data_platform_pipeline.py"],
                ),
                Step(
                    name="Managed-vector contract evaluation",
                    command=[sys.executable, "scripts/run_vector_store_contract_eval.py"],
                ),
                Step(
                    name="Data-platform reliability drills",
                    command=[sys.executable, "scripts/run_data_platform_reliability_eval.py"],
                ),
                Step(
                    name="Azure AI Search index readiness",
                    command=[sys.executable, "scripts/run_azure_search_index_readiness.py"],
                ),
                Step(
                    name="Managed-vector shadow sync readiness",
                    command=[sys.executable, "scripts/run_managed_vector_shadow_sync.py"],
                ),
                Step(
                    name="Managed-vector frozen shadow comparison readiness",
                    command=[sys.executable, "scripts/run_managed_vector_shadow_comparison.py"],
                ),
                Step(
                    name="Azure compiled reference-infrastructure readiness",
                    command=[sys.executable, "scripts/run_cloud_infrastructure_readiness.py"],
                ),
    ]
