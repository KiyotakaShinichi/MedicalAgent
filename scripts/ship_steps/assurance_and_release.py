"""Cross-domain assurance dossiers, benchmark refreshes, and the release
decision surface and artifact gate.

Extracted from ``scripts/ship.py`` as part of splitting a 477-line
``_build_steps``. Step definitions are relocated verbatim: the command, working
directory, environment, and timeout of every step are unchanged, and the order
within and across these modules reproduces the original list exactly.
"""

from __future__ import annotations

import sys

from scripts.ship_steps.common import Step

__all__ = ["assurance_and_release_steps"]


def assurance_and_release_steps() -> list[Step]:
    return [
                Step(
                    name="Canonical proxy-removed synthetic feature policy",
                    command=[sys.executable, "scripts/run_synthetic_feature_policy.py"],
                ),
                Step(
                    name="Synthetic ML perturbation and retraining stress",
                    command=[
                        sys.executable,
                        "scripts/run_synthetic_model_perturbation_retrain_eval.py",
                    ],
                    timeout_seconds=900,
                ),
                Step(
                    name="Disposable synthetic staging readiness",
                    command=[
                        sys.executable,
                        "scripts/run_disposable_synthetic_staging_readiness.py",
                    ],
                ),
                Step(
                    name="Synthetic staging resilience dossier",
                    command=[
                        sys.executable,
                        "scripts/run_synthetic_staging_resilience_dossier.py",
                    ],
                ),
                Step(
                    name="Fine-tune hardening assurance",
                    command=[sys.executable, "scripts/run_finetune_hardening_assurance.py"],
                ),
                Step(
                    name="Fail-closed RAG release assurance",
                    command=[sys.executable, "scripts/run_fail_closed_rag_assurance.py"],
                    timeout_seconds=240,
                ),
                Step(
                    name="Restricted synthetic staging assurance",
                    command=[
                        sys.executable,
                        "scripts/run_restricted_synthetic_staging_assurance.py",
                    ],
                    timeout_seconds=240,
                ),
                Step(
                    name="Required safety benchmark refresh",
                    command=[sys.executable, "scripts/run_safety_benchmark.py"],
                    timeout_seconds=300,
                ),
                Step(
                    name="Required adversarial benchmark refresh",
                    command=[sys.executable, "scripts/run_adversarial_benchmark.py"],
                    timeout_seconds=300,
                ),
                Step(
                    name="Required RAG benchmark refresh",
                    command=[sys.executable, "scripts/run_rag_benchmark.py"],
                    timeout_seconds=600,
                ),
                Step(
                    name="Required intent-aware RAG benchmark refresh",
                    command=[sys.executable, "scripts/run_rag_intent_aware_eval.py"],
                    timeout_seconds=600,
                ),
                Step(
                    name="Cross-domain evidence maturity matrix",
                    command=[sys.executable, "scripts/run_evidence_maturity_matrix.py"],
                ),
                Step(
                    name="Canonical credibility-gap registry",
                    command=[sys.executable, "scripts/run_credibility_gap_registry.py"],
                ),
                Step(
                    name="Cross-domain composed assurance drill",
                    command=[sys.executable, "scripts/run_cross_domain_assurance_eval.py"],
                ),
                Step(
                    name="Senior engineering evidence dossier",
                    command=[sys.executable, "scripts/run_senior_engineering_evidence.py"],
                ),
                Step(
                    name="Canonical release decision surface",
                    command=[sys.executable, "scripts/run_release_decision_surface.py"],
                ),
                Step(
                    name="Constraint-aware cross-domain improvement program",
                    command=[sys.executable, "scripts/run_constraint_aware_improvement_program.py"],
                ),
                # The consolidated registry treats MLE readiness as critical
                # evidence, and docs/ci_cd.md lists "MLE readiness hard gates
                # must pass" as a release gate - but Ship never produced it, so
                # on a clean runner the artifact was simply absent and the
                # registry blocked on a file this pipeline had no way to create.
                # ci.yml and run_quality_gate.py already run this exact command;
                # it reads tracked inputs, needs no network or model, and does
                # not touch the locked holdout it reports on.
                #
                # Placed immediately before the registry so it runs after every
                # step that may refresh the evidence it aggregates.
                Step(
                    name="MLE readiness gate",
                    command=[sys.executable, "scripts/run_mle_checks.py"],
                ),
                Step(
                    name="Consolidated benchmark registry",
                    command=[sys.executable, "scripts/generate_benchmark_report.py"],
                ),
                Step(
                    name="Evidence-backed service health snapshot",
                    command=[sys.executable, "scripts/run_ops_health_snapshot.py"],
                ),
                Step(
                    name="Focused release summary",
                    command=[sys.executable, "scripts/run_focused_release_summary.py"],
                ),
                Step(
                    name="Release artifact gate",
                    command=[sys.executable, "scripts/run_release_gate.py"],
                ),
    ]
