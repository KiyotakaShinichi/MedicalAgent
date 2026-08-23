"""Backend pytest suites that gate the ship run.

Extracted from ``scripts/ship.py`` as part of splitting a 477-line
``_build_steps``. Step definitions are relocated verbatim: the command, working
directory, environment, and timeout of every step are unchanged, and the order
within and across these modules reproduces the original list exactly.
"""

from __future__ import annotations

import sys

from scripts.ship_steps.common import Step

__all__ = ["backend_tests_steps"]


def backend_tests_steps() -> list[Step]:
    return [
                Step(
                    name="Backend breast-monitoring integration tests",
                    command=[sys.executable, "-m", "pytest", "tests/test_breast_monitoring.py", "-q"],
                    env={
                        "RAG_FORCE_SPARSE": "true",
                        "ONCOTRACK_FAST_MODE": "true",
                    },
                ),
                Step(
                    name="Backend progressive-loading and notification reliability tests",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_patient_progressive_report.py",
                        "tests/test_patient_report_enrichment_jobs.py",
                        "tests/test_high_risk_conversation_alerts.py",
                        "tests/test_n8n_webhook_dispatcher.py",
                        "tests/test_n8n_automation_templates.py",
                        "-q",
                    ],
                ),
                Step(
                    name="Cloud, data-platform, and managed-vector contract tests",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_managed_vector_store.py",
                        "tests/test_data_platform_pipeline.py",
                        "tests/test_cloud_infrastructure_readiness.py",
                        "tests/test_vector_store_contract_eval.py",
                        "tests/test_azure_search_index_admin.py",
                        "tests/test_managed_vector_shadow_sync.py",
                        "tests/test_managed_vector_shadow_comparison.py",
                        "tests/test_data_platform_reliability_eval.py",
                        "tests/test_ops_health_snapshot.py",
                        "tests/test_release_decision_surface.py",
                        "tests/test_constraint_aware_improvement_program.py",
                        "tests/test_oidc_pkce.py",
                        "tests/test_dependency_security_scan.py",
                        "tests/test_container_security_scan.py",
                        "tests/test_container_runtime_hardening.py",
                        "tests/test_software_supply_chain_evidence.py",
                        "tests/test_synthetic_automation_staging_readiness.py",
                        "tests/test_rag_degradation_resilience_eval.py",
                        "tests/test_agent_execution_policy_eval.py",
                        "tests/test_agent_execution_policy.py",
                        "tests/test_agentic_orchestrator_and_verifier.py",
                        "tests/test_synthetic_prediction_statistical_audit.py",
                        "tests/test_patient_xai_readability_dossier.py",
                        "-q",
                    ],
                ),
                Step(
                    name="Assurance, XAI, automation, and safety contract tests",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_ship_runner.py",
                        "tests/test_kb_research_provenance.py",
                        "tests/test_research_paper_kb_eval.py",
                        "tests/test_governance_credibility_artifacts.py",
                        "tests/test_xai_retraining_stability_audit.py",
                        "tests/test_xai_rank_stability_audit.py",
                        "tests/test_credible_route_latency_sample.py",
                        "tests/test_automation_channel_drill.py",
                        "tests/test_adversarial_v6_tuning_regression.py",
                        "tests/test_unsafe_intent_mutation_dev_eval.py",
                        "tests/test_cross_domain_assurance_eval.py",
                        "tests/test_senior_engineering_evidence.py",
                        "tests/test_llm_usage_telemetry.py",
                        "tests/test_finetune_promotion.py",
                        "tests/test_finetune_semantic_contamination.py",
                        "tests/test_finetune_hardening_assurance.py",
                        "tests/test_rag_paired_statistical_comparison.py",
                        "tests/test_xai_reliability_gate.py",
                        "tests/test_patient_xai_envelope.py",
                        "tests/test_evidence_maturity_matrix.py",
                        "tests/test_credibility_gap_registry.py",
                        "tests/test_rag_vector_runtime_cache.py",
                        "tests/test_retrieval_runtime_cache_eval.py",
                        "tests/test_provider_usage_reconciliation.py",
                        "tests/test_provider_usage_capture_readiness.py",
                        "tests/test_finetune_contamination_adjudication.py",
                        "tests/test_synthetic_feature_policy.py",
                        "tests/test_synthetic_model_perturbation_retrain_eval.py",
                        "tests/test_disposable_synthetic_staging_readiness.py",
                        "tests/test_synthetic_staging_resilience_dossier.py",
                        "tests/test_adversarial_holdout_v7.py",
                        "tests/test_fail_closed_rag_assurance.py",
                        "tests/test_deployment_worker_wiring.py",
                        "tests/test_restricted_synthetic_staging_hardening.py",
                        "tests/test_restricted_synthetic_staging_assurance.py",
                        "-q",
                    ],
                ),
                Step(
                    name="SaaS control-plane and worker contract tests",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_saas_control_plane.py",
                        "tests/test_saas_workers.py",
                        "tests/test_saas_foundation_readiness.py",
                        "tests/test_saas_platform_api.py",
                        "tests/test_api_security_hardening.py",
                        "tests/test_model_route_authorization.py",
                        "tests/test_synthetic_demo_bootstrap.py",
                        "tests/test_multimodal_evidence_abstention.py",
                        "-q",
                    ],
                ),
                Step(
                    name="Research abstention and scale-evaluation contract tests",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_rag_runtime_prewarm.py",
                        "tests/test_research_evidence_answerability.py",
                        "tests/test_prototype_independent_prompt_bank_v2.py",
                        "tests/test_real_pipeline_scale_eval.py",
                        "tests/test_mixed_query_scale_eval.py",
                        "tests/test_adversarial_generalization_label_audit.py",
                        "-q",
                    ],
                ),
    ]
