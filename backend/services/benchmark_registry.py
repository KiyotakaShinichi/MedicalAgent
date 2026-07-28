from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.artifact_manifest import build_artifact_manifest, freshness_status


DEFAULT_JSON_PATH = "Data/evals/benchmark/latest_benchmark_summary.json"
DEFAULT_MD_PATH = "benchmarks/benchmark_report.md"
DEFAULT_CSV_PATH = "benchmarks/benchmark_results.csv"

ROOT_DIR = Path(__file__).resolve().parents[2]


BENCHMARK_SPECS: list[dict[str, Any]] = [
    {
        "id": "safety_red_team",
        "title": "Safety red-team",
        "path": "Data/evals/safety/latest_safety_benchmark.json",
        "fallback": "Data/evals/safety/latest_safety_red_team.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "failed_cases": ["summary", "failed_cases"],
            "total_cases": ["summary", "total_cases"],
        },
    },
    {
        "id": "adversarial",
        "title": "Adversarial prompt/jailbreak",
        "path": "Data/evals/safety/latest_adversarial_eval.json",
        "tier": "critical",
        "metrics": {
            "attack_block_rate": ["summary", "pass_rate"],
            "failed_cases": ["summary", "failed_cases"],
        },
    },
    {
        "id": "adversarial_safety_regression",
        "title": "Fixed-bank adversarial safety regression",
        "path": "Data/evals/safety/latest_adversarial_safety_regression.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "total_cases": ["total_cases"],
            "overall_attack_block_rate": ["overall_attack_block_rate"],
            "hard_gate_passed": ["hard_gate", "passed"],
            "safe_answer_rate": ["metrics", "safe_answer_rate"],
        },
    },
    {
        "id": "adversarial_failure_analysis",
        "title": "Adversarial failure analysis",
        "path": "Data/evals/safety/latest_adversarial_failure_analysis.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "focus_case_count": ["summary", "focus_case_count"],
            "focus_failed_count": ["summary", "focus_failed_count"],
            "focus_pass_rate": ["summary", "focus_pass_rate"],
        },
    },
    {
        "id": "adversarial_holdout_monitor",
        "title": "Adversarial dev/holdout monitor",
        "path": "Data/evals/safety/latest_adversarial_safety_regression_holdout.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dev_pass_rate": ["dev", "pass_rate"],
            "holdout_pass_rate": ["holdout", "pass_rate"],
            "holdout_n": ["holdout", "total_n"],
        },
    },
    {
        "id": "adversarial_generalization_eval",
        "title": "Adversarial generalization eval",
        "path": "Data/evals/safety/latest_adversarial_generalization_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "original_bank_pass_rate": ["metrics", "original_bank_pass_rate"],
            "heldout_pass_rate": ["metrics", "heldout_pass_rate"],
            "paraphrase_pass_rate": ["metrics", "paraphrase_pass_rate"],
            "safe_negative_control_pass_rate": ["metrics", "safe_negative_control_pass_rate"],
            "category_gap_between_dev_and_holdout": ["metrics", "category_gap_between_dev_and_holdout"],
        },
    },
    {
        "id": "adversarial_generalization_v2_eval",
        "title": "Adversarial generalization eval v2",
        "path": "Data/evals/safety/latest_adversarial_generalization_v2_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "original_bank_pass_rate": ["metrics", "original_bank_pass_rate"],
            "heldout_v1_pass_rate": ["metrics", "heldout_v1_pass_rate"],
            "heldout_v2_pass_rate": ["metrics", "heldout_v2_pass_rate"],
            "paraphrase_pass_rate": ["metrics", "paraphrase_pass_rate"],
            "safe_negative_control_pass_rate": ["metrics", "safe_negative_control_pass_rate"],
            "unsafe_leakage_rate": ["metrics", "unsafe_leakage_rate"],
        },
    },
    {
        "id": "heldout_adversarial_failure_analysis",
        "title": "Held-out adversarial failure analysis",
        "path": "Data/evals/safety/latest_heldout_adversarial_failure_analysis.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "total_n": ["total_n"],
            "failed_n": ["failed_n"],
            "failure_rate": ["failure_rate"],
        },
    },
    {
        "id": "unsafe_intent_classifier_eval",
        "title": "Unsafe-intent semantic classifier eval",
        "path": "Data/evals/safety/latest_unsafe_intent_classifier_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "total_n": ["total_n"],
            "pass_rate": ["pass_rate"],
            "fail_count": ["fail_count"],
        },
    },
    {
        "id": "multilingual_refusal",
        "title": "Multilingual refusal routing",
        "path": "Data/evals/safety/latest_multilingual_refusal_eval.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "passed": ["summary", "passed"],
            "case_count": ["summary", "case_count"],
        },
    },
    {
        "id": "rag_regression",
        "title": "RAG regression",
        "path": "Data/evals/rag/latest_rag_benchmark.json",
        "fallback": "Data/evals/rag/latest_rag_eval.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "citation_coverage_rate": ["summary", "citation_coverage_rate"],
            "expected_source_hit_rate": ["summary", "expected_source_hit_rate"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
            "average_grounding_score": ["summary", "average_grounding_score"],
        },
    },
    {
        "id": "rag_governance_tradeoff",
        "title": "RAG effectiveness and governance trade-off",
        "path": "Data/evals/rag/latest_rag_governance_tradeoff.json",
        "tier": "supporting",
        "metrics": {
            "recall_delta": ["tradeoffs", "full_minus_bm25_recall_at_10"],
            "source_tier_delta": ["tradeoffs", "full_minus_bm25_source_tier_correctness"],
            "latency_ratio": ["tradeoffs", "full_vs_bm25_latency_p95_ratio"],
            "improvement_proven": ["improvement_proven_vs_bm25"],
            "external_holdout_completed": ["external_holdout", "completed"],
        },
    },
    {
        "id": "rag_gold",
        "title": "Hand-labeled RAG gold set",
        "path": "Data/evals/rag/latest_rag_gold_eval.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "expected_source_hit_rate": ["summary", "expected_source_hit_rate"],
            "case_count": ["gold_set", "case_count"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
        },
    },
    {
        "id": "tool_action_benchmark",
        "title": "Patient-support tool action benchmark",
        "path": "Data/evals/tool_actions/latest_tool_action_benchmark.json",
        "tier": "critical",
        "metrics": {
            "pass_rate": ["summary", "pass_rate"],
            "case_count": ["summary", "case_count"],
            "average_latency_ms": ["summary", "average_latency_ms"],
            "max_latency_ms": ["summary", "max_latency_ms"],
        },
    },
    {
        "id": "genetic_counseling_readiness",
        "title": "Genetic counseling readiness safety",
        "path": "Data/evals/genetics/latest_genetic_counseling_eval.json",
        "tier": "critical",
        "metrics": {
            "genetic_overclaim_rate": ["metrics", "genetic_overclaim_rate"],
            "treatment_advice_leakage": ["metrics", "treatment_advice_leakage"],
            "tumor_marker_overclaim_rate": ["metrics", "tumor_marker_overclaim_rate"],
            "vus_correctness": ["metrics", "VUS_handling_correctness"],
            "referral_correctness": ["metrics", "referral_correctness"],
        },
    },
    {
        "id": "biomarker_feature_benchmark",
        "title": "Biomarker and tumor-marker feature ablation",
        "path": "Data/mle_monitoring/biomarker_feature_benchmark.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "biomarker_vs_clinical_auroc_delta": ["deltas", "biomarker_vs_clinical_auroc_delta"],
            "biomarker_imaging_vs_clinical_auroc_delta": ["deltas", "biomarker_imaging_vs_clinical_auroc_delta"],
            "enhanced_vs_current_default_auroc_delta": ["deltas", "enhanced_vs_current_default_auroc_delta"],
            "leakage_status": ["leakage_audit", "status"],
            "recommendation": ["recommendation", "decision"],
        },
    },
    {
        "id": "full_feature_group_ablation",
        "title": "Full modality feature-group ablation",
        "path": "Data/evals/models/latest_full_feature_group_ablation.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "full_vs_clinical_auroc_delta": ["deltas", "full_vs_clinical_auroc_delta"],
            "full_vs_clinical_brier_delta": ["deltas", "full_vs_clinical_brier_delta"],
            "full_vs_clinical_ece_delta": ["deltas", "full_vs_clinical_ece_delta"],
            "recommended_use": ["recommendation", "recommended_use"],
            "leakage_status": ["leakage_audit", "status"],
        },
    },
    {
        "id": "toxicity_shortcut_audit",
        "title": "Toxicity label shortcut audit",
        "path": "Data/evals/models/latest_toxicity_shortcut_audit.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "rule_accuracy": ["rule_reconstruction", "accuracy"],
            "rule_auroc": ["rule_reconstruction", "auroc"],
            "direct_rule_reconstruction": ["rule_reconstruction", "direct_rule_reconstruction"],
            "recommended_use": ["recommendation", "use"],
        },
    },
    {
        "id": "learned_abstention_experiment",
        "title": "Learned abstention-head experiment",
        "path": "Data/evals/models/latest_learned_abstention_experiment.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "auroc": ["abstention_head", "auroc"],
            "learned_coverage": ["comparison", "learned", "coverage_rate"],
            "rule_based_coverage": ["comparison", "rule_based", "coverage_rate"],
        },
    },
    {
        "id": "soft_toxicity_target_benchmark",
        "title": "Softer synthetic toxicity target benchmark",
        "path": "Data/evals/models/latest_soft_toxicity_target_benchmark.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "auroc": ["soft_target_model", "auroc"],
            "old_rule_accuracy_against_soft_label": ["shortcut_comparison", "old_toxicity_rule_accuracy_against_soft_label"],
            "positive_rate": ["label_design", "positive_rate"],
        },
    },
    {
        "id": "hybrid_subgroup_metrics",
        "title": "Hybrid prediction subgroup metrics",
        "path": "Data/evals/models/latest_hybrid_subgroup_metrics.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "n": ["overall", "n"],
            "classification_coverage": ["overall", "classification_coverage"],
            "regression_coverage": ["overall", "regression_coverage"],
            "toxicity_coverage": ["overall", "toxicity_coverage"],
        },
    },
    {
        "id": "synthetic_realism_hardening",
        "title": "Synthetic realism hardening patterns",
        "path": "Data/evals/models/latest_synthetic_realism_hardening.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "checks_passed": ["summary", "checks_passed"],
            "checks_total": ["summary", "checks_total"],
            "rows": ["summary", "rows"],
        },
    },
    {
        "id": "self_supervised_timeline",
        "title": "Self-supervised synthetic timeline pretraining",
        "path": "Data/evals/models/latest_self_supervised_timeline.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "masked_lab_mae": ["metrics", "masked_lab_mae"],
            "masked_symptom_f1": ["metrics", "masked_symptom_f1"],
            "masked_imaging_signal_accuracy": ["metrics", "masked_imaging_signal_accuracy"],
            "leakage_check_status": ["metrics", "leakage_check_status"],
        },
    },
    {
        "id": "counterfactual_stability",
        "title": "Counterfactual stability",
        "path": "Data/evals/models/latest_counterfactual_stability.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "scenario_count": ["summary", "scenario_count"],
            "unacceptable_flip_count": ["summary", "unacceptable_flip_count"],
            "max_probability_delta": ["summary", "max_probability_delta"],
        },
    },
    {
        "id": "learned_abstention",
        "title": "Learned abstention head",
        "path": "Data/evals/models/latest_learned_abstention.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "auroc": ["abstention_head", "auroc"],
            "learned_coverage": ["comparison", "learned", "coverage_rate"],
            "rule_based_coverage": ["comparison", "rule_based", "coverage_rate"],
        },
    },
    {
        "id": "per_head_calibration",
        "title": "Per-head hybrid calibration",
        "path": "Data/evals/models/latest_per_head_calibration.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "classification_brier": ["heads", "response_classification", "brier"],
            "classification_ece": ["heads", "response_classification", "ece"],
            "toxicity_brier": ["heads", "toxicity", "brier"],
            "toxicity_ece": ["heads", "toxicity", "ece"],
        },
    },
    {
        "id": "uncertainty_dossier",
        "title": "Uncertainty dossier",
        "path": "Data/evals/models/latest_uncertainty_dossier.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "synthetic_only": ["synthetic_only"],
            "clinical_validation": ["clinical_validation"],
        },
    },
    {
        "id": "real_data_readiness_checklist",
        "title": "Real-data readiness checklist",
        "path": "Data/evals/models/latest_real_data_readiness_checklist.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "completed_count": ["completed_count"],
            "required_count": ["required_count"],
        },
    },
    {
        "id": "clinical_performance_dossier_status",
        "title": "Clinical performance dossier template status",
        "path": "Data/evals/models/latest_clinical_performance_dossier_status.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "clinical_validation": ["current_status", "clinical_validation"],
            "treatment_decision_influence": ["current_status", "treatment_decision_influence"],
        },
    },
    {
        "id": "shortcut_audit",
        "title": "Hybrid shortcut audit",
        "path": "Data/evals/models/latest_shortcut_audit.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "shortcut_audit_status": ["status"],
            "near_label_proxy_risk": ["toxicity_audit", "near_label_proxy_risk"],
            "review_hint_only": ["toxicity_audit", "review_hint_only"],
            "regression_mae_increase_without_mri": ["regression_audit", "mae_increase_without_mri_percent_change"],
        },
    },
    {
        "id": "medical_advisor_review_packet",
        "title": "Unreviewed clinical advisor packet",
        "path": "Data/evals/medical/latest_medical_advisor_review_packet.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "interaction_rule_count": ["interaction_rule_count"],
        },
    },
    {
        "id": "minimum_evidence_controlled_doc",
        "title": "Minimum evidence controlled doc",
        "path": "Data/evals/medical/latest_minimum_evidence_controlled_doc.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "owner": ["owner"],
        },
    },
    {
        "id": "human_factors_risk_eval",
        "title": "Human-factors overtrust risk eval",
        "path": "Data/evals/medical/latest_human_factors_risk_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "advisory_workflow_readiness",
        "title": "Clinical advisory workflow readiness",
        "path": "Data/evals/medical/latest_advisory_workflow_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "minimum_evidence_standards",
        "title": "Minimum evidence standards",
        "path": "Data/evals/medical/latest_minimum_evidence_standards.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "version": ["version"],
        },
    },
    {
        "id": "medical_claim_boundary_eval",
        "title": "Medical claim-boundary eval",
        "path": "Data/evals/safety/latest_medical_claim_boundary_eval.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "pass_rate": ["summary", "pass_rate"],
            "case_count": ["summary", "case_count"],
        },
    },
    {
        "id": "public_biomarker_dataset_manifest",
        "title": "Public biomarker predictor-source manifest",
        "path": "Data/data_lineage/public_biomarker_dataset_manifest.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dataset_count": ["dataset_count"],
            "manifest_hash": ["manifest_hash"],
        },
    },
    {
        "id": "public_biomarker_mapping_readiness",
        "title": "Public biomarker mapping readiness",
        "path": "Data/mle_monitoring/public_biomarker_mapping_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "mapping_hash": ["mapping_hash"],
            "breastdcedl_status": ["datasets", "breastdcedl", "status"],
        },
    },
    {
        "id": "public_biomarker_dataset_readiness",
        "title": "Public biomarker/tumor-marker dataset readiness",
        "path": "Data/evals/models/latest_public_biomarker_dataset_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dataset_count": ["summary", "dataset_count"],
            "biomarker_external_candidate_count": ["summary", "biomarker_external_candidate_count"],
            "tumor_marker_response_train_ready": ["summary", "tumor_marker_response_train_ready"],
            "production_retrain_now": ["retraining_decision", "production_retrain_now"],
            "candidate_training_recommended": ["retraining_decision", "candidate_training_recommended"],
        },
    },
    {
        "id": "public_treatment_dataset_readiness",
        "title": "Public treatment-combination dataset readiness",
        "path": "Data/evals/models/latest_public_treatment_dataset_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dataset_count": ["summary", "dataset_count"],
            "treatment_combination_candidate_count": ["summary", "treatment_combination_candidate_count"],
            "immediate_full_treatment_combo_training_ready": ["summary", "immediate_full_treatment_combo_training_ready"],
            "best_future_real_world_treatment_dataset": ["summary", "best_future_real_world_treatment_dataset"],
        },
    },
    {
        "id": "deep_learning_candidate_benchmark",
        "title": "Deep-learning classification/regression candidate benchmark",
        "path": "Data/evals/models/latest_deep_learning_candidate_benchmark.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "best_model": ["best_model", "model"],
            "best_variant": ["best_model", "variant"],
            "classification_auroc": ["best_model", "classification_auroc"],
            "best_regression_model": ["best_models", "regression", "model"],
            "regression_mae_percent": ["best_models", "regression", "regression_mae_percent"],
            "genetic_context_decision": ["genetic_context_ablation", "decision"],
            "treatment_context_decision": ["treatment_context_ablation", "decision"],
        },
    },
    {
        "id": "canonical_oncology_schema",
        "title": "Canonical oncology schema bridge",
        "path": "Data/external_bridge/canonical_oncology_schema.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "schema_version": ["schema_version"],
        },
    },
    {
        "id": "external_data_bridge_eval",
        "title": "External public-data canonical bridge",
        "path": "Data/evals/models/latest_external_data_bridge_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "row_count": ["row_count"],
            "validation_status": ["validation", "status"],
            "breastdcedl_roc_auc": ["external_model_snapshot", "models", "logistic_regression", "roc_auc"],
        },
    },
    {
        "id": "ispy2_tcia_external_stress",
        "title": "TCIA I-SPY2 separate-task external stress benchmark",
        "path": "Data/evals/models/latest_ispy2_tcia_external_stress.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "joined_row_count": ["source", "joined_row_count"],
            "nlcare_target_match": ["task_boundary", "nlcare_target_match"],
            "used_for_nlcare_training": ["used_for_nlcare_training"],
            "patient_facing_allowed": ["patient_facing_allowed"],
            "promotion_allowed": ["promotion_allowed"],
        },
    },
    {
        "id": "duke_tcia_external_stress",
        "title": "Duke/TCIA separate-task external stress benchmark",
        "path": "Data/evals/models/latest_duke_tcia_external_stress.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "joined_labeled_row_count": ["source", "joined_labeled_row_count"],
            "nlcare_target_match": ["task_boundary", "nlcare_target_match"],
            "used_for_nlcare_training": ["used_for_nlcare_training"],
            "patient_facing_allowed": ["patient_facing_allowed"],
            "promotion_allowed": ["promotion_allowed"],
        },
    },
    {
        "id": "external_failure_case_gallery",
        "title": "External benchmark failure-case gallery",
        "path": "Data/evals/models/latest_external_failure_case_gallery.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "false_positive_count": ["summary", "false_positive_count"],
            "false_negative_count": ["summary", "false_negative_count"],
        },
    },
    {
        "id": "treatment_sequence_feature_eval",
        "title": "Synthetic treatment-sequence feature eval",
        "path": "Data/evals/models/latest_treatment_sequence_feature_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "patient_count": ["patient_count"],
            "pattern_count": ["pattern_count"],
            "chemotherapy_count": ["modality_counts", "chemotherapy"],
            "targeted_anti_her2_count": ["modality_counts", "targeted_anti_her2"],
        },
    },
    {
        "id": "data_promotion_roadmap",
        "title": "Data-to-promotion roadmap",
        "path": "Data/evals/models/latest_data_promotion_roadmap.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "model_head_count": ["summary", "model_head_count"],
            "current_global_policy": ["promotion_policy", "current_global_policy"],
            "may_influence_treatment": ["promotion_policy", "may_influence_treatment"],
        },
    },
    {
        "id": "tcga_metabric_canonical_mapping",
        "title": "TCGA/METABRIC canonical mapping",
        "path": "Data/evals/models/latest_tcga_metabric_canonical_mapping.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "mapped_dataset_count": ["mapped_dataset_count"],
        },
    },
    {
        "id": "strict_common_feature_ab_eval",
        "title": "Strict common-feature external A/B eval",
        "path": "Data/evals/models/latest_strict_common_feature_ab_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "synthetic_auc": ["datasets", "synthetic_patient_level", "metrics", "roc_auc"],
            "external_auc": ["datasets", "breastdcedl_spy1", "metrics", "roc_auc"],
            "promotion_allowed": ["ab_decision", "promotion_allowed"],
        },
    },
    {
        "id": "toxicity_review_target_v2",
        "title": "Toxicity review-priority target v2",
        "path": "Data/evals/models/latest_toxicity_review_target_v2.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "auroc": ["model", "auroc"],
            "legacy_rule_accuracy_against_v2": ["shortcut_comparison", "legacy_rule_accuracy_against_v2"],
            "legacy_rule_does_not_define_v2": ["shortcut_comparison", "legacy_rule_does_not_define_v2"],
        },
    },
    {
        "id": "toxicity_review_target_v3",
        "title": "Toxicity review-priority target v3",
        "path": "Data/evals/models/latest_toxicity_review_target_v3.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "auroc": ["model", "auroc"],
            "legacy_rule_auroc_against_v3": ["shortcut_comparison", "legacy_rule_auroc_against_v3"],
            "legacy_rule_does_not_define_v3": ["shortcut_comparison", "legacy_rule_does_not_define_v3"],
            "promotion_decision": ["recommendation", "promotion_decision"],
        },
    },
    {
        "id": "ml_coverage_risk_diagnostics",
        "title": "ML coverage/risk diagnostics",
        "path": "Data/evals/models/latest_ml_coverage_risk_diagnostics.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "scenario_count": ["scenario_count"],
            "minimum_required_abstention_rate": [
                "required_abstention_scenarios",
                "minimum_required_abstention_rate",
            ],
            "promotion_decision": ["promotion_decision"],
        },
    },
    {
        "id": "automation_reliability_dossier",
        "title": "Automation reliability dossier",
        "path": "Data/evals/ops/latest_automation_reliability_dossier.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "check_count": ["check_count"],
            "passed_count": ["passed_count"],
            "failed_required_count": ["failed_required_count"],
            "external_delivery_enabled_by_default": ["external_delivery_enabled_by_default"],
            "real_emergency_coverage_claim": ["real_emergency_coverage_claim"],
            "automation_center_requirement_count": ["automation_center_requirement_count"],
        },
    },
    {
        "id": "durable_automation_worker",
        "title": "Durable automation worker controls",
        "path": "Data/evals/ops/latest_durable_automation_worker_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "control_pass_rate": ["control_pass_rate"],
            "live_n8n_delivery_enabled": ["live_n8n_delivery_enabled"],
            "live_delivery_test_completed": ["live_delivery_test_completed"],
            "clinical_validation": ["clinical_validation"],
        },
    },
    {
        "id": "patient_xai_readability_dossier",
        "title": "Patient XAI readability dossier",
        "path": "Data/evals/governance/latest_patient_xai_readability_dossier.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "surface_count": ["surface_count"],
            "failed_check_count": ["failed_check_count"],
            "clinical_validation": ["clinical_validation"],
            "diagnostic_authority_claim": ["diagnostic_authority_claim"],
            "treatment_recommendation_claim": ["treatment_recommendation_claim"],
        },
    },
    {
        "id": "automation_xai_industry_alignment",
        "title": "Automation and XAI industry-alignment roadmap",
        "path": "Data/evals/governance/latest_automation_xai_industry_alignment.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "automation_control_count": ["automation_control_count"],
            "xai_control_count": ["xai_control_count"],
            "automation_live_delivery_enabled": ["automation_live_delivery_enabled"],
            "healthcare_production_ready": ["healthcare_production_ready"],
            "real_emergency_coverage_claim": ["real_emergency_coverage_claim"],
        },
    },
    {
        "id": "external_failure_case_analysis",
        "title": "External failure cases by subtype/confidence",
        "path": "Data/evals/models/latest_external_failure_case_analysis.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "failure_count": ["summary", "failure_count"],
            "high_confidence_failure_count": ["summary", "high_confidence_failure_count"],
        },
    },
    {
        "id": "restricted_data_access_packet",
        "title": "Restricted dataset access packet",
        "path": "Data/evals/models/latest_restricted_data_access_packet.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "cbioportal_clinical_export",
        "title": "cBioPortal TCGA/METABRIC clinical export",
        "path": "Data/evals/models/latest_cbioportal_clinical_export.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "row_count": ["combined", "row_count"],
            "validation_status": ["combined", "validation", "status"],
            "full_temporal_validation": ["combined", "coverage", "roles_supported", "full_oncotrack_temporal_validation"],
        },
    },
    {
        "id": "external_distribution_alignment",
        "title": "External distribution alignment",
        "path": "Data/evals/models/latest_external_distribution_alignment.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "synthetic_rows": ["cohort_sizes", "synthetic"],
            "cbioportal_rows": ["cohort_sizes", "cbioportal_tcga_metabric"],
        },
    },
    {
        "id": "student_constraint_elevation_plan",
        "title": "Student-constraint elevation plan",
        "path": "Data/evals/models/latest_student_constraint_elevation_plan.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "common_feature_transfer_stress",
        "title": "Common-feature transfer stress",
        "path": "Data/evals/models/latest_common_feature_transfer_stress.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "synthetic_auc": ["within_dataset_models", "synthetic_treatment_success", "roc_auc"],
            "breastdcedl_auc": ["within_dataset_models", "breastdcedl_pcr", "roc_auc"],
            "promotion_allowed": ["promotion_decision", "promotion_allowed"],
        },
    },
    {
        "id": "public_distribution_realism_candidate",
        "title": "Public-distribution synthetic realism candidate",
        "path": "Data/evals/models/latest_public_distribution_realism_candidate.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "rows": ["rows"],
            "patients": ["patients"],
            "production_replacement_allowed": ["realism_candidate_decision", "production_replacement_allowed"],
        },
    },
    {
        "id": "realism_candidate_ab_gate",
        "title": "Current vs public-distribution realism candidate A/B gate",
        "path": "Data/evals/models/latest_realism_candidate_ab_gate.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "decision": ["recommendation", "decision"],
            "candidate_use": ["recommendation", "candidate_use"],
            "production_replacement_allowed": ["recommendation", "production_replacement_allowed"],
            "classification_auroc_delta": ["deltas", "classification_auroc_delta"],
            "regression_mae_delta": ["deltas", "regression_mae_delta"],
        },
    },
    {
        "id": "dataset_expansion_deep_search",
        "title": "Dataset expansion deep-search catalog",
        "path": "Data/evals/models/latest_dataset_expansion_deep_search.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dataset_count": ["dataset_count"],
        },
    },
    {
        "id": "priority_dataset_bridge",
        "title": "GENIE BPC + Duke MRI priority dataset bridge",
        "path": "Data/evals/models/latest_priority_dataset_bridge.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "mapped_dataset_count": ["summary", "mapped_dataset_count"],
            "ready_for_mapping_count": ["summary", "ready_for_mapping_count"],
            "full_oncotrack_temporal_validation_ready": ["summary", "full_oncotrack_temporal_validation_ready"],
        },
    },
    {
        "id": "priority_external_stress",
        "title": "Priority external schema/endpoint stress",
        "path": "Data/evals/models/latest_priority_external_stress.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "mapped_dataset_count": ["mapped_dataset_count"],
            "promotion_allowed": ["promotion_decision", "promotion_allowed"],
            "exact_oncotrack_label_match": ["endpoint_compatibility", "exact_oncotrack_label_match"],
        },
    },
    {
        "id": "external_stress_test_readiness",
        "title": "External stress-test readiness",
        "path": "Data/evals/models/latest_external_stress_test_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dataset_count": ["summary", "dataset_count"],
            "datasets_with_local_rows": ["summary", "datasets_with_local_rows"],
            "expected_abstained_rows": ["summary", "expected_abstained_rows"],
            "promotion_allowed": ["summary", "promotion_allowed"],
        },
    },
    {
        "id": "mutation_context_mapping",
        "title": "Mutation-context mapping readiness",
        "path": "Data/evals/models/latest_mutation_context_mapping.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "mapped_row_count": ["mapped_row_count"],
            "promotion_allowed": ["feature_policy", "promotion_allowed"],
        },
    },
    {
        "id": "dataset_fit_matrix",
        "title": "Dataset fit matrix",
        "path": "Data/evals/models/latest_dataset_fit_matrix.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dataset_count": ["dataset_count"],
            "production_training_allowed": ["recommendation", "production_training_allowed"],
        },
    },
    {
        "id": "cbioportal_biomarker_schema_mapping",
        "title": "TCGA/METABRIC cBioPortal schema mapping",
        "path": "Data/mle_monitoring/cbioportal_biomarker_schema_mapping.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "mapped_dataset_count": ["mapped_dataset_count"],
            "mapping_hash": ["mapping_hash"],
        },
    },
    {
        "id": "offline_ab_eval_controls",
        "title": "Offline A/B safety-control suite",
        "path": "Data/evals/ab_tests/latest_ab_test_report.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "test_count": ["test_count"],
            "overall_decision": ["overall_decision"],
            "expectations_passed": ["expectations", "passed"],
            "expectations_failed": ["expectations", "failed"],
        },
    },
    {
        "id": "finetune_behavior_governance",
        "title": "Behavior-only fine-tuning governance",
        "path": "Data/evals/models/latest_finetune_governance.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "model_trained": ["model_trained"],
            "readiness_state": ["readiness_state"],
            "training_ready": ["training_ready"],
            "promotion_ready": ["promotion_ready"],
            "dataset_accepted": ["dataset", "accepted"],
            "contamination_status": ["dataset", "contamination_status"],
            "promotion_decision": ["promotion", "decision"],
            "promotion_scope": ["promotion", "promotion_scope"],
        },
    },
    {
        "id": "mle_readiness",
        "title": "MLE readiness gate",
        "path": "Data/mle_monitoring/latest_mle_readiness.json",
        "tier": "critical",
        "metrics": {
            "hard_gate_status": ["hard_gate_status"],
            "release_recommendation": ["release_recommendation"],
            "safety_regression": ["category_statuses", "safety_regression"],
            "monitoring": ["category_statuses", "monitoring"],
        },
    },
    {
        "id": "mle_readiness_realism_candidate",
        "title": "MLE readiness - realism candidate",
        "path": "Data/mle_monitoring/latest_mle_readiness_realism_candidate.json",
        "tier": "supporting",
        "metrics": {
            "hard_gate_status": ["hard_gate_status"],
            "release_recommendation": ["release_recommendation"],
            "safety_regression": ["category_statuses", "safety_regression"],
            "realism": ["category_statuses", "realism"],
            "monitoring": ["category_statuses", "monitoring"],
        },
    },
    {
        "id": "leakage_audit",
        "title": "Training-data leakage audit",
        "path": "Data/evals/models/latest_leakage_audit.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "checks_passed": ["summary", "checks_passed"],
            "checks_failed": ["summary", "checks_failed"],
            "temporal_sub_audit_status": ["temporal_sub_audit", "status"],
        },
    },
    {
        "id": "evidence_abstention_eval",
        "title": "Evidence-aware abstention eval",
        "path": "Data/evals/models/latest_evidence_abstention_eval.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "full_data_coverage_rate": ["summary", "full_data_coverage_rate"],
            "full_data_covered_accuracy": ["summary", "full_data_covered_accuracy"],
            "demographics_only_abstention_rate": ["summary", "demographics_only_abstention_rate"],
            "scenario_count": ["summary", "scenario_count"],
        },
    },
    {
        "id": "modality_robustness_comparison",
        "title": "Modality-dropout robustness comparison",
        "path": "Data/evals/models/latest_modality_robustness_comparison.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "robust_wins": ["summary", "force_score_accuracy_wins_for_robust"],
            "robust_losses": ["summary", "force_score_accuracy_losses_for_robust"],
            "full_data_accuracy_delta": ["summary", "full_data_accuracy_delta"],
            "full_data_brier_delta": ["summary", "full_data_brier_delta"],
        },
    },
    {
        "id": "modality_robust_training",
        "title": "Modality-robust classifier training",
        "path": "Data/evals/models/latest_modality_robust_training.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "test_roc_auc": ["test_metrics", "roc_auc"],
            "test_brier": ["test_metrics", "brier"],
            "augmented_rows_added": ["augmentation_stats", "augmented_rows_added"],
            "mean_dropouts_per_augmented_row": ["augmentation_stats", "mean_dropouts_per_augmented_row"],
        },
    },
    {
        "id": "quantile_regression_training",
        "title": "Quantile response-score regression training",
        "path": "Data/evals/models/latest_quantile_regression_training.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "empirical_coverage": ["interval", "empirical_coverage"],
            "nominal_coverage": ["interval", "nominal_coverage"],
            "monotonic_rate": ["monotonic_rate_p10_p50_p90"],
            "test_rows": ["test_rows"],
        },
    },
    {
        "id": "modality_robust_regression_training",
        "title": "Modality-robust regression training",
        "path": "Data/evals/models/latest_modality_robust_regression_training.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "test_mae": ["test_metrics", "mae"],
            "test_rmse": ["test_metrics", "rmse"],
            "augmented_rows_added": ["augmentation_stats", "augmented_rows_added"],
            "mean_dropouts_per_augmented_row": ["augmentation_stats", "mean_dropouts_per_augmented_row"],
        },
    },
    {
        "id": "regression_robustness_comparison",
        "title": "Legacy vs modality-robust regression comparison",
        "path": "Data/evals/models/latest_regression_robustness_comparison.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "robust_mae_wins": ["summary", "force_score_mae_wins_for_robust"],
            "robust_mae_losses": ["summary", "force_score_mae_losses_for_robust"],
            "full_data_mae_delta": ["summary", "full_data_mae_delta"],
            "scenario_count": ["summary", "scenario_count"],
        },
    },
    {
        "id": "modality_dropout_quantile_regression",
        "title": "Modality-dropout quantile regression",
        "path": "Data/evals/models/latest_modality_dropout_quantile_regression_training.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "empirical_coverage": ["interval", "empirical_coverage"],
            "nominal_coverage": ["interval", "nominal_coverage"],
            "robust_mae_wins": ["scenario_comparison", "robust_mae_wins"],
            "robust_mae_losses": ["scenario_comparison", "robust_mae_losses"],
        },
    },
    {
        "id": "response_conformal_calibration",
        "title": "Response-score conformal calibration",
        "path": "Data/evals/models/latest_response_conformal_calibration.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "raw_coverage": ["raw_coverage"],
            "adjusted_coverage": ["adjusted_coverage"],
            "nominal_coverage": ["nominal_coverage"],
            "qhat_percent": ["qhat_percent"],
        },
    },
    {
        "id": "robustness_stress",
        "title": "Synthetic robustness stress suite",
        "path": "Data/evals/robustness/latest_robustness_report.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "pass_rate": ["summary", "pass_rate"],
            "case_count": ["summary", "case_count"],
            "abstention_or_review_rate": ["summary", "abstention_or_review_rate"],
        },
    },
    {
        "id": "synthetic_generator_card",
        "title": "Synthetic generator card",
        "path": "Data/evals/models/latest_synthetic_generator_card.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "dataset_schema_version": ["dataset_schema_version"],
            "patients_created": ["cohort", "patients_created"],
            "rows_fingerprint": ["cohort", "rows_fingerprint"],
            "card_version_matches_dataset": ["card_version_matches_dataset"],
        },
    },
    {
        "id": "failure_mode_registry",
        "title": "Consolidated failure-mode registry",
        "path": "Data/evals/safety/latest_failure_mode_registry.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "entry_count": ["entry_count"],
            "high_severity_count": ["summary", "by_severity", "high"],
            "entries_with_unresolved_gap": ["summary", "entries_with_unresolved_gap"],
        },
    },
    {
        "id": "kb_source_governance",
        "title": "KB source governance (tier + allowed_use + staleness)",
        "path": "Data/evals/rag/latest_kb_source_governance.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "source_count": ["source_count"],
            "chunk_count": ["chunk_count"],
            "governance_issue_count": ["governance_issues"],
        },
    },
    {
        "id": "toxicity_feature_audit",
        "title": "Toxicity classifier feature-importance audit + no-proxy baseline",
        "path": "Data/evals/models/latest_toxicity_feature_audit.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "dominant_features": ["dominant_features"],
            "near_label_proxy_features": ["near_label_proxy_features"],
            "no_proxy_baseline_auc": ["no_proxy_baseline", "auc"],
            "strict_no_proxy_baseline_auc": ["strict_no_proxy_baseline", "auc"],
        },
    },
    {
        "id": "rag_intent_aware_eval",
        "title": "Intent-aware RAG benchmark",
        "path": "Data/evals/rag/latest_rag_intent_aware_eval.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "pass_rate": ["summary", "pass_rate"],
            "claim_support_rate": ["summary", "claim_support_rate"],
            "citation_precision": ["summary", "citation_precision"],
            "source_tier_correctness": ["summary", "source_tier_correctness"],
            "refusal_correctness": ["summary", "refusal_correctness"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
            "latency_p50_ms": ["summary", "latency_p50_ms"],
        },
    },
    {
        "id": "live_rag_eval",
        "title": "Live-agent RAG benchmark",
        "path": "Data/evals/rag/latest_live_rag_eval.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "pass_rate": ["summary", "pass_rate"],
            "claim_support_rate": ["summary", "claim_support_rate"],
            "citation_precision": ["summary", "citation_precision"],
            "source_tier_correctness": ["summary", "source_tier_correctness"],
            "refusal_correctness": ["summary", "refusal_correctness"],
            "escalation_correctness": ["summary", "escalation_correctness"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
            "taglish_safety_parity_rate": ["summary", "taglish_safety_parity_rate"],
            "latency_p50_ms": ["summary", "latency_p50_ms"],
        },
    },
    {
        "id": "claim_level_citation_eval",
        "title": "Claim-level citation validation",
        "path": "Data/evals/rag/latest_claim_level_citation_eval.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "hard_failures": ["summary", "hard_failures"],
            "nli_required_cases": ["summary", "nli_required_cases"],
            "nli_available_cases": ["summary", "nli_available_cases"],
        },
    },
    {
        "id": "gold_claim_grounding_eval",
        "title": "Gold claim-grounding eval set",
        "path": "Data/evals/rag/latest_gold_claim_grounding_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "contradiction_trap_total": ["summary", "contradiction_trap_total"],
        },
    },
    {
        "id": "semantic_citation_verification",
        "title": "Semantic citation verification",
        "path": "Data/evals/rag/latest_semantic_citation_verification.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "hard_failures": ["summary", "hard_failures"],
            "contradicted_cases": ["summary", "contradicted_cases"],
        },
    },
    {
        "id": "nli_claim_validation_eval",
        "title": "Optional NLI/entailment claim validation",
        "path": "Data/evals/rag/latest_nli_claim_validation_eval.json",
        "tier": "optional",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "hard_failures": ["summary", "hard_failures"],
            "nli_required_cases": ["summary", "nli_required_cases"],
            "nli_available_cases": ["summary", "nli_available_cases"],
        },
    },
    {
        "id": "semantic_claim_validation",
        "title": "Semantic claim validation",
        "path": "Data/evals/rag/latest_semantic_claim_validation.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "hard_failures": ["summary", "hard_failures"],
            "contradicted_cases": ["summary", "contradicted_cases"],
            "missing_citation_cases": ["summary", "missing_citation_cases"],
        },
    },
    {
        "id": "live_rag_failure_analysis",
        "title": "Live RAG failure analysis",
        "path": "Data/evals/rag/latest_live_rag_failure_analysis.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "failure_count": ["summary", "failure_count"],
            "live_pass_rate": ["summary", "pass_rates", "live_agent"],
        },
    },
    {
        "id": "over_refusal_eval",
        "title": "Over-refusal negative controls",
        "path": "Data/evals/rag/latest_over_refusal_eval.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "safe_answer_rate": ["summary", "safe_answer_rate"],
            "inappropriate_refusal_rate": ["summary", "inappropriate_refusal_rate"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
        },
    },
    {
        "id": "multilingual_adversarial_security",
        "title": "Multilingual adversarial security",
        "path": "Data/evals/safety/latest_multilingual_adversarial_security.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "pass_rate": ["summary", "pass_rate"],
            "unsafe_leakage_rate": ["summary", "unsafe_leakage_rate"],
        },
    },
    {
        "id": "rag_tier_ablation",
        "title": "RAG source-tier retrieval ablation (T1 / T1+T2 / T1+T2+T3 / all)",
        "path": "Data/evals/rag/latest_rag_tier_ablation.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "taglish_safety_parity",
        "title": "Taglish ↔ English safety-route parity",
        "path": "Data/evals/safety/latest_taglish_safety_parity.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "pass_rate": ["pass_rate"],
            "intent_parity_rate": ["intent_parity_rate"],
            "safety_scope_parity_rate": ["safety_scope_parity_rate"],
            "case_count": ["case_count"],
        },
    },
    {
        "id": "near_boundary_safety_eval",
        "title": "Near-boundary medical safety eval",
        "path": "Data/evals/safety/latest_near_boundary_safety_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
        },
    },
    {
        "id": "model_benchmark",
        "title": "Model benchmark",
        "path": "Data/evals/models/latest_model_benchmark.json",
        "tier": "critical",
        "metrics": {
            "synthetic_champion_auroc": ["synthetic_classification", 0, "roc_auc"],
            "synthetic_champion_auprc": ["synthetic_classification", 0, "auprc"],
            "synthetic_champion_brier": ["synthetic_classification", 0, "brier"],
            "external_breastdcedl_auroc": ["external_baselines", 0, "roc_auc"],
        },
    },
    {
        "id": "current_vs_realism_candidate",
        "title": "Current vs realism-calibrated candidate",
        "path": "Data/mle_monitoring/current_vs_realism_candidate.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "decision": ["recommendation", "decision"],
            "candidate_use": ["recommendation", "candidate_use"],
            "production_replacement_allowed": ["recommendation", "production_replacement_allowed"],
            "classification_auroc_delta": ["recommendation", "classification_auroc_delta"],
            "regression_mae_delta": ["recommendation", "regression_mae_delta"],
        },
    },
    {
        "id": "synthetic_realism_candidate",
        "title": "Synthetic realism candidate",
        "path": "Data/mle_monitoring/synthetic_realism_candidate_report.json",
        "tier": "critical",
        "metrics": {
            "alignment_score": ["realism_alignment_score"],
            "training_patients": ["training_patients"],
            "threshold_coverage_status": ["threshold_coverage", "status"],
        },
    },
    {
        "id": "noise_eval",
        "title": "Noise robustness",
        "path": "Data/mle_monitoring/noise_eval_report.json",
        "tier": "supporting",
        "metrics": {
            "max_auroc_drop": ["max_auroc_drop"],
            "test_patients": ["test_patients"],
            "test_rows": ["test_rows"],
        },
    },
    {
        "id": "temporal_eval",
        "title": "Temporal generalization",
        "path": "Data/mle_monitoring/temporal_eval_report.json",
        "tier": "supporting",
        "metrics": {
            "temporal_auroc": ["temporal_split", "eval_auroc"],
            "random_baseline_auroc": ["random_split_baseline", "eval_auroc"],
            "generalization_gap": ["generalization_gap", "temporal_minus_random_auroc"],
        },
    },
    {
        "id": "calibration_eval",
        "title": "Calibration reliability",
        "path": "Data/mle_monitoring/calibration_eval_report.json",
        "tier": "supporting",
        "metrics": {
            "best_method": ["best_method"],
            "best_ece": ["best_ece"],
            "best_brier": ["best_brier"],
        },
    },
    {
        "id": "clinician_summary",
        "title": "Clinician summary quality",
        "path": "Data/evals/clinician_summary/latest_clinician_summary_eval.json",
        "tier": "supporting",
        "metrics": {
            "decision_accuracy": ["decision_accuracy"],
            "summary_completeness_rate_legitimate": ["summary_completeness_rate_legitimate"],
            "unsafe_leakage_rate": ["unsafe_leakage_rate"],
            "unsafe_detection_recall": ["unsafe_detection_recall"],
        },
    },
    {
        "id": "llm_judge",
        "title": "Optional LLM judge",
        "path": "Data/evals/llm_judge/latest_llm_judge_eval.json",
        "tier": "optional",
        "metrics": {
            "coverage_rate": ["summary", "coverage_rate"],
            "pass_rate": ["summary", "pass_rate"],
            "unsafe_medical_advice_rate": ["summary", "unsafe_medical_advice_rate"],
        },
    },
    {
        "id": "clinical_safety_review_checklist",
        "title": "Clinical safety review checklist",
        "path": "Data/evals/safety/clinical_safety_review_checklist.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "section_count": ["sections"],
        },
    },
    {
        "id": "medical_safety_contract",
        "title": "Medical safety contract",
        "path": "Data/evals/safety/latest_medical_safety_contract.json",
        "tier": "critical",
        "metrics": {
            "status": ["status"],
            "ontology_version": ["clinical_ontology", "version"],
            "evidence_standards_version": ["minimum_evidence_standards", "version"],
            "claim_boundary_version": ["medical_claim_boundary", "version"],
        },
    },
    {
        "id": "system_health",
        "title": "System health",
        "path": "Data/evals/system/latest_system_health.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "issue_count": ["issues"],
        },
    },
    {
        "id": "event_taxonomy_manifest",
        "title": "Structured event taxonomy",
        "path": "Data/evals/ops/latest_event_taxonomy_manifest.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "service_health_snapshot",
        "title": "PoC service health snapshot",
        "path": "Data/evals/ops/latest_service_health_snapshot.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "stale_artifact_count": ["metrics", "stale_artifact_count"],
            "failed_benchmark_count": ["metrics", "failed_benchmark_count"],
        },
    },
    {
        "id": "deployment_recovery_drill",
        "title": "Local synthetic backup and restore drill",
        "path": "Data/evals/ops/latest_deployment_recovery_drill.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "passed": ["passed"],
            "content_hash_match": ["restore", "content_hash_match"],
            "strict_profile_validated": ["strict_profile_validated"],
            "postgres_restore_tested": ["postgres_restore_tested"],
            "multi_instance_restore_tested": ["multi_instance_restore_tested"],
            "healthcare_production_ready": ["healthcare_production_ready"],
        },
    },
    {
        "id": "container_recovery_smoke",
        "title": "Disposable Postgres/Redis migration and recovery smoke",
        "path": "Data/evals/ops/latest_container_recovery_smoke.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "completed": ["completed"],
            "docker_available": ["docker_available"],
            "healthcare_production_ready": ["healthcare_production_ready"],
        },
    },
    {
        "id": "trace_diagnostics_coverage",
        "title": "Per-turn trace diagnostics coverage",
        "path": "Data/evals/ops/latest_trace_diagnostics_coverage.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "rows_checked": ["summary", "rows_checked"],
            "rows_with_trace_diagnostics": ["summary", "rows_with_trace_diagnostics"],
            "rows_with_retrieval_confidence": ["summary", "rows_with_retrieval_confidence"],
            "sample_trace_schema_valid": ["summary", "sample_trace_schema_valid"],
        },
    },
    {
        "id": "cost_latency_report",
        "title": "Cost and latency observability",
        "path": "Data/evals/ops/latest_cost_latency_report.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "request_count": ["summary", "request_count"],
            "latency_p50_ms": ["summary", "overall_latency_ms", "p50"],
            "latency_p95_ms": ["summary", "overall_latency_ms", "p95"],
            "estimated_total_cost_usd": ["summary", "estimated_total_cost_usd"],
            "cache_hit_rate": ["summary", "cache_hit_rate"],
        },
    },
    {
        "id": "runtime_quality_sentinel",
        "title": "Runtime quality sentinel",
        "path": "Data/evals/ops/latest_runtime_quality_sentinel.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "alert_count": ["summary", "alert_count"],
            "unsafe_answer_rate": ["summary", "unsafe_answer_rate"],
            "unsupported_claim_rate": ["summary", "unsupported_claim_rate"],
            "latency_p95_ms": ["summary", "latency_p95_ms"],
            "cache_hit_rate": ["summary", "cache_hit_rate"],
        },
    },
    {
        "id": "route_latency_budget",
        "title": "Route latency budget",
        "path": "Data/evals/ops/latest_route_latency_budget.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "route_count": ["summary", "route_count"],
            "needs_attention_count": ["summary", "needs_attention_count"],
            "highest_observed_p95_ms": ["summary", "highest_observed_p95_ms"],
        },
    },
    {
        "id": "latency_profile_phase2",
        "title": "Latency profile phase 2",
        "path": "Data/evals/ops/latest_latency_profile_phase2.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "normal_rag_previous_p95_ms": ["phase2_notes", "normal_rag_previous_p95_ms"],
            "load_smoke_previous_p95_ms": ["phase2_notes", "load_smoke_previous_p95_ms"],
        },
    },
    {
        "id": "latency_profile",
        "title": "Latency profile",
        "path": "Data/evals/ops/latest_latency_profile.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "rag_reranker_ablation",
        "title": "Cross-encoder reranker ablation",
        "path": "Data/evals/rag/latest_reranker_ablation.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "after_pass_rate_proxy": ["summary", "after_live_rag_pass_rate_proxy"],
            "after_source_tier_correctness": ["summary", "after_source_tier_correctness"],
            "after_unsupported_answer_rate": ["summary", "after_unsupported_answer_rate"],
            "reranker_latency_ms": ["summary", "reranker_latency_ms"],
        },
    },
    {
        "id": "retrieval_goldset_eval",
        "title": "Retrieval goldset eval",
        "path": "Data/evals/rag/latest_retrieval_goldset_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "recall_at_5": ["summary", "recall_at_5"],
            "recall_at_10": ["summary", "recall_at_10"],
            "mrr": ["summary", "mrr"],
            "unsupported_context_rate": ["summary", "unsupported_context_rate"],
            "improvement_proven": ["summary", "improvement_proven"],
        },
    },
    {
        "id": "rag_baseline_comparison",
        "title": "RAG baseline comparison",
        "path": "Data/evals/rag/latest_rag_baseline_comparison.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "case_count": ["summary", "case_count"],
            "bm25_recall_at_10": ["summary", "bm25_recall_at_10"],
            "full_stack_recall_at_10": ["summary", "full_stack_recall_at_10"],
            "best_recall_at_10": ["summary", "best_recall_at_10"],
            "complex_stack_improvement_over_bm25": ["summary", "complex_stack_improvement_over_bm25"],
            "unsupported_context_rate": ["summary", "unsupported_context_rate"],
            "source_tier_correctness": ["summary", "source_tier_correctness"],
            "improvement_proven_vs_bm25": ["summary", "improvement_proven_vs_bm25"],
        },
    },
    {
        "id": "citation_window_sensitivity",
        "title": "Citation window sensitivity",
        "path": "Data/evals/rag/latest_citation_window_sensitivity.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "case_count": ["case_count"],
            "recommended_cited_context_k": ["recommended_cited_context_k"],
            "promotion_recommendation": ["promotion_recommendation"],
            "live_patient_route_changed": ["live_patient_route_changed"],
            "retrieval_ranking_changed": ["retrieval_ranking_changed"],
        },
    },
    {
        "id": "retrieval_failure_analysis",
        "title": "Retrieval failure analysis",
        "path": "Data/evals/rag/latest_retrieval_failure_analysis.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "total_n": ["total_n"],
            "failed_n": ["failed_n"],
        },
    },
    {
        "id": "medical_semantic_chunking",
        "title": "Medical semantic chunking quality",
        "path": "Data/evals/rag/latest_chunking_quality_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "heading_metadata_coverage": ["heading_metadata_coverage"],
            "critical_context_split_rate": ["critical_context_split_rate"],
            "chunk_source_traceability": ["chunk_source_traceability"],
        },
    },
    {
        "id": "fhir_alignment_readiness",
        "title": "FHIR-aligned schema readiness",
        "path": "Data/evals/medical/latest_fhir_alignment_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "mapping_coverage": ["mapping_coverage"],
            "unmapped_field_count": ["unmapped_field_count"],
            "unit_normalization_success_rate": ["unit_normalization_success_rate"],
        },
    },
    {
        "id": "realtime_ood_gate",
        "title": "Real-time OOD/data-quality gate",
        "path": "Data/evals/ops/latest_realtime_ood_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "ood_detection_rate": ["summary", "ood_detection_rate"],
            "false_ood_rate": ["summary", "false_ood_rate"],
            "severe_ood_abstention_rate": ["summary", "severe_ood_abstention_rate"],
        },
    },
    {
        "id": "local_slm_readiness",
        "title": "Local SLM readiness",
        "path": "Data/evals/ops/latest_local_slm_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "low_risk_task_count": ["summary", "enabled_low_risk_task_count"],
            "blocked_solo_task_count": ["summary", "blocked_solo_task_count"],
            "production_default": ["route_policy", "production_default"],
        },
    },
    {
        "id": "release_gate_explanation",
        "title": "Release-gate explanation",
        "path": "Data/evals/governance/latest_release_gate_explanation.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
        },
    },
    {
        "id": "public_imaging_manifest",
        "title": "Public imaging readiness",
        "path": "Data/public_imaging/public_imaging_manifest.json",
        "tier": "supporting",
        "metrics": {
            "available_dataset_count": ["available_dataset_count"],
            "recommended_next_task": ["recommended_next_task"],
        },
    },
    {
        "id": "ultrasound_baseline",
        "title": "Ultrasound baseline",
        "path": "Data/public_imaging/ultrasound_baseline/metrics.json",
        "tier": "optional",
        "metrics": {
            "reason": ["reason"],
            "macro_f1": ["macro_f1"],
            "balanced_accuracy": ["balanced_accuracy"],
        },
    },
    {
        "id": "ct_lesion_workflow",
        "title": "CT lesion workflow",
        "path": "Data/public_imaging/ct_lesion_workflow/report.json",
        "tier": "optional",
        "metrics": {
            "reason": ["reason"],
            "study_count": ["study_count"],
            "workflow_stage": ["workflow_stage"],
        },
    },
    {
        "id": "structured_claim_shadow",
        "title": "Structured claim/source shadow verifier",
        "path": "Data/evals/rag/latest_structured_claim_shadow_eval.json",
        "tier": "supporting",
        "metrics": {"pass_rate": ["pass_rate"], "live_enabled": ["live_patient_agent_enabled"]},
    },
    {
        "id": "synthetic_causal_v3",
        "title": "Synthetic causal-order multi-seed stress test",
        "path": "Data/evals/models/latest_synthetic_causal_v3_stress.json",
        "tier": "supporting",
        "metrics": {"seed_count": ["seed_count"], "promotion": ["model_promotion_decision"], "realism_claim": ["realism_claim"]},
    },
    {
        "id": "xai_comprehension_proxy",
        "title": "XAI explanation-contract proxy",
        "path": "Data/evals/models/latest_xai_comprehension_contract_eval.json",
        "tier": "supporting",
        "metrics": {"pass_rate": ["pass_rate"], "human_study": ["human_participant_study_completed"]},
    },
    {
        "id": "automation_fault_injection",
        "title": "Automation queue and webhook fault injection",
        "path": "Data/evals/ops/latest_automation_fault_injection.json",
        "tier": "supporting",
        "metrics": {"pass_rate": ["pass_rate"], "scenario_count": ["scenario_count"], "external_delivery": ["external_delivery_performed"]},
    },
    {
        "id": "finetune_runtime_preflight",
        "title": "Behavior-adapter runtime preflight",
        "path": "Data/evals/models/latest_finetune_runtime_preflight.json",
        "tier": "supporting",
        "metrics": {"status": ["status"], "model_trained": ["model_trained"], "ready": ["ready_for_offline_experiment"]},
    },
    {
        "id": "oidc_browser_pkce_readiness",
        "title": "OIDC browser PKCE readiness",
        "path": "Data/evals/ops/latest_oidc_browser_pkce_readiness.json",
        "tier": "supporting",
        "metrics": {"status": ["status"], "browser_login": ["browser_login_completed"], "production_ready": ["production_auth_ready"]},
    },
    {
        "id": "data_platform_pipeline",
        "title": "Incremental non-patient data-platform pipeline",
        "path": "Data/lakehouse/manifests/latest_pipeline_run.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "source_count": ["quality", "source_count"],
            "silver_records": ["quality", "silver_record_count"],
            "gold_records": ["quality", "gold_record_count"],
            "quarantined": ["quality", "quarantined_record_count"],
            "metadata_complete": ["quality", "all_gold_metadata_complete"],
            "external_cloud_write": ["external_cloud_write_performed"],
        },
    },
    {
        "id": "managed_vector_contract",
        "title": "Managed vector-store offline contract",
        "path": "Data/evals/rag/latest_vector_store_contract_eval.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "pass_rate": ["pass_rate"],
            "gold_records": ["gold_record_count"],
            "network_request": ["managed_network_request_performed"],
            "comparison_completed": ["managed_vector_comparison_completed"],
        },
    },
    {
        "id": "azure_reference_infrastructure",
        "title": "Azure reference-infrastructure readiness",
        "path": "Data/evals/ops/latest_cloud_infrastructure_readiness.json",
        "tier": "supporting",
        "metrics": {
            "status": ["status"],
            "checks_passed": ["passed"],
            "checks_failed": ["failed"],
            "cloud_deployed": ["cloud_deployment_completed"],
            "bicep_compiled": ["bicep_compile_completed"],
        },
    },
]


def build_benchmark_registry(
    *,
    output_path: str = DEFAULT_JSON_PATH,
    report_path: str = DEFAULT_MD_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
    freshness_ttl_seconds: int = 24 * 60 * 60,
) -> dict[str, Any]:
    rows = [_build_row(spec, freshness_ttl_seconds) for spec in BENCHMARK_SPECS]
    issues = _collect_issues(rows)
    payload: dict[str, Any] = {
        **build_artifact_manifest(
            dataset_paths={
                "benchmark_sources": "benchmarks",
                "rag_gold_cases": "evals/rag_gold_cases.json",
                "safety_cases": "evals/safety_red_team_cases.json",
                "synthetic_realism_candidate": "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv",
            },
            ttl_seconds=freshness_ttl_seconds,
        ),
        "schema_version": "benchmark_registry_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _overall_status(rows),
        "critical_status": _tier_status(rows, "critical"),
        "supporting_status": _tier_status(rows, "supporting"),
        "optional_status": _tier_status(rows, "optional"),
        "benchmarks": rows,
        "issues": issues,
        "next_actions": _next_actions(rows, issues),
        "claim_boundary": (
            "Benchmarks are engineering evidence only. They test reproducibility, "
            "guardrails, retrieval behavior, calibration, and synthetic realism; "
            "they do not establish clinical safety or clinical validity."
        ),
        "report_path": report_path,
        "csv_path": csv_path,
    }
    _write_json(output_path, payload)
    _write_csv(csv_path, rows)
    _write_markdown(report_path, payload)
    return payload


def _build_row(spec: dict[str, Any], ttl_seconds: int) -> dict[str, Any]:
    payload, source_path, load_status = _load_with_fallback(spec["path"], spec.get("fallback"))
    generated_at = payload.get("generated_at") or _dig(payload, ["artifact_freshness", "generated_at"])
    artifact_ttl = _dig(payload, ["artifact_freshness", "ttl_seconds"]) or ttl_seconds
    freshness = freshness_status(generated_at, int(artifact_ttl)) if payload else "unknown"
    raw_status = _extract_status(payload, load_status)
    normalized_status = _normalize_status(raw_status, freshness, spec["tier"])
    metrics = {name: _dig(payload, path) for name, path in spec.get("metrics", {}).items()}
    return {
        "id": spec["id"],
        "title": spec["title"],
        "tier": spec["tier"],
        "status": normalized_status,
        "source_status": raw_status,
        "freshness": freshness,
        "generated_at": generated_at,
        "source_path": source_path,
        "metrics": metrics,
        "limitations": payload.get("limitations") or [],
        "claim_boundary": payload.get("claim_boundary"),
    }


def _load_with_fallback(path: str, fallback: str | None = None) -> tuple[dict[str, Any], str, str]:
    payload, status = _load_json(path)
    if status == "missing" and fallback:
        fallback_payload, fallback_status = _load_json(fallback)
        if fallback_status != "missing":
            return fallback_payload, fallback, fallback_status
    return payload, path, status


def _load_json(path: str) -> tuple[dict[str, Any], str]:
    file_path = ROOT_DIR / path
    if not file_path.exists():
        return {"status": "missing", "path": path}, "missing"
    try:
        return json.loads(file_path.read_text(encoding="utf-8")), "loaded"
    except json.JSONDecodeError as exc:
        return {"status": "error", "path": path, "error": str(exc)}, "error"


def _extract_status(payload: dict[str, Any], load_status: str) -> str:
    if load_status != "loaded":
        return load_status
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return str(summary.get("status") or payload.get("status") or "available")


def _normalize_status(raw_status: str, freshness: str, tier: str) -> str:
    raw = (raw_status or "unknown").lower()
    if raw in {"missing", "error", "failed"}:
        return raw
    if raw == "unavailable":
        return "optional_unavailable" if tier == "optional" else "unavailable"
    if freshness == "stale":
        return "stale"
    return raw


def _overall_status(rows: list[dict[str, Any]]) -> str:
    critical = [row for row in rows if row["tier"] == "critical"]
    critical_status = _tier_status(rows, "critical")
    if critical_status in {"failed", "missing", "error", "unavailable"}:
        return "blocked"
    if any(row["status"] in {"needs_attention", "unideal", "stale"} for row in critical):
        return "needs_attention"
    if any(row["status"] in {"acceptable", "available"} for row in critical):
        return "acceptable"
    return "strong"


def _tier_status(rows: list[dict[str, Any]], tier: str) -> str:
    statuses = {row["status"] for row in rows if row["tier"] == tier}
    if not statuses:
        return "not_configured"
    for status in ("error", "missing", "failed", "unavailable"):
        if status in statuses:
            return status
    if "stale" in statuses:
        return "stale"
    if statuses & {"needs_attention", "unideal"}:
        return "needs_attention"
    if statuses <= {"strong", "passed", "robust", "stable"}:
        return "strong"
    if statuses <= {"strong", "passed", "robust", "stable", "acceptable", "available", "optional_unavailable"}:
        return "acceptable"
    return "needs_attention"


def _collect_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bad_statuses = {"error", "missing", "failed", "unavailable", "needs_attention", "unideal", "stale"}
    for row in rows:
        if row["status"] in bad_statuses:
            severity = "high" if row["tier"] == "critical" and row["status"] not in {"stale"} else "medium"
            issues.append({
                "benchmark_id": row["id"],
                "severity": severity,
                "status": row["status"],
                "message": _issue_message(row),
            })
    return issues


def _issue_message(row: dict[str, Any]) -> str:
    if row["status"] == "stale":
        return "Artifact is older than the freshness TTL; rerun this benchmark before quoting it."
    if row["status"] in {"missing", "unavailable"}:
        return "Artifact is not available; dashboard should show this as missing, not hidden validation."
    if row["id"] == "mle_readiness":
        return "MLE readiness has one or more unideal categories; inspect category_statuses before promotion."
    return "Benchmark needs review before using it as supporting evidence."


def _next_actions(rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> list[str]:
    by_id = {row["id"]: row for row in rows}
    actions: list[str] = []
    candidate = by_id.get("current_vs_realism_candidate", {})
    if _dig(candidate, ["metrics", "decision"]) == "promote_candidate_after_review":
        actions.append(
            "Review the realism-calibrated synthetic candidate carefully; promotion language should stay blocked without external temporal validation."
        )
    if by_id.get("public_imaging_manifest", {}).get("metrics", {}).get("available_dataset_count") == 0:
        actions.append(
            "Download one public imaging dataset into Datasets/ first; BUSI is the lightest hardware-friendly ultrasound start."
        )
    if by_id.get("llm_judge", {}).get("status") in {"optional_unavailable", "unavailable", "missing"}:
        actions.append(
            "Keep LLM-judge optional, or configure a provider and rerun it as a heuristic grounding review."
        )
    if any(issue["benchmark_id"] == "mle_readiness" for issue in issues):
        actions.append(
            "Rerun MLE readiness after benchmark refresh; it should consume the latest realism, noise, temporal, and safety artifacts."
        )
    if not actions:
        actions.append("No hard benchmark blocker detected; focus next on public-data-backed validation and UI polish.")
    return actions


def _dig(payload: Any, path: list[Any]) -> Any:
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = ROOT_DIR / path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output = ROOT_DIR / path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark_id",
                "title",
                "tier",
                "status",
                "source_status",
                "freshness",
                "source_path",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or {"status": row.get("status")}
            for metric, value in metrics.items():
                writer.writerow({
                    "benchmark_id": row["id"],
                    "title": row["title"],
                    "tier": row["tier"],
                    "status": row["status"],
                    "source_status": row["source_status"],
                    "freshness": row["freshness"],
                    "source_path": row["source_path"],
                    "metric": metric,
                    "value": value,
                })


def _write_markdown(path: str, payload: dict[str, Any]) -> None:
    output = ROOT_DIR / path
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MedicalAgent Benchmark Registry",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
        f"Overall status: **{payload['status']}**",
        f"Critical status: **{payload['critical_status']}**",
        "",
        payload["claim_boundary"],
        "",
        "## Benchmark Matrix",
        "",
        "| Benchmark | Tier | Status | Freshness | Key metrics | Source |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload["benchmarks"]:
        metric_text = "; ".join(
            f"{key}={_format_metric(value)}"
            for key, value in (row.get("metrics") or {}).items()
            if value is not None
        ) or "no extracted metrics"
        lines.append(
            f"| {row['title']} | {row['tier']} | {row['status']} | "
            f"{row['freshness']} | {metric_text} | `{row['source_path']}` |"
        )
    lines.extend(["", "## Issues"])
    if payload["issues"]:
        for issue in payload["issues"]:
            lines.append(
                f"- {issue['severity']}: {issue['benchmark_id']} "
                f"({issue['status']}) - {issue['message']}"
            )
    else:
        lines.append("- No current hard issues detected.")
    lines.extend(["", "## Next Actions"])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
