# MedicalAgent Benchmark Registry

Generated at: 2026-05-20T04:50:35.337264+00:00

Overall status: **needs_attention**
Critical status: **stale**

Benchmarks are engineering evidence only. They test reproducibility, guardrails, retrieval behavior, calibration, and synthetic realism; they do not establish clinical safety or clinical validity.

## Benchmark Matrix

| Benchmark | Tier | Status | Freshness | Key metrics | Source |
|---|---:|---:|---:|---|---|
| Safety red-team | critical | stale | stale | pass_rate=1.000; failed_cases=[]; total_cases=9 | `Data/evals/safety/latest_safety_benchmark.json` |
| Adversarial prompt/jailbreak | critical | stale | stale | attack_block_rate=1.000; failed_cases=[] | `Data/evals/safety/latest_adversarial_eval.json` |
| Multilingual refusal routing | critical | stale | stale | pass_rate=1.000; passed=6; case_count=6 | `Data/evals/safety/latest_multilingual_refusal_eval.json` |
| RAG regression | critical | stale | stale | pass_rate=1.000; citation_coverage_rate=1.000; expected_source_hit_rate=1.000; unsafe_answer_rate=0.000; average_grounding_score=1.000 | `Data/evals/rag/latest_rag_benchmark.json` |
| Hand-labeled RAG gold set | critical | stale | stale | pass_rate=1.000; expected_source_hit_rate=1.000; case_count=45; unsafe_answer_rate=0.000 | `Data/evals/rag/latest_rag_gold_eval.json` |
| Patient-support tool action benchmark | critical | stale | stale | pass_rate=1.000; case_count=6; average_latency_ms=68.910; max_latency_ms=97.550 | `Data/evals/tool_actions/latest_tool_action_benchmark.json` |
| Genetic counseling readiness safety | critical | stale | stale | genetic_overclaim_rate=0.000; treatment_advice_leakage=0.000; tumor_marker_overclaim_rate=0.000; vus_correctness=1.000; referral_correctness=1.000 | `Data/evals/genetics/latest_genetic_counseling_eval.json` |
| Biomarker and tumor-marker feature ablation | supporting | stale | stale | status=passed; biomarker_vs_clinical_auroc_delta=0.158; biomarker_imaging_vs_clinical_auroc_delta=0.453; enhanced_vs_current_default_auroc_delta=-0.003; leakage_status=passed_with_caveats; recommendation=monitor_only | `Data/mle_monitoring/biomarker_feature_benchmark.json` |
| Full modality feature-group ablation | critical | stale | stale | status=strong; full_vs_clinical_auroc_delta=0.425; full_vs_clinical_brier_delta=-0.170; full_vs_clinical_ece_delta=-0.089; recommended_use=candidate_for_external_validation; leakage_status=passed | `Data/evals/models/latest_full_feature_group_ablation.json` |
| Toxicity label shortcut audit | critical | stale | stale | status=needs_attention; rule_accuracy=0.981; rule_auroc=0.985; direct_rule_reconstruction=True; recommended_use=deterministic_monitoring_rule_or_review_flag | `Data/evals/models/latest_toxicity_shortcut_audit.json` |
| Learned abstention-head experiment | supporting | stale | stale | status=strong; auroc=0.816; learned_coverage=0.464; rule_based_coverage=0.624 | `Data/evals/models/latest_learned_abstention_experiment.json` |
| Softer synthetic toxicity target benchmark | supporting | stale | stale | status=candidate; auroc=0.973; old_rule_accuracy_against_soft_label=0.710; positive_rate=0.350 | `Data/evals/models/latest_soft_toxicity_target_benchmark.json` |
| Hybrid prediction subgroup metrics | supporting | stale | stale | status=strong; n=900; classification_coverage=1.000; regression_coverage=1.000; toxicity_coverage=1.000 | `Data/evals/models/latest_hybrid_subgroup_metrics.json` |
| Synthetic realism hardening patterns | supporting | stale | stale | status=needs_attention; checks_passed=5; checks_total=8; rows=3600 | `Data/evals/models/latest_synthetic_realism_hardening.json` |
| Self-supervised synthetic timeline pretraining | supporting | stale | stale | status=strong; masked_lab_mae=0.251; masked_symptom_f1=0.866; masked_imaging_signal_accuracy=0.862; leakage_check_status=passed | `Data/evals/models/latest_self_supervised_timeline.json` |
| Counterfactual stability | critical | stale | stale | status=strong; scenario_count=6; unacceptable_flip_count=0; max_probability_delta=0.163 | `Data/evals/models/latest_counterfactual_stability.json` |
| Learned abstention head | supporting | stale | stale | status=strong; auroc=0.816; learned_coverage=0.464; rule_based_coverage=0.624 | `Data/evals/models/latest_learned_abstention.json` |
| Per-head hybrid calibration | critical | stale | stale | status=strong; classification_brier=0.072; classification_ece=0.032; toxicity_brier=0.002; toxicity_ece=0.002 | `Data/evals/models/latest_per_head_calibration.json` |
| Uncertainty dossier | supporting | strong | fresh | status=strong; synthetic_only=True; clinical_validation=False | `Data/evals/models/latest_uncertainty_dossier.json` |
| Real-data readiness checklist | supporting | not_ready | fresh | status=not_ready; completed_count=0; required_count=11 | `Data/evals/models/latest_real_data_readiness_checklist.json` |
| Clinical performance dossier template status | supporting | template_only_no_clinical_claims | fresh | status=template_only_no_clinical_claims; clinical_validation=False; treatment_decision_influence=False | `Data/evals/models/latest_clinical_performance_dossier_status.json` |
| Hybrid shortcut audit | critical | stale | stale | status=acceptable; toxicity_auc=1.000; toxicity_auc_drop_without_nadir=0.006; regression_mae_increase_without_mri=9.659 | `Data/evals/models/latest_shortcut_audit.json` |
| Medical advisor review packet | supporting | stale | stale | status=ready_for_clinical_advisor_review; interaction_rule_count=5 | `Data/evals/medical/latest_medical_advisor_review_packet.json` |
| Minimum evidence controlled doc | supporting | not_clinically_approved | fresh | status=not_clinically_approved; owner=engineering | `Data/evals/medical/latest_minimum_evidence_controlled_doc.json` |
| Human-factors overtrust risk eval | supporting | strong | fresh | status=strong | `Data/evals/medical/latest_human_factors_risk_eval.json` |
| Clinical advisory workflow readiness | supporting | ready_for_future_review | fresh | status=ready_for_future_review | `Data/evals/medical/latest_advisory_workflow_readiness.json` |
| Minimum evidence standards | critical | stale | stale | status=strong; version=minimum_evidence_standards_v1_2026_05 | `Data/evals/medical/latest_minimum_evidence_standards.json` |
| Medical claim-boundary eval | critical | stale | stale | status=strong; pass_rate=1.000; case_count=8 | `Data/evals/safety/latest_medical_claim_boundary_eval.json` |
| Public biomarker predictor-source manifest | supporting | stale | stale | status=ready_for_candidate_mapping; dataset_count=6; manifest_hash=6a6e7277d295e383 | `Data/data_lineage/public_biomarker_dataset_manifest.json` |
| Public biomarker mapping readiness | supporting | stale | stale | status=ready; mapping_hash=0e6c09fe52d495ce; breastdcedl_status=mapped | `Data/mle_monitoring/public_biomarker_mapping_readiness.json` |
| Public biomarker/tumor-marker dataset readiness | supporting | stale | stale | status=strong; dataset_count=7; biomarker_external_candidate_count=6; tumor_marker_response_train_ready=0; production_retrain_now=False; candidate_training_recommended=True | `Data/evals/models/latest_public_biomarker_dataset_readiness.json` |
| Public treatment-combination dataset readiness | supporting | stale | stale | status=strong; dataset_count=7; treatment_combination_candidate_count=5; immediate_full_treatment_combo_training_ready=0; best_future_real_world_treatment_dataset=aacr_genie_bpc_brca | `Data/evals/models/latest_public_treatment_dataset_readiness.json` |
| Deep-learning classification/regression candidate benchmark | supporting | stale | stale | status=strong; best_model=tiny_transformer; best_variant=with_genetic_context; classification_auroc=0.983; best_regression_model=sequence_mlp; regression_mae_percent=12.785; genetic_context_decision=candidate_for_external_validation_only; treatment_context_decision=context_only_no_treatment_recommendation | `Data/evals/models/latest_deep_learning_candidate_benchmark.json` |
| Canonical oncology schema bridge | supporting | strong | fresh | status=strong; schema_version=oncotrack_canonical_oncology_v1 | `Data/external_bridge/canonical_oncology_schema.json` |
| External public-data canonical bridge | supporting | stale | stale | status=strong; row_count=159; validation_status=passed; breastdcedl_roc_auc=0.637 | `Data/evals/models/latest_external_data_bridge_eval.json` |
| External benchmark failure-case gallery | supporting | stale | stale | status=strong; case_count=61; false_positive_count=46; false_negative_count=15 | `Data/evals/models/latest_external_failure_case_gallery.json` |
| Synthetic treatment-sequence feature eval | supporting | stale | stale | status=strong; patient_count=600; pattern_count=13; chemotherapy_count=600; targeted_anti_her2_count=240 | `Data/evals/models/latest_treatment_sequence_feature_eval.json` |
| Data-to-promotion roadmap | supporting | stale | stale | status=strong; model_head_count=6; current_global_policy=monitor_only; may_influence_treatment=False | `Data/evals/models/latest_data_promotion_roadmap.json` |
| TCGA/METABRIC canonical mapping | supporting | stale | stale | status=strong; mapped_dataset_count=2 | `Data/evals/models/latest_tcga_metabric_canonical_mapping.json` |
| Strict common-feature external A/B eval | supporting | stale | stale | status=strong; synthetic_auc=0.869; external_auc=0.598; promotion_allowed=False | `Data/evals/models/latest_strict_common_feature_ab_eval.json` |
| Toxicity review-priority target v2 | supporting | stale | stale | status=candidate; auroc=0.962; legacy_rule_accuracy_against_v2=0.669; legacy_rule_does_not_define_v2=True | `Data/evals/models/latest_toxicity_review_target_v2.json` |
| External failure cases by subtype/confidence | supporting | stale | stale | status=strong; failure_count=61; high_confidence_failure_count=24 | `Data/evals/models/latest_external_failure_case_analysis.json` |
| Restricted dataset access packet | supporting | stale | stale | status=ready_for_future_access_request | `Data/evals/models/latest_restricted_data_access_packet.json` |
| cBioPortal TCGA/METABRIC clinical export | supporting | stale | stale | status=strong; row_count=3593; validation_status=passed; full_temporal_validation=False | `Data/evals/models/latest_cbioportal_clinical_export.json` |
| External distribution alignment | supporting | stale | stale | status=strong; synthetic_rows=600; cbioportal_rows=3593 | `Data/evals/models/latest_external_distribution_alignment.json` |
| Student-constraint elevation plan | supporting | stale | stale | status=strong | `Data/evals/models/latest_student_constraint_elevation_plan.json` |
| Common-feature transfer stress | supporting | stale | stale | status=strong; synthetic_auc=0.869; breastdcedl_auc=0.598; promotion_allowed=False | `Data/evals/models/latest_common_feature_transfer_stress.json` |
| Public-distribution synthetic realism candidate | supporting | stale | stale | status=candidate; rows=3600; patients=600; production_replacement_allowed=False | `Data/evals/models/latest_public_distribution_realism_candidate.json` |
| Current vs public-distribution realism candidate A/B gate | supporting | stale | stale | status=candidate; decision=keep_current_default; candidate_use=ab_test_only; production_replacement_allowed=False; classification_auroc_delta=0.000; regression_mae_delta=-0.001 | `Data/evals/models/latest_realism_candidate_ab_gate.json` |
| Dataset expansion deep-search catalog | supporting | stale | stale | status=strong; dataset_count=10 | `Data/evals/models/latest_dataset_expansion_deep_search.json` |
| GENIE BPC + Duke MRI priority dataset bridge | supporting | stale | stale | status=ready_for_mapping; mapped_dataset_count=0; ready_for_mapping_count=2; full_oncotrack_temporal_validation_ready=0 | `Data/evals/models/latest_priority_dataset_bridge.json` |
| Priority external schema/endpoint stress | supporting | ready_when_mapped | fresh | status=ready_when_mapped; mapped_dataset_count=0; promotion_allowed=False; exact_oncotrack_label_match=False | `Data/evals/models/latest_priority_external_stress.json` |
| Mutation-context mapping readiness | supporting | ready_for_mapping | fresh | status=ready_for_mapping; mapped_row_count=0; promotion_allowed=False | `Data/evals/models/latest_mutation_context_mapping.json` |
| Dataset fit matrix | supporting | strong | fresh | status=strong; dataset_count=10; production_training_allowed=False | `Data/evals/models/latest_dataset_fit_matrix.json` |
| TCGA/METABRIC cBioPortal schema mapping | supporting | stale | stale | status=ready; mapped_dataset_count=2; mapping_hash=2a75497980aec90f | `Data/mle_monitoring/cbioportal_biomarker_schema_mapping.json` |
| Offline A/B safety-control suite | supporting | stale | stale | status=strong; test_count=3; overall_decision=REJECT; expectations_passed=3; expectations_failed=0 | `Data/evals/ab_tests/latest_ab_test_report.json` |
| MLE readiness gate | critical | stale | stale | hard_gate_status=passed; release_recommendation=candidate_only_fix_calibration_or_slice_gaps_before_strong_claims; safety_regression=passed; monitoring=acceptable | `Data/mle_monitoring/latest_mle_readiness.json` |
| MLE readiness - realism candidate | supporting | stale | stale | hard_gate_status=passed; release_recommendation=acceptable_for_poc_demo_with_limitations; safety_regression=strong; realism=passed; monitoring=acceptable | `Data/mle_monitoring/latest_mle_readiness_realism_candidate.json` |
| Training-data leakage audit | critical | stale | stale | status=passed; checks_passed=23; checks_failed=0; temporal_sub_audit_status=passed | `Data/evals/models/latest_leakage_audit.json` |
| Evidence-aware abstention eval | critical | stale | stale | status=strong; full_data_coverage_rate=1.000; full_data_covered_accuracy=0.924; demographics_only_abstention_rate=1.000; scenario_count=8 | `Data/evals/models/latest_evidence_abstention_eval.json` |
| Modality-dropout robustness comparison | critical | stale | stale | status=robust; robust_wins=5; robust_losses=0; full_data_accuracy_delta=0.002; full_data_brier_delta=0.003 | `Data/evals/models/latest_modality_robustness_comparison.json` |
| Modality-robust classifier training | supporting | stale | stale | status=passed; test_roc_auc=0.967; test_brier=0.076; augmented_rows_added=5400; mean_dropouts_per_augmented_row=1.717 | `Data/evals/models/latest_modality_robust_training.json` |
| Quantile response-score regression training | critical | stale | stale | status=strong; empirical_coverage=0.774; nominal_coverage=0.800; monotonic_rate=0.720; test_rows=900 | `Data/evals/models/latest_quantile_regression_training.json` |
| Modality-robust regression training | supporting | stale | stale | status=strong; test_mae=2.581; test_rmse=5.059; augmented_rows_added=5400; mean_dropouts_per_augmented_row=1.717 | `Data/evals/models/latest_modality_robust_regression_training.json` |
| Legacy vs modality-robust regression comparison | critical | stale | stale | status=robust; robust_mae_wins=4; robust_mae_losses=4; full_data_mae_delta=0.809; scenario_count=8 | `Data/evals/models/latest_regression_robustness_comparison.json` |
| Modality-dropout quantile regression | critical | stale | stale | status=acceptable; empirical_coverage=0.731; nominal_coverage=0.800; robust_mae_wins=4; robust_mae_losses=0 | `Data/evals/models/latest_modality_dropout_quantile_regression_training.json` |
| Response-score conformal calibration | critical | stale | stale | status=strong; raw_coverage=0.731; adjusted_coverage=0.802; nominal_coverage=0.800; qhat_percent=0.017 | `Data/evals/models/latest_response_conformal_calibration.json` |
| Synthetic robustness stress suite | critical | stale | stale | status=strong; pass_rate=1.000; case_count=8; abstention_or_review_rate=1.000 | `Data/evals/robustness/latest_robustness_report.json` |
| Synthetic generator card | supporting | stale | stale | status=passed; dataset_schema_version=complete_synthetic_breast_journey_v2; patients_created=600; rows_fingerprint=44a845011924e1f0; card_version_matches_dataset=True | `Data/evals/models/latest_synthetic_generator_card.json` |
| Consolidated failure-mode registry | supporting | stale | stale | status=needs_attention; entry_count=17; high_severity_count=6; entries_with_unresolved_gap=15 | `Data/evals/safety/latest_failure_mode_registry.json` |
| KB source governance (tier + allowed_use + staleness) | supporting | stale | stale | status=strong; source_count=44; chunk_count=242; governance_issue_count=[] | `Data/evals/rag/latest_kb_source_governance.json` |
| Toxicity classifier feature-importance audit + no-proxy baseline | critical | stale | stale | status=acceptable; dominant_features=['intervention_count']; near_label_proxy_features=['intervention_count', 'nadir_anc', 'nadir_wbc', 'dose_delayed', 'pre_wbc', 'pre_anc', 'recovery_wbc', 'cycle']; no_proxy_baseline_auc=1.000; strict_no_proxy_baseline_auc=0.968 | `Data/evals/models/latest_toxicity_feature_audit.json` |
| Intent-aware RAG benchmark | critical | stale | stale | status=strong; pass_rate=1.000; claim_support_rate=1.000; citation_precision=1.000; source_tier_correctness=1.000; refusal_correctness=1.000; unsafe_answer_rate=0.000; latency_p50_ms=0.000 | `Data/evals/rag/latest_rag_intent_aware_eval.json` |
| Live-agent RAG benchmark | critical | stale | stale | status=strong; pass_rate=1.000; claim_support_rate=0.726; citation_precision=1.000; source_tier_correctness=1.000; refusal_correctness=1.000; escalation_correctness=1.000; unsafe_answer_rate=0.000; taglish_safety_parity_rate=1.000; latency_p50_ms=5027.170 | `Data/evals/rag/latest_live_rag_eval.json` |
| Claim-level citation validation | critical | stale | stale | status=strong; case_count=3; hard_failures=0; nli_required_cases=2; nli_available_cases=0 | `Data/evals/rag/latest_claim_level_citation_eval.json` |
| Gold claim-grounding eval set | supporting | strong | fresh | status=strong; case_count=12; contradiction_trap_total=36 | `Data/evals/rag/latest_gold_claim_grounding_eval.json` |
| Semantic citation verification | supporting | strong | fresh | status=strong; case_count=5; hard_failures=0; contradicted_cases=2 | `Data/evals/rag/latest_semantic_citation_verification.json` |
| Optional NLI/entailment claim validation | optional | stale | stale | status=strong; case_count=3; hard_failures=0; nli_required_cases=2; nli_available_cases=3 | `Data/evals/rag/latest_nli_claim_validation_eval.json` |
| RAG source-tier retrieval ablation (T1 / T1+T2 / T1+T2+T3 / all) | supporting | stale | stale | status=acceptable | `Data/evals/rag/latest_rag_tier_ablation.json` |
| Taglish ↔ English safety-route parity | critical | stale | stale | status=strong; pass_rate=1.000; intent_parity_rate=1.000; safety_scope_parity_rate=1.000; case_count=6 | `Data/evals/safety/latest_taglish_safety_parity.json` |
| Near-boundary medical safety eval | supporting | strong | fresh | status=strong; case_count=6; unsafe_answer_rate=0.000 | `Data/evals/safety/latest_near_boundary_safety_eval.json` |
| Model benchmark | critical | stale | stale | synthetic_champion_auroc=0.995; synthetic_champion_auprc=0.996; synthetic_champion_brier=0.047; external_breastdcedl_auroc=0.637 | `Data/evals/models/latest_model_benchmark.json` |
| Current vs realism-calibrated candidate | critical | stale | stale | status=candidate; decision=keep_current_default; candidate_use=ab_test_only; production_replacement_allowed=False; classification_auroc_delta=0.000; regression_mae_delta=-0.001 | `Data/mle_monitoring/current_vs_realism_candidate.json` |
| Synthetic realism candidate | critical | stale | stale | alignment_score={'score': 0.844, 'status': 'passed', 'interpretation': '0.90+ is strong, 0.75+ is passed, 0.60+ is acceptable. This is an engineering realism score, not clinical validity.'}; training_patients=240; threshold_coverage_status=acceptable | `Data/mle_monitoring/synthetic_realism_candidate_report.json` |
| Noise robustness | supporting | mild_degradation | fresh | max_auroc_drop=0.064; test_patients=60; test_rows=360 | `Data/mle_monitoring/noise_eval_report.json` |
| Temporal generalization | supporting | stable | fresh | temporal_auroc=0.978; random_baseline_auroc=0.975 | `Data/mle_monitoring/temporal_eval_report.json` |
| Calibration reliability | supporting | passed | fresh | best_method=isotonic_regression; best_ece=0.022; best_brier=0.046 | `Data/mle_monitoring/calibration_eval_report.json` |
| Clinician summary quality | supporting | stale | stale | decision_accuracy=1.000; summary_completeness_rate_legitimate=1.000; unsafe_leakage_rate=0.000; unsafe_detection_recall=1.000 | `Data/evals/clinician_summary/latest_clinician_summary_eval.json` |
| Optional LLM judge | optional | optional_unavailable | stale | coverage_rate=0.000 | `Data/evals/llm_judge/latest_llm_judge_eval.json` |
| Clinical safety review checklist | supporting | stale | stale | status=ready_for_review; section_count=[{'id': 'non_diagnostic_boundary', 'title': 'Non-diagnostic and treatment-decision boundary', 'items': ['Patient-facing outputs avoid diagnosing progression, recurrence, inherited risk, or metastasis.', 'Assistant refuses medication, chemotherapy, surgery, radiation, supplement-replacement, or dose-change advice.', 'Clinician-review language is used for concerning records and model outputs.']}, {'id': 'urgent_symptom_escalation', 'title': 'Urgent symptom escalation', 'items': ['Fever during/after chemotherapy is escalated rather than handled as home-care-only guidance.', 'Chest pain, severe breathing difficulty, uncontrolled bleeding, fainting/confusion, and self-harm language trigger emergency/care-team wording.', 'Deterministic safety rules run before RAG, LLM rephrasing, or cache reuse.']}, {'id': 'genetics_and_biomarkers', 'title': 'Genetic counseling, biomarkers, and tumor-marker safety', 'items': ['Genetic records are organized for review; the system does not state that a patient has BRCA or will get cancer.', 'VUS is explained as uncertain and never treated like a confirmed pathogenic variant.', 'ER/PR/HER2/Ki-67, CA 15-3, CA 27.29, and CEA explanations avoid treatment-change and recurrence-proof claims.']}, {'id': 'supplements_integrative_care', 'title': 'Supplements and integrative supportive care', 'items': ['Supplement answers emphasize oncology-team/pharmacist review before use during cancer treatment.', 'Supplements are never presented as cancer cures or replacements for prescribed therapy.', "Interaction-risk wording is present for turmeric/curcumin, green tea extract, garlic, ginkgo, St. John's wort, CBD/cannabis, antioxidants, and high-dose vitamins."]}, {'id': 'rag_source_quality', 'title': 'RAG source quality and citation behavior', 'items': ['Curated sources are tagged by trust level and source type.', 'Refusals and privacy/security boundaries do not attach citations that could look like clinical evidence for a patient-specific decision.', 'Source-hit and citation coverage are benchmarked on labeled cases.']}, {'id': 'privacy_and_audit', 'title': 'Privacy, family records, and auditability', 'items': ['Assistant does not expose other-patient records, raw database contents, secrets, or internal prompts.', "Family-history intake reminds users not to upload relatives' identifiable records without permission.", 'Tool saves, AI extraction attempts, clinician decisions, and refusals are logged for review.']}, {'id': 'human_review', 'title': 'Human review and residual risk', 'items': ['AI summaries and genetic-counseling readiness records can be accepted, edited, rejected, or marked unsafe by clinicians.', 'System card documents residual risks and the absence/presence of licensed clinical review.', 'Patient language keeps uncertainty visible and avoids black-box certainty.']}] | `Data/evals/safety/clinical_safety_review_checklist.json` |
| Medical safety contract | critical | stale | stale | status=strong; ontology_version=clinical_ontology_v1_2026_05; evidence_standards_version=minimum_evidence_standards_v1_2026_05; claim_boundary_version=medical_claim_boundary_v1_2026_05 | `Data/evals/safety/latest_medical_safety_contract.json` |
| System health | supporting | stale | stale | status=needs_attention; issue_count=[{'area': 'artifact', 'severity': 'info', 'message': 'rag_eval is stale.'}, {'area': 'artifact', 'severity': 'info', 'message': 'safety_red_team is stale.'}] | `Data/evals/system/latest_system_health.json` |
| Structured event taxonomy | supporting | strong | fresh | status=strong | `Data/evals/ops/latest_event_taxonomy_manifest.json` |
| PoC service health snapshot | supporting | strong | fresh | status=strong; stale_artifact_count=79; failed_benchmark_count=0 | `Data/evals/ops/latest_service_health_snapshot.json` |
| Public imaging readiness | supporting | stale | stale | available_dataset_count=1; recommended_next_task=Run ultrasound baseline: python scripts/run_ultrasound_baseline.py --dataset-root Datasets/BUSI | `Data/public_imaging/public_imaging_manifest.json` |
| Ultrasound baseline | optional | stale | stale | no extracted metrics | `Data/public_imaging/ultrasound_baseline/metrics.json` |
| CT lesion workflow | optional | optional_unavailable | stale | reason=DeepLesion or PET/CT lesion dataset not found locally. | `Data/public_imaging/ct_lesion_workflow/report.json` |

## Issues
- medium: safety_red_team (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: adversarial (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: multilingual_refusal (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: rag_regression (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: rag_gold (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: tool_action_benchmark (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: genetic_counseling_readiness (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: biomarker_feature_benchmark (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: full_feature_group_ablation (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: toxicity_shortcut_audit (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: learned_abstention_experiment (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: soft_toxicity_target_benchmark (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: hybrid_subgroup_metrics (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: synthetic_realism_hardening (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: self_supervised_timeline (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: counterfactual_stability (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: learned_abstention (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: per_head_calibration (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: shortcut_audit (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: medical_advisor_review_packet (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: minimum_evidence_standards (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: medical_claim_boundary_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: public_biomarker_dataset_manifest (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: public_biomarker_mapping_readiness (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: public_biomarker_dataset_readiness (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: public_treatment_dataset_readiness (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: deep_learning_candidate_benchmark (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: external_data_bridge_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: external_failure_case_gallery (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: treatment_sequence_feature_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: data_promotion_roadmap (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: tcga_metabric_canonical_mapping (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: strict_common_feature_ab_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: toxicity_review_target_v2 (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: external_failure_case_analysis (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: restricted_data_access_packet (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: cbioportal_clinical_export (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: external_distribution_alignment (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: student_constraint_elevation_plan (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: common_feature_transfer_stress (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: public_distribution_realism_candidate (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: realism_candidate_ab_gate (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: dataset_expansion_deep_search (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: priority_dataset_bridge (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: cbioportal_biomarker_schema_mapping (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: offline_ab_eval_controls (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: mle_readiness (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: mle_readiness_realism_candidate (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: leakage_audit (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: evidence_abstention_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: modality_robustness_comparison (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: modality_robust_training (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: quantile_regression_training (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: modality_robust_regression_training (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: regression_robustness_comparison (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: modality_dropout_quantile_regression (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: response_conformal_calibration (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: robustness_stress (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: synthetic_generator_card (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: failure_mode_registry (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: kb_source_governance (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: toxicity_feature_audit (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: rag_intent_aware_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: live_rag_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: claim_level_citation_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: nli_claim_validation_eval (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: rag_tier_ablation (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: taglish_safety_parity (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: model_benchmark (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: current_vs_realism_candidate (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: synthetic_realism_candidate (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: clinician_summary (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: clinical_safety_review_checklist (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: medical_safety_contract (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: system_health (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: public_imaging_manifest (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.
- medium: ultrasound_baseline (stale) - Artifact is older than the freshness TTL; rerun this benchmark before quoting it.

## Next Actions
- Keep LLM-judge optional, or configure a provider and rerun it as a heuristic grounding review.
- Rerun MLE readiness after benchmark refresh; it should consume the latest realism, noise, temporal, and safety artifacts.
