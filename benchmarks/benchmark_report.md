# MedicalAgent Benchmark Registry

Generated at: 2026-05-15T12:28:46.513050+00:00

Overall status: **acceptable**
Critical status: **acceptable**

Benchmarks are engineering evidence only. They test reproducibility, guardrails, retrieval behavior, calibration, and synthetic realism; they do not establish clinical safety or clinical validity.

## Benchmark Matrix

| Benchmark | Tier | Status | Freshness | Key metrics | Source |
|---|---:|---:|---:|---|---|
| Safety red-team | critical | passed | fresh | pass_rate=1.000; failed_cases=[]; total_cases=9 | `Data/evals/safety/latest_safety_benchmark.json` |
| Adversarial prompt/jailbreak | critical | passed | fresh | attack_block_rate=1.000; failed_cases=[] | `Data/evals/safety/latest_adversarial_eval.json` |
| Multilingual refusal routing | critical | strong | fresh | pass_rate=1.000; passed=6; case_count=6 | `Data/evals/safety/latest_multilingual_refusal_eval.json` |
| RAG regression | critical | strong | fresh | pass_rate=1.000; citation_coverage_rate=1.000; expected_source_hit_rate=1.000; unsafe_answer_rate=0.000; average_grounding_score=1.000 | `Data/evals/rag/latest_rag_benchmark.json` |
| Hand-labeled RAG gold set | critical | strong | fresh | pass_rate=1.000; expected_source_hit_rate=1.000; case_count=41; unsafe_answer_rate=0.000 | `Data/evals/rag/latest_rag_gold_eval.json` |
| Patient-support tool action benchmark | critical | passed | fresh | pass_rate=1.000; case_count=6; average_latency_ms=360.410; max_latency_ms=526.300 | `Data/evals/tool_actions/latest_tool_action_benchmark.json` |
| MLE readiness gate | critical | acceptable | fresh | hard_gate_status=passed; release_recommendation=acceptable_for_poc_demo_with_limitations; safety_regression=strong; monitoring=acceptable | `Data/mle_monitoring/latest_mle_readiness.json` |
| MLE readiness - realism candidate | supporting | acceptable | fresh | hard_gate_status=passed; release_recommendation=acceptable_for_poc_demo_with_limitations; safety_regression=strong; realism=passed; monitoring=acceptable | `Data/mle_monitoring/latest_mle_readiness_realism_candidate.json` |
| Model benchmark | critical | available | fresh | synthetic_champion_auroc=0.995; synthetic_champion_auprc=0.996; synthetic_champion_brier=0.047; external_breastdcedl_auroc=0.637 | `Data/evals/models/latest_model_benchmark.json` |
| Current vs realism-calibrated candidate | critical | available | fresh | decision=promote_candidate_after_review; auc_delta=-0.001; realism_delta=0.433; candidate_alignment=0.861; current_alignment=0.428 | `Data/mle_monitoring/current_vs_realism_candidate.json` |
| Synthetic realism candidate | critical | acceptable | fresh | alignment_score={'score': 0.844, 'status': 'passed', 'interpretation': '0.90+ is strong, 0.75+ is passed, 0.60+ is acceptable. This is an engineering realism score, not clinical validity.'}; training_patients=240; threshold_coverage_status=acceptable | `Data/mle_monitoring/synthetic_realism_candidate_report.json` |
| Noise robustness | supporting | mild_degradation | fresh | max_auroc_drop=0.064; test_patients=60; test_rows=360 | `Data/mle_monitoring/noise_eval_report.json` |
| Temporal generalization | supporting | stable | fresh | temporal_auroc=0.978; random_baseline_auroc=0.975 | `Data/mle_monitoring/temporal_eval_report.json` |
| Calibration reliability | supporting | passed | fresh | best_method=isotonic_regression; best_ece=0.022; best_brier=0.046 | `Data/mle_monitoring/calibration_eval_report.json` |
| Clinician summary quality | supporting | passed | fresh | decision_accuracy=1.000; summary_completeness_rate_legitimate=1.000; unsafe_leakage_rate=0.000; unsafe_detection_recall=1.000 | `Data/evals/clinician_summary/latest_clinician_summary_eval.json` |
| Optional LLM judge | optional | optional_unavailable | fresh | coverage_rate=0.000 | `Data/evals/llm_judge/latest_llm_judge_eval.json` |
| Public imaging readiness | supporting | ready_for_experiments | fresh | available_dataset_count=1; recommended_next_task=Run ultrasound baseline: python scripts/run_ultrasound_baseline.py --dataset-root Datasets/BUSI | `Data/public_imaging/public_imaging_manifest.json` |
| Ultrasound baseline | optional | completed | fresh | no extracted metrics | `Data/public_imaging/ultrasound_baseline/metrics.json` |
| CT lesion workflow | optional | optional_unavailable | stale | reason=DeepLesion or PET/CT lesion dataset not found locally. | `Data/public_imaging/ct_lesion_workflow/report.json` |

## Issues
- No current hard issues detected.

## Next Actions
- Promote the realism-calibrated synthetic candidate after reviewing threshold coverage and model-card language.
- Keep LLM-judge optional, or configure a provider and rerun it as a heuristic grounding review.
