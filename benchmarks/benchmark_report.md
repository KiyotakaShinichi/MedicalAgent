# MedicalAgent Benchmark Report

Generated at: 2026-05-14T14:26:37.952861+00:00

Claim boundary: benchmarks are engineering evidence only. They do not establish clinical validation.

Benchmark philosophy:
MedicalAgent is evaluated not by how often it answers, but by whether it answers only when safe, cites only when grounded, escalates when needed, and exposes uncertainty when the data is weak.

## Safety benchmark
- unsafe_pass_rate: 0.000
- urgent_escalation_recall: 1.000
- privacy_leak_rate: 0.000
- prompt_injection_resistance: 1.000

## Adversarial benchmark
- attack_block_rate: 1.000

## RAG benchmark
- pass_rate: 0.875
- citation_coverage: 1.000
- expected_source_hit: 1.000
- refusal_correct: 1.000
- unsafe_answer_rate: 0.000

## Model benchmark
- synthetic_champion_auroc: 0.995
- synthetic_champion_auprc: 0.996
- synthetic_champion_brier: 0.047
- synthetic_champion_ece_after: 0.064

## Synthetic realism
- realism_alignment_score: not_generated
- realism_checks_status: needs_attention

## Clinician summary benchmark
- summary_completeness_rate: 0.594
- unsafe_advice_rate: 0.250
