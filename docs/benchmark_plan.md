# Benchmark Plan for MedicalAgent

This project should report three kinds of benchmarks: agent/RAG quality,
clinical-monitoring ML quality, and imaging baselines. Keep every benchmark
clearly labeled as engineering evidence, not clinical validation.

## 1. Agent and RAG Benchmarks

Use these for the patient-support agent, retrieval, citations, and refusals.

| Benchmark | What it tests | Target |
| --- | --- | --- |
| Local RAG gold set | expected source hit, citation coverage, refusal correctness | >= 0.95 pass rate |
| Safety red-team suite | diagnosis refusal, treatment refusal, prompt injection, privacy | 1.0 attack block rate |
| Multilingual refusal eval | Tagalog/Taglish diagnosis, treatment, urgent routing | >= 0.95 pass rate |
| RAG ablation | BM25 vs dense FAISS vs RRF vs rerank | full pipeline should beat BM25-only |
| Optional LLM judge | groundedness, answer relevance, refusal quality | report as heuristic, not truth |
| Cache benchmark | exact/semantic hit rate, p50/p95 latency, safety cache gating | cache hit faster than miss |

Recommended external tooling:

- RAGAS for answer faithfulness/context precision.
- DeepEval or TruLens if you want a second LLM-judge view.
- BEIR-style retrieval metrics for retrieval-only experiments if you build a larger labeled KB.

## 2. ML and MLE Benchmarks

Use these for the synthetic temporal monitoring and response-risk models.

| Benchmark | What it tests | Target |
| --- | --- | --- |
| Current vs realism-v2 candidate | metric retention vs better data realism | realism improves without >0.03 AUROC loss |
| Locked synthetic holdout | frozen patient-level split | no tuning against holdout |
| Temporal split | train early cycles, evaluate later cycles | stable AUROC/PR-AUC and no leakage |
| High-noise robustness | missingness, unit jitter, timing irregularity | graceful metric degradation |
| Cost-sensitive thresholding | false negative vs false positive asymmetry | choose threshold with clinical-review rationale |
| Calibration | Brier score, ECE, reliability diagram | lower ECE/Brier after calibration |
| Subgroup calibration | subtype/regimen/stage slices | surface weak slices, do not hide them |
| Per-prediction audit | TP/FP/TN/FN, threshold, top features, dataset hash | every test prediction explainable |

## 3. Public Data Benchmarks

Use public datasets to reduce synthetic-only weakness.

| Dataset | Use in project | Notes |
| --- | --- | --- |
| BreastDCEDL / I-SPY / Duke-derived features | response modeling and sim-to-public calibration | best fit for treatment-response framing |
| Duke Breast Cancer MRI (TCIA) | MRI imaging baseline and DCE-MRI workflow | large public breast MRI collection |
| ACRIN 6698 / I-SPY2 (TCIA) | treatment response MRI benchmark | closer to neoadjuvant response modeling |
| BUSI / BUS-BRA ultrasound | benign/malignant/normal classification and segmentation | good for hardware-friendly CV baselines |
| MIMIC-IV | EHR/lab pipeline realism and missingness patterns | credentialed access; not breast-cancer-specific |

## 4. Imaging Benchmarks

Use small, honest baselines rather than overclaiming.

| Task | Model family | Metrics |
| --- | --- | --- |
| Breast ultrasound classification | logistic baseline on handcrafted features, small CNN, transfer learning | AUROC, PR-AUC, sensitivity, specificity |
| Ultrasound segmentation | classical thresholding, U-Net-lite | Dice, IoU, boundary error |
| Breast MRI tabular response | scikit-learn and gradient boosting on extracted features | AUROC, PR-AUC, Brier, ECE |
| CT/metastatic workflow | metadata/report ingestion first; segmentation later | completeness, routing correctness, source trace |

## 5. What To Show in the Portfolio

Lead with this:

1. A benchmark matrix with current, candidate, and target values.
2. The synthetic-to-public gap and what changed after realism-v2.
3. One per-prediction error table.
4. One RAG trace with route, retrieval IDs, citations, cache status, and latency.
5. One failure case gallery showing what still breaks.

Do not lead with synthetic AUROC alone.
