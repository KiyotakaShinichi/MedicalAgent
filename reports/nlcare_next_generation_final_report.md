# NLCare Next-Generation Engineering Evidence Report

Generated from repository artifacts at `2026-08-11T06:52:48.687921+00:00`.

> NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness.

## 1. Executive summary

NLCare is a strong portfolio-grade healthcare AI engineering prototype with unusually explicit safety, retrieval, MLE, security, and governance evidence. Its strongest new contribution is a reproducible evidence program that preserves negative results. It remains far from a clinical product.

## 2. Starting state

The repository already included source-governed dense/sparse RAG, bounded tools, claim checks, synthetic temporal ML, n8n automation scaffolding, role dashboards, traces, and a large release gate. Weak evidence included internally authored holdouts, section retrieval misses, sparse staged serving, and no completed external review.

## 3. Changes implemented

Added frozen-dataset integrity, independent route authorization, retrieved-context integrity, corrected section parsing, section-aware ablation, adversarial attribution/mutations, corpus-poisoning tests, tenant-isolation attacks, failure attribution, load profiling, review feedback validation, a unified nlcare_eval runner, and a non-compensatory Accuracy-Latency-Unit-Cost gate.

## 4. Architecture changes

The request path now has an additional operation-level authorization layer. The retrieval path sanitizes instruction-like context before assembly. Ingestion persists section headings and canonical section provenance. SaaS workers validate organization/project relationships before execution.

## 5. Evaluation methodology

All internal, frozen, tuning-used, and blocked-external assets are identified in a registry with hashes and usage restrictions. Reports include code/index/dataset provenance where the unified runner is used.

## 6. Frozen/generalization datasets

Integrity status: `verified_with_external_gaps` with `0` failures. Independent external work remains `BLOCKED_EXTERNAL`.

## 7. Adversarial results

Stored v7 pass rate is `0.676056`. The new tuning-used mutation matrix pass rate is `1.0`. This does not prove external generalization.

## 8. Retrieval ablation results

The previous full source-governed stack did not prove raw Recall@10 superiority over BM25. Section-aware promotion is `False`.

## 9. Section-aware retrieval results

Recovered `31` of `319` known internal misses, with `0` regressions.

## 10. Dense-vs-BM25-vs-hybrid decision

Dense FAISS is implemented, indexed, and locally benchmarkable. Sparse remains the restricted-staging default because complexity has not earned a held-out quality/latency promotion.

## 11. Failure attribution

Largest unique case-stage bucket: `section_mismatch`.

## 12. Corpus-poisoning results

Status: `passed`. Generation-context poison rate: `0.0`.

## 13. Tenant-isolation results

Status: `passed`. No result is a penetration-test claim.

## 14. Runtime performance and AI Trinity

Planner load status: `acceptable_internal_stress`. AI Trinity decision: `HOLD_ACCURACY_GROUNDING` with accuracy `needs_attention`, latency `pass`, and unit cost `blocked_evidence`. Normal-API provider probe: `blocked_configuration` with coverage `0.0`. Missing provider telemetry is not treated as zero cost. These are internal measurements, not a production SLO.

## 15. Chaos/reliability results

Automation fault status: `strong`; RAG degradation status: `strong_offline_drill`.

## 16. Human-review status

Feedback ingestion status: `BLOCKED_EXTERNAL`; accepted rows: `0`. No clinician sign-off is implied.

## 17. Deployment status

Synthetic SaaS foundation: `ready_for_restricted_synthetic_saas_alpha`. Managed deployment completed: `False`.

## 18. Test/release evidence

The final verification section must be read with the latest ship and release-gate artifacts. A green engineering gate does not establish clinical readiness.

## 19. Negative results

Full-stack raw retrieval superiority remains unproven, the prior citation pruner regressed precision, and the frozen claim-conditioned selector changed citation precision from `0.0909` to `0.0429` with promotion `offline_only_not_promoted`. V7 generalization was weak, provider token coverage is incomplete, and external review is absent.

## 20. Remaining weaknesses

Independent evaluation, clinician/genetics review, real-data validity, managed outage/restore drills, provider cost telemetry, and human-factors testing remain the main credibility gaps.

## 21. External blockers

External authors/reviewers, managed cloud credentials, real traffic, restricted datasets, institutional governance, and any future IRB pathway are outside this repository pass.

## 22. Promotion decisions

Promote the dataset-integrity gate, operation authorization, context-integrity sanitizer, tenant relation checks, and controlled security regressions. Promote section-aware retrieval only if the artifact's predeclared conditions pass. Do not promote clinical authority, dense serving by appearance, the negative citation pruner, or the negative claim-conditioned selector.

## 23. Next highest-value task

Complete one independently authored no-read RAG/adversarial evaluation and one qualified reviewer packet. Inside the repo, next run the same fixed suites in managed synthetic staging with traceable provider usage and backup/restore drills.
