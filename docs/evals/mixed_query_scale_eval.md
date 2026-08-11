# Mixed-query scale evaluation

This internal stress suite exercises three traffic families with a deterministic seed:

- 1,000 research-KB query variants derived from the local research-paper grounding cases;
- 1,000 explicit off-topic or ambiguous/noise inputs; and
- 1,000 high-risk or unsafe-intent variants, including urgent wording.

Run it with:

```powershell
python scripts/run_mixed_query_scale_eval.py
```

The evaluation intentionally has three layers. All 3,000 prompts pass through local safety, semantic unsafe-intent, scope, and intent routing. All 1,000 research prompts pass through the source-governed retrieval stack. A stratified 300-prompt sample passes through the real stateful support-chat service with provider generation disabled. The last layer checks response completion, expected routing, citations for educational answers, safe next-step wording for dangerous requests, authority leakage, local latency, and token telemetry.

The split matters: a routing classifier success is not counted as a successful RAG answer, and a retrieval hit is not counted as medically correct. Hosted generation is not scaled to thousands because doing so would add cost and provider variance without making this internally authored bank independent.

Artifacts:

- `Data/evals/agentic_tool_use/mixed_query_scale_bank.jsonl`
- `Data/evals/agentic_tool_use/latest_mixed_query_scale_eval.json`
- `Data/evals/agentic_tool_use/latest_mixed_query_scale_failures.json`

## Latest internal run

The 2026-08-10 run completed 3,000 generated prompts: 1,000 research-KB
queries, 1,000 garbage/off-topic inputs, and 1,000 dangerous variants. It
finished with `status: needs_attention`.

| Measure | Result |
| --- | ---: |
| Local routing contract | 3,000 / 3,000 |
| Research Recall@5 | 0.998 |
| Research Recall@10 | 1.000 |
| Research MRR | 0.989 |
| Research source-tier correctness | 1.000 |
| Stateful support-chat sample | 265 / 300 (0.8833) |
| Dangerous stateful sample | 100 / 100 |
| Dangerous safe-next-step rate | 1.000 |
| Dangerous unsafe-authority leakage | 0.000 |
| Garbage/off-topic stateful sample | 100 / 100 |
| Research stateful sample | 65 / 100 |

The retrieval headline needs qualification. Although the expected paper was
present by rank 10 for every research query, 319 cases missed the expected
paper section at rank 10. The real support-chat sample also had 35 research
answers without citations (16 topic paraphrases, 15 exact-title requests, and
4 negative-result requests). Those cases safely abstained instead of emitting
unsupported medical education, but they remain answerability failures.

Cold initialization took 52,173.4 ms. After prewarming, stateful support-chat
latency was 711.71 ms p50, 2,878.64 ms p95, and 5,379.48 ms p99. Retrieval-only
latency was 302.75 ms p50, 592.95 ms p95, and 759.29 ms p99. The 300 stateful
cases consumed an estimated 95,502 tokens. Provider-reported token coverage was
zero because this run disabled hosted generation, so the token number is an
engineering estimate rather than a billing measurement.

These results support regression coverage and failure discovery. They do not
show independent generalization, clinical correctness, real-world safety, or
production latency readiness.

## Interpretation boundary

This bank is generated from project-owned prompts and KB-derived cases, is visible during development, and is marked `was_used_for_tuning: true`. It is regression and failure-discovery evidence only. It is not an independent holdout, clinician review, clinical validation, proof of medical correctness, proof of real-world safety, or production healthcare readiness.
