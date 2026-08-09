# Prototype-Independent V2 And Real-Pipeline Scale Evaluation

## Purpose

These two suites answer different engineering questions:

- `prototype_independent_prompt_bank_v2.jsonl` is a frozen, one-pass bank of
  3,000 compositional routing cases. Its generator does not import classifier
  prototypes or prior prompt banks. It remains internally authored and is not
  external evidence.
- `latest_real_pipeline_scale_eval.json` runs 300 synthetic prompts through
  the actual `run_patient_agent_pipeline` path and records retrieval, cache,
  citation, post-generation validation, latency, and token telemetry without
  retaining response text.

## Freeze Discipline

The v2 bank is created before its first execution, SHA-256 locked, and marked
`evaluated_once: true` after the first run. Failures from that run are not used
to change the classifier during the same pass. A future development bank must
be created for tuning; v2 remains a historical baseline.

## Cold And Warm Latency

The scale run clears process-local retrieval objects, explicitly prewarms the
KB/index/encoder, and reports that startup cost separately from the 300 warm
query latencies. This prevents a first-request model load from being hidden
inside a normal-route p95 while also avoiding a false production SLO claim.

## Token Accounting

Provider-reported token counts are stored only when the configured provider
returns usage metadata. Offline runs report chars/4 estimates separately and
must not label them as billed or provider-measured tokens.

## Latest Frozen V2 Result

The first completed one-pass run evaluated all 3,000 SHA-locked cases and
passed 1,953 (`0.6510`). Unsafe-intent misses were `0.3663`, while safe-control
over-refusal was `0.2222`. The weakest categories were privacy/PII (`0.2375`),
genetic-risk interpretation (`0.3625`), VUS interpretation (`0.5125`), and
diagnosis confirmation (`0.5292`). Taglish and English performance were nearly
identical (`0.6504` and `0.6511`), so language alone does not explain the weak
generalization.

These failures remain frozen baseline evidence. They were not used to tune the
classifier in this pass. The result is `needs_attention`, not evidence that
agent safety is solved.

## Latest Real-Pipeline Result

All 300 calls completed through `run_patient_agent_pipeline`; 216 met the
route/contract oracle (`0.7200`). The run observed the post-generation
validator on all 300 calls, 45 citation-bearing turns, 20 research-evidence
abstentions, and 5 cache hits. It estimated 98,126 pipeline tokens. Provider
usage coverage was zero because the local/offline provider did not return
usage metadata; estimates are not billed-token measurements.

The copy-on-read unsafe-intent result cache preserved the `0.7200` contract
score while changing warm latency as follows:

| Measurement | Before cache | After cache |
| --- | ---: | ---: |
| warm p50 | 531.15 ms | 290.40 ms |
| warm p95 | 786.60 ms | 449.31 ms |
| warm max | 1,766.13 ms | 665.10 ms |
| explicit prewarm | 24,509.58 ms | 85,223.23 ms |

The warm-path reduction is useful local engineering evidence. The highly
variable prewarm time remains a startup bottleneck and prevents any production
latency claim. The classifier cache changes repeated computation, not routing
semantics; mutation-isolation tests protect against shared mutable results.

## Claim Boundary

Both suites use synthetic/internal prompts. They are engineering evidence, not
clinical validation, real-world safety evidence, patient-benefit evidence,
clinician approval, or production healthcare readiness.
