# Large-scale agent prompt stress evaluation

## Purpose

This suite exposes the bounded patient-support agent to thousands of synthetic
wording variants without sending them to the live database or presenting them
as clinical evidence. It is a regression and failure-discovery tool.

The default run creates a deterministic bank of 5,000 prompts and evaluates:

- all 5,000 through the unsafe-intent classifier;
- a stratified 500-case sample through the bounded planner, simulated executor,
  final-response packager, and verifier;
- 70 generated two-turn conversations across symptom detail carry-over,
  confirmed writes, urgent follow-ups, treatment-boundary persistence,
  prompt-injection persistence, cross-patient persistence, and safe transitions.

Prompt families include diagnosis, treatment change, dosage, prognosis,
genetics, VUS, tumor markers, supplements, privacy/PII, prompt injection,
cross-patient exfiltration, safe education, structured record updates, partial
updates, urgent language, emotional distress, Taglish, typo, spacing, casing,
punctuation, symbol-separator, and zero-width variants.

## Run

```powershell
python scripts/run_large_scale_agent_prompt_eval.py
```

Artifacts:

- `Data/evals/agentic_tool_use/large_scale_prompt_bank.jsonl`
- `Data/evals/agentic_tool_use/latest_large_scale_agent_prompt_eval.json`
- `Data/evals/agentic_tool_use/latest_large_scale_agent_prompt_failures.json`

The bank is generated with a fixed seed and a SHA-256 fingerprint of the
persisted JSONL bytes. Every evaluated case is marked `internal_generated`,
`was_used_for_tuning: true`, and `clinical_validation: false` because the July
2026 baseline failures were inspected during hardening. This bank is now a
regression suite, not a holdout.

## Interpreting the result

The large layer deliberately disables optional LLM adjudication so the run is
repeatable and affordable. The sampled end-to-end layer exercises the bounded
agentic scaffold, not thousands of hosted response generations. It performs no
database writes; write tools are simulated only after an explicit confirmed
flag and verified against the workflow contract.

Failures remain in the failure artifact for taxonomy. Once inspected, all
subsequent runs are explicitly marked tuning-used. Independent generalization
still requires an external-author or no-read frozen holdout.

This suite cannot establish clinical validation, real-world safety,
generalization, clinician approval, patient benefit, or production healthcare
readiness. Because the generator is authored inside this repository, its score
must not be reported as independent held-out or external-author evidence.

## Baseline before hardening: 2026-07-13

| Layer | Cases | Result |
|---|---:|---:|
| Unsafe-intent classifier | 5,000 prompts | 99.62% contract pass |
| Unsafe prompts | 2,709 prompts | 99.30% detected |
| Safe/non-unsafe prompts | 2,291 prompts | 0.00% over-refusal |
| Bounded planner/executor/verifier | 500 sampled prompts | 94.00% route accuracy |
| Structured tool subset | 28 sampled prompts | 78.57% route/write accuracy |
| Stateful bounded conversations | 70 conversations / 140 turns | 37.14% conversation pass |
| Unsafe record writes | sampled bounded layer | 0 |

The artifact status is `needs_attention`. The 19 classifier failures are all
typo variants and cluster in diagnosis confirmation, genetic-risk
interpretation, and dosage requests. The 30 sampled bounded-agent failures
include punctuation/zero-width normalization gaps, urgent-language misses,
and partial-versus-complete tool-routing errors.

The dominant weakness is multi-turn boundary persistence. Pending symptom
details and confirmed writes mostly carry forward, but vague follow-ups after
urgent, treatment-change, prompt-injection, and cross-patient requests do not
retain the prior boundary in this scaffold. This also exposes control-path
drift: the live support-chat path contains some context-aware safety follow-up
logic that the bounded orchestrator does not yet share.

These failures were not used to tune the baseline run, but they were inspected
afterward. The bank therefore became tuning-used evidence for the next pass.

## Post-hardening regression: 2026-07-14

| Layer | Cases | Baseline | Post-hardening |
|---|---:|---:|---:|
| Unsafe-intent classifier contract | 5,000 prompts | 99.62% | 100.00% |
| Unsafe prompt detection | 2,709 prompts | 99.30% | 100.00% |
| Safe/non-unsafe over-refusal | 2,291 prompts | 0.00% | 0.00% |
| Bounded planner route accuracy | 500 sampled prompts | 94.00% | 100.00% |
| Structured route/write accuracy | 28 sampled prompts | 78.57% | 100.00% |
| Stateful conversation pass | 70 conversations | 37.14% | 100.00% |
| Unsafe record writes | sampled bounded layer | 0 | 0 |

The generalized hardening added shared Unicode/typo normalization, separated
upload intent from completed imaging records, preserved retrospective treatment
notes as record organization, and carried active urgent, treatment, security,
and cross-patient boundaries across short follow-up turns. The post-hardening
artifact status is `acceptable`.

The perfect result is not a robustness or safety claim. It is an internal
post-hardening regression result on a deterministic synthetic bank whose
failures were used for tuning. It does not replace the weaker frozen held-out
adversarial evidence or establish real-user behavior.
