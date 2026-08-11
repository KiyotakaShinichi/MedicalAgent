# Adversarial Generalization V-Next

Generated from repository artifacts at `2026-08-11T06:52:48.637897+00:00`.

> NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness.

## Method

The v7 result is read-only attribution over an already-inspected internal holdout. A separate post-change mutation matrix evaluates generalized wrappers and safe controls and is explicitly marked tuning-used.

## Before and after evidence

| Evidence | Pass rate | Unsafe leakage/block signal | Interpretation |
|---|---:|---:|---|
| Frozen v7 stored baseline | 0.676056 | leakage 0.354545 | Internal and author-contaminated; not rerun |
| V-next mutation matrix | 1.0 | unsafe block 1.0 | Tuning-used regression only |
| Safe negative controls | 1.0 | over-refusal 0.0 | Internal controls |

## Architecture change

An independent operation-authorization guard now sits before tool planning. Privacy, prompt-injection, and cross-patient operations can be denied even if another classifier misses, while the existing semantic/security route remains an independent blocking layer.

## Remaining failures and limits

- Frozen v7 attributed failure count: `46`.
- External generalization status: `BLOCKED_EXTERNAL`.
- A perfect tuning-used mutation score must not replace an untouched independent result.
- No claim of solved safety is authorized.
