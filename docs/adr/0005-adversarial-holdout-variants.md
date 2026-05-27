# 0005 — Adversarial bank held-out variant set with anti-contamination test

**Status**: accepted

## Context

After the 2026-05-20 hardening pass, the deterministic safety
vocabulary in `agent_safety.py` and `security_guardrails.py` was
extended using phrasings drawn directly from the 200-case adversarial
bank's failing cases. That raised the in-sample attack_block_rate on
the four hardened categories to 1.00.

A 1.00 score on a fixed bank that has been used for rule-tuning is
**not** a generalisation claim — it's a memorisation receipt.

## Decision

Two parallel artifacts, never confused:

1. `Data/evals/safety/adversarial_safety_regression_bank.jsonl`
   — 200 cases, may be used for rule-tuning, surfaces gaps.
2. `Data/evals/safety/adversarial_safety_holdout_variants.jsonl`
   — 32 cases, **never used for tuning**, fresh wording, 4 hardened
   categories only.

The runner writes two separate JSON outputs. The drift tracker tracks
both. The release gate has a `latest_adversarial_safety_holdout.json`
entry that is **informational only** — its pass rate is NOT enforced
because the artifact's purpose is to surface the gap between in-sample
and held-out, not to be tuned to pass.

The test `test_adversarial_safety_holdout::test_holdout_queries_do_not_overlap_original_bank`
makes the anti-contamination invariant CI-gated: no held-out query may
be a substring of any original-bank query within the same category.

## Consequences

- ✅ The first held-out run produced 0.06 overall vs in-sample 1.00.
  That gap is the honest baseline and is recorded in
  `docs/adversarial_safety_regression.md`.
- ✅ Bank tuning is allowed and tracked; held-out generalisation is
  measured and reported separately.
- ⚠ A held-out rate of 0.06 is **bad news**, and we keep it visible
  rather than tuning it away. Future hardening of the safety vocab
  must be evaluated against held-out movement, not in-sample.

## Reversal cost

Low. Merging the two banks reverts to the old framing, but reverses
the trust the held-out artifact buys us. Don't.
