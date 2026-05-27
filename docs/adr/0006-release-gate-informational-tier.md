# 0006 — Release-gate informational tier + warn-on-regression floors

**Status**: accepted

## Context

The release gate had grown to ~61 artifacts during the post-stabilisation
pass. The external critique called this "paperwork dilution" — no
human reads 60+ artifacts top-to-bottom, so a regression in the middle
of the list could ship unnoticed.

But shrinking the gate by removing artifacts was the wrong move: the
artifacts encode real engineering invariants. We needed a tier
structure.

## Decision

Three implicit tiers, encoded via two YAML fields per artifact:

- **Blocker** — `required: true`. Build fails if missing, stale, or
  any metric threshold fails.
- **Informational** — `required: false`, `status: "informational"`
  written by the producer. Surfaces as `[OK]` in the gate output,
  shows up in the eval drift report, can warn but cannot block.
- **Warn-on-regression** — `required: false` BUT with
  `metric_thresholds` set to a floor below the current value. Drops
  to `[warn]` if the metric crosses the floor.

Examples:

- The 200-case adversarial bank's per-category floors (`>= 0.9` on
  urgent_symptom and safe_negative_control) are blocker-tier.
- The four hardened-category floors (`>= 0.9` on privacy_pii,
  prompt_injection, genetic_risk, vus) are warn-on-regression.
- The held-out adversarial result is informational only — never gated.

## Consequences

- ✅ The gate still has ~70 artifacts, but a reviewer can read the
  blocker tier alone (~20 artifacts) and trust the build.
- ✅ The held-out adversarial 0.06 result cannot be tuned away because
  it is never a release-blocker.
- ⚠ The tier is implicit — the artifact's `status` field encodes part
  of it. A future change to status semantics would require an ADR
  update.

## Reversal cost

Low. Set every artifact to `required: true`. But you lose the
"add an artifact safely" path; new measurement layers would either be
blockers (too aggressive) or invisible (no gate at all).
