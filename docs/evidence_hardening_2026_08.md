# Evidence Hardening Pass — August 2026

NLCare remains a synthetic-only, non-diagnostic engineering prototype. This
pass strengthens falsifiability and release discipline; it does not establish
clinical validation, real-world safety, patient benefit, clinician approval,
or production healthcare readiness.

## Changes

- The compact release decision now treats a missing, failed, or stale full
  ship manifest as a hard blocker.
- The former 186-test backend assurance batch is split into two bounded steps
  so failures are attributable and each step stays within the default timeout.
- An evaluation-only claim-conditioned citation selector assigns governed
  sources to individual claims, leaves unsupported claims visible, excludes
  stale/clinician-only sources, and strips citations on refusal routes.
- The selector remains disabled on live patient routes. Its initial positive
  comparison is internally authored, tuning-used, and permits only an offline
  shadow candidate.
- Synthetic ML perturbation testing now includes severity-dependent non-random
  missingness. Targets remain unchanged so measured degradation is attributable
  to evidence loss rather than target mutation.
- The compact MLE/XAI decision surface now follows the perturbation stress and
  fail-closed XAI reliability gate instead of relying on narrower green checks.

## Current Decision

The compact engineering release decision remains blocked until a fresh full
ship manifest passes. Synthetic perturbation failures and unstable XAI claims
remain warnings. Detailed release artifacts are supporting evidence and cannot
override these blocking states through volume or averaging.

## Next Falsifiable Evidence

1. Complete a fresh full ship run after the bounded-suite split.
2. Run the citation selector on frozen generated answers, including
   contradiction, refusal, unsupported-claim, and Taglish cases.
3. Improve MNAR and cross-generator robustness without changing synthetic
   targets or weakening abstention.
4. Keep exact feature-rank claims hidden until retraining stability passes.
5. Complete independent no-read and qualified human review when available.
