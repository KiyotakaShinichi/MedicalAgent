# Benchmark Contamination Disclosure

NLCare's current benchmark sets are mostly internal engineering regression
sets. They are useful for preventing regressions, but they are not independent
clinical evidence.

## Current Status

- Authorship: primarily engineering-authored.
- Tuning exposure: many cases may have been used to tune prompts, safety rules,
  retrieval policies, validators, or thresholds.
- Clinical review: not clinician-reviewed.
- Patient data: no real patient cohort.
- Intended use: regression protection and proof packaging for a student-built
  synthetic-only prototype.

## How To Read Scores

High pass rates mean the current implementation satisfies the authored
contracts. They do not prove real-world safety, clinical correctness, patient
benefit, or production readiness.

## Required Fields

Each eval set should expose:

- `authored_by`
- `authored_date`
- `was_used_for_tuning`
- `internal_vs_external_authored`
- `case_source`
- `contamination_disclosure`
- `baseline_version`
- `release_id`

## Future Improvement

When access becomes possible, add externally authored and clinician-reviewed
holdout sets. Keep those locked and never tune directly on them.
