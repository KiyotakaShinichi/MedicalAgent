# DEP-001C Evaluation-Integrity Incident Analysis

## Scope and decision

This analysis concerns evidence integrity only. It does not reinterpret or tune
the DEP-001B behavioral candidate. The consumed 750-case internal blind bank
and official 400-case external holdout remain burned and are not rerun.

DEP-001B remains `BLOCKED_EVIDENCE_INTEGRITY`. DEP-001 remains `BLOCKED`.

## Artifact that changed

| Field | Value |
|---|---|
| Repository path | `Data/evals/safety/dep001b/latest_overlap_audit.json` |
| Frozen SHA-256 | `805adefd9bf60fae8c3b2a9312bd8fa3370693c5ace9805a63086264498feb4f` |
| Post-run SHA-256 | `2c1daeb04998ba89dbaf3adc0db6511d3d36fc75bcb6899055ac83606f54fccf` |
| File creation time (UTC) | `2026-08-13T16:21:39.8694166Z` |
| Frozen manifest time (UTC) | `2026-08-13T18:40:11.381897Z` |
| Blind run interval (UTC) | `2026-08-13T18:40:47Z` to `2026-08-13T18:46:56Z` |
| Mismatched file last-write time (UTC) | `2026-08-13T18:46:16.5535771Z` |
| Post-run verification | 25/26 artifacts matched |

## Root cause

An encoder-heavy pytest command exceeded its parent command timeout. The child
Python/pytest process continued after the parent shell returned. The overlap
test invoked `run_overlap_audit()` with a one-case temporary external fixture,
but the service still held a repository-global `OUTPUT_PATH`. The orphaned
child therefore completed a legitimate atomic-looking write to the mutable
`latest_overlap_audit.json` path during the official blind interval.

The process could write after freeze because the DEP-001B freeze was a hash
declaration over mutable development paths. It did not create a separate
read-only snapshot, acquire an evaluation lock, inventory active writer
processes, or enforce a filesystem write barrier. Verification happened only
before case 1.

The responsible test was `tests/test_dep001b_overlap_audit.py`. It now redirects
all outputs and comparison paths to pytest temporary storage and asserts that
the repository artifact hash is unchanged.

## Determinism and transient modification analysis

The destination-selection failure was deterministic: any invocation using the
unpatched global output constant would target the same repository file. The
fixture-derived aggregate values were deterministic for the fixture and model;
the embedded generation timestamp and orphan completion time were not.

The frozen manifest and post-run audit identify exactly one persistent mismatch.
There is no evidence that another frozen artifact was transiently modified.
The former design had no continuous watcher, so absence of evidence is not a
proof that no transient write occurred. DEP-001C closes that observability gap
with deterministic checkpoint verification.

## Could scoring observe different bytes?

All 26 hashes matched before scoring. The overlap artifact was not read by the
classifier, policy action selector, post-generation validator, or metric
summarizer after startup. The changed bytes therefore did not affect a scored
prediction. Nevertheless, the declared freeze invariant failed, so the
behavioral metrics remain historical, non-decisive internal evidence.

## Mechanical prevention

DEP-001C replaces the mutable-path design with:

1. content-addressed candidate and blind-bank snapshots;
2. copied runtime artifacts and frozen source modules;
3. read-only snapshot files;
4. explicit candidate and blind-bank identifiers rather than `latest` aliases;
5. process inventory and conflict refusal before freeze/evaluation;
6. atomic candidate-scoped evaluation locks with stale-lock handling;
7. pre-run, deterministic checkpoint, and post-run hash verification;
8. canonical manifest verification before evidence acceptance; and
9. an irreversible `INVALIDATED` evidence-transaction state.

Nothing in this remediation is clinical validation or real-patient safety
evidence.
