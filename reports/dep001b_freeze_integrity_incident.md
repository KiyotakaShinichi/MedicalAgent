# DEP-001B Freeze-Integrity Incident

## Decision

DEP-001B internal targets passed, but DEP-001B is not formally complete and is
not ready for a new external holdout under the declared protocol. One of 26
frozen artifacts changed during the one-shot blind run. The blind bank must not
be rerun and the mismatch must not be silently repaired.

## What Happened

Two encoder-heavy pytest commands exceeded their parent command timeouts. A
child process from `tests/test_dep001b_overlap_audit.py` continued running. That
test called `run_overlap_audit()` with a one-case temporary fixture, but the
service wrote to the repository-global `latest_overlap_audit.json`. The orphan
completed at `2026-08-13T18:46:16Z`, during the blind interval
`18:40:47Z`-`18:46:56Z`, replacing the frozen 400-case aggregate diagnostic.

The test now redirects output to pytest's temporary directory and asserts that
the repository artifact hash is unchanged.

## Impact

The blind runner verified all 26 hashes before scoring and would not otherwise
have started. The changed overlap artifact is not read by the scorer after that
start check and is not model code, policy code, configuration, thresholds,
calibration, model weights, or a blind input. The blind bank, manifest, result,
raw result, and receipt hashes remain mutually consistent. The one-shot metrics
are therefore preserved as useful but non-decisive internal evidence.

Strictly, however, post-run candidate verification is `25/26`, so the declared
freeze invariant failed. This is an evidence-governance failure even though it
did not alter a scored prediction. Calling the candidate cleanly frozen or
ready for external evaluation would overstate the evidence.

## Corrective Action

1. Do not rerun the consumed 750-case bank.
2. Do not tune from its perfect aggregate result.
3. Add end-of-run manifest verification before the next one-shot protocol.
4. Build a new internally withheld bank under a new version and seed.
5. Freeze the unchanged safety candidate plus corrected evaluation harness.
6. Run the new bank once; only a start-and-end integrity pass can authorize a
   new independently authored external holdout.

This incident does not change DEP-001: it remains blocked. Nothing here is
clinical validation or evidence of real-patient safety.
