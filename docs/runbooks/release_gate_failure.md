# Release Gate Failure Runbook

1. Read `Data/evals/governance/latest_release_gate_explanation.json`.
2. Separate hard blockers from warnings.
3. For hard blockers, rerun the exact artifact script listed in the gate.
4. Fix the root cause; do not relax thresholds unless the threshold is clearly
   wrong and the reason is documented.
5. Rerun `python scripts/ship.py`.
6. Record remaining warnings in the final release notes.

Never describe a red or stale gate as ready.
