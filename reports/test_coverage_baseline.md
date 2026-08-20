# Backend test and coverage baseline

Recorded: 2026-08-20

This is an engineering coverage baseline for the Python backend. It is not
clinical validation, safety certification, or evidence of real-patient
performance.

## Initial complete offline run

The first complete run after introducing the coverage job produced:

- 1,734 passed, 20 failed, 138 warnings.
- 48,357 measured statements; 35,398 covered lines.
- Line coverage: 70.07%.
- Branch coverage: 58.26% (7,466 of 12,814 branches).
- Configured initial floor: 35% total backend coverage.
- Coverage floor: passed; full test job: failed because functional regressions
  are never averaged away by coverage.

The failures clustered around safety-routing precedence, stale assertions,
requirements-pin parsing, and an evaluator that counted validator-prevented
generations as final unsafe leakage. They were treated as behavior defects, not
as exclusions from the suite.

## Major uncovered surfaces

The baseline JSON report identifies zero-execution paths in several categories:

- Optional public-dataset import and imaging-model utilities, including
  `breastdcedl_cnn.py`, `breastdcedl_importer.py`, and
  `breastdcedl_previews.py`.
- Offline report and failure-analysis builders, including
  `chat_latency_report.py`, `citation_precision_failure_analysis.py`, and
  `dataset_lineage.py`.
- Consumed-bank construction and historical DEP-001 experiment modules. These
  must not be executed merely to increase coverage; immutable evaluation
  integrity takes precedence over a percentage.

Coverage should increase through focused tests around active runtime contracts,
not through executing network downloads, consumed banks, or deprecated
experiment paths.

## Warning debt

The run emitted repeated unclosed SQLite connection warnings from test fixture
lifecycle paths. They do not invalidate line coverage, but they are genuine
resource-cleanup debt and should be eliminated before raising the coverage
floor.

## Gate policy

CI runs the complete offline suite with branch coverage and fails below 35%.
This deliberately conservative first floor prevents a regression to no
measurement while the repository stabilizes. Raise it in small increments only
after the full job is consistently green and high-value untested runtime paths
gain focused coverage.

Machine-readable evidence:

- `reports/coverage.json`
- `reports/coverage.xml`

## Remediated complete offline run

After isolating demo-auth fixtures and correcting privacy-process routing
precedence, the same complete offline command produced:

- 1,772 passed, 0 failed, 150 warnings in 1,011.09 seconds.
- 48,428 measured statements; 35,374 covered lines.
- Statement coverage: 73.04%.
- Branch coverage: 58.09% (7,460 of 12,842 branches).
- Combined branch-aware coverage: 69.91%.
- Configured 35% coverage floor: passed.

The remaining warnings are dominated by unclosed SQLite connections in legacy
test and evaluation lifecycles. The functional suite is green, but that
resource-cleanup debt remains an explicit maintainability issue.
