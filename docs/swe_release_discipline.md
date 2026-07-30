# Software Release Discipline

The project treats "done" as a gated state: tests, frontend checks, and
benchmark freshness must pass before demo or ship.

## Local Gates

```powershell
python -m pytest tests\test_breast_monitoring.py -q
python scripts\run_release_gate.py
python scripts\ship.py
```

`python scripts\ship.py` is the cross-platform release gate. It supports:

```powershell
python scripts\ship.py --tier fast
python scripts\ship.py --tier evidence
python scripts\ship.py --tier release
python scripts\ship.py --tier release --resume
python scripts\ship.py --list
```

- `fast` runs the core backend contract tests plus frontend unit, lint, and
  production-build checks.
- `evidence` refreshes the repeatable evaluation artifacts.
- `release` runs every selected test, frontend, evidence, and release-gate
  step. This remains the default.
- `--resume` reuses only previously passed steps whose command, environment,
  and relevant source/config/test dependency fingerprint is unchanged.

Each tier writes its own manifest under `Data/evals/ops/`. The frozen
adversarial v7 holdout is intentionally excluded because it is a one-pass
measurement, not a repeatable regression suite.

The release tier includes:

- backend breast-monitoring integration tests
- focused safety, RAG, ML/MLE, automation, infrastructure, and governance tests
- frontend Vitest
- frontend Playwright smoke tests
- frontend lint
- frontend production build
- repeatable evidence refreshes
- release artifact gate

On systems with `make`, `make ship` delegates to the same release discipline.

## Pre-Commit Gate

Install with:

```powershell
pip install pre-commit
pre-commit install
```

The local hook runs:

```powershell
python -m pytest tests\test_breast_monitoring.py -q
```

## Release Gate

The release gate reads explicit thresholds from:

```text
config/release_gate_thresholds.yaml
```

It fails when critical artifacts are missing, stale, below threshold, or in an
unaccepted status.

## CI

GitHub Actions now runs the same controllable proof set: backend tests,
`tests/test_breast_monitoring.py`, frontend Vitest, frontend build,
OpenAPI schema export/type generation freshness, RAG claim validation, live
RAG eval, synthetic hardening artifacts, the release gate, and Playwright smoke
coverage. Optional NLI is reported as its own artifact; it is not allowed to
silently become a clinical-validation claim.

## Admin Benchmark Shape

Legacy admin artifact endpoints are kept for backward compatibility. New
frontend work can use:

```text
GET /admin/benchmark-artifacts/{artifact_id}
```

which returns the normalized envelope:

```json
{
  "status": "...",
  "headline_metric": "...",
  "metrics": {},
  "rows": [],
  "artifact_path": "...",
  "last_run_at": "...",
  "claim_boundary": "...",
  "can_rerun": true,
  "errors": []
}
```

## What This Proves

This proves engineering release readiness for a synthetic-only prototype. It
does not prove clinical safety, clinical accuracy, real-world prediction
validity, or clinician approval.
