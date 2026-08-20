# Dependency security policy

NLCare uses `pyproject.toml` plus `uv.lock` as the canonical Python dependency
contract and `frontend-react/package-lock.json` as the frontend lock. The
compatibility requirement files are pinned, but they are not the resolver of
record.

## Automated checks

CI runs:

```bash
uv lock --check
uv run python scripts/check_dependency_contract.py
uv run python -m pip check
uv export --frozen --no-dev --format requirements-txt --no-hashes --quiet \
  --output-file /tmp/nlcare-production-requirements.txt
uv run pip-audit -r /tmp/nlcare-production-requirements.txt \
  --no-deps --disable-pip --progress-spinner off --timeout 60
cd frontend-react && npm audit --audit-level=high
```

The exported Python inventory contains the complete resolved production graph,
so `--no-deps` avoids constructing a second environment while still auditing
transitive packages.

## Finding policy

- Python: any known vulnerability fails CI unless a time-bounded exception is
  documented with package, advisory, exposure analysis, owner, expiry, and
  compensating control.
- npm critical/high: fail CI.
- npm moderate/low: review during dependency updates and document material
  runtime exposure; they do not automatically block this prototype.
- Audit-service outage: the audit is inconclusive, not a clean result. CI should
  be rerun; release preparation remains blocked until a successful query.

## Current remediation

The 2026-08-20 local audit found vulnerable `pillow==12.2.0`,
`torch==2.11.0`, and transitive `setuptools==81.0.0`. The lock inputs were
updated to Pillow `12.3.0`, Torch `2.13.0`, Torchvision `0.28.0`, and Setuptools
`83.0.0`, which are the advisory-listed fixed versions. A fresh audit of the
frozen production graph then reported no known vulnerabilities, and the
frontend audit reported zero vulnerabilities. CI repeats both audits; a future
advisory-service outage remains inconclusive rather than clean.

This policy concerns software dependencies only. It does not establish HIPAA,
clinical, regulatory, or real-patient deployment readiness.
