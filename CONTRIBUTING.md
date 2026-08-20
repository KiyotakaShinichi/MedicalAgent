# Contributing to NLCare

NLCare is a synthetic-only, nonclinical healthcare AI engineering prototype.
Contributions must preserve that boundary and must not introduce diagnostic,
treatment, prognostic, genetic-risk, tumor-marker, medication, or patient-care
authority.

## Setup

```bash
python -m pip install uv==0.8.24
uv sync --frozen
cd frontend-react
npm ci
```

Use Python 3.11 for the canonical CI environment. Start from `.env.example`,
keep `NLCARE_SYNTHETIC_ONLY=true`, and never commit secrets, raw patient data,
or a local `.env` file.

## Branches and commits

- Use `codex/<short-purpose>` or another short feature branch agreed by the
  maintainer.
- Prefer focused conventional commits such as `fix(api): ...`, `test(rag): ...`,
  or `docs(env): ...`.
- Keep behavior and its tests in the same commit.
- Do not rewrite history, fabricate contributors, or split work merely to
  increase commit count.

## Required checks

```bash
uv lock --check
uv run python scripts/check_dependency_contract.py
uv run python scripts/check_env_documentation.py
uv run ruff check backend scripts tests
uv run mypy
uv run pytest tests -q --cov=backend --cov-branch --cov-fail-under=35
cd frontend-react
npm run lint
npm run typecheck
npm run test
npm run build
```

Run `python scripts/ship.py` only after the focused checks pass. Network and
paid-provider calls must be mocked or explicitly isolated from offline tests.

## Safety and evidence integrity

- Never rerun, inspect for tuning, modify, relabel, or copy consumed blind or
  one-shot DEP-001 banks.
- Do not move final holdout examples into prompts, rules, training data, tests,
  or development fixtures.
- Preserve candidate, bank, receipt, and result hashes. Protocol-owned evidence
  changes require their documented workflow.
- Any unsafe released output is a blocker. Do not average severe safety failures
  into an aggregate score.
- Keep internal, frozen, synthetic, external-prepared, and external-completed
  evidence labels distinct.
- Negative results and failed gates stay visible.

## Pull requests

Describe the problem, responsibility boundary, behavioral effect, tests run,
security/privacy impact, and remaining limitations. UI changes should include
desktop and mobile evidence. Changes to RAG, safety, ML promotion, or automation
must identify the affected release gates and prove that legitimate educational
usefulness has not regressed.
