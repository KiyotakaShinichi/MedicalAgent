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

### Verifying a fresh clone

The offline test path must run from a clean checkout with **no `.env`**, no
prebuilt RAG index, and no network access. Confirm that before opening a PR
that touches configuration, fixtures, or test bootstrapping:

```bash
python scripts/check_fresh_clone_offline.py --run-tests
```

It fails if the offline path starts depending on an untracked or gitignored
file — the failure mode where a repository runs only on the machine that
generated its artifacts. CI runs the same command on its own fresh checkout.

The check covers *repository content sufficiency* only. Dependency resolution
and installation are a separate contract, verified by `uv lock --check` and
`scripts/check_dependency_contract.py`.

### Dependency updates

`.github/dependabot.yml` opens weekly, grouped, **review-based** update PRs for
Python (uv), frontend npm, GitHub Actions, and the Docker base image. Nothing
auto-merges, and no auto-merge workflow exists. A dependency bump can move model
output or refusal routing without failing a type check, so every PR gets read and
the full offline suite runs before it lands. Majors for the numerical/ML stack
(`numpy`, `pandas`, `scikit-learn`, `torch`, `transformers`) are excluded from
automation and handled as separately-evidenced migrations.

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
uv run python scripts/check_fresh_clone_offline.py
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

### Lint policy

Ruff enforces `E4`, `E7`, `E9`, `F`, and `B` — import placement, statement-level
correctness, syntax/runtime errors, full pyflakes, and bugbear's real bug
patterns. The rationale for each family, and for the families deliberately left
off (`E501`, `I`, `UP`, `SIM`), is documented inline in `[tool.ruff.lint]` in
`pyproject.toml` with the measured violation counts behind each decision.

Two rules to know about when you hit them:

- **Fix violations; do not blanket-ignore them.** Repository-wide `ignore`
  entries and new `per-file-ignores` need a written reason in `pyproject.toml`
  saying why the pattern is deliberate. The existing `E402` exemptions are
  `sys.path` bootstrapping and genuine circular-import breaks, listed file by
  file so the rule still protects everything else.
- **A `# noqa` needs a comment saying why.** Two exist today, both marking
  latent defects that need domain review rather than a lint-pass rewrite; each
  is labelled `LATENT DEFECT` in the source.

Do not widen the rule set and mass-reformat in the same change. Enabling a
family means fixing its violations in a focused, reviewable diff.

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
