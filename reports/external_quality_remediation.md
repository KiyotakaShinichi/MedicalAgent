# External repository-quality remediation

Assessment baseline: 61/C (external assessment supplied by the repository
owner). All statuses below are internal engineering findings. No third-party
rescore, clinical validation, or healthcare deployment approval is claimed.

## Original findings

| # | Original issue | Remediation and evidence | Status | Remaining limitation |
|---:|---|---|---|---|
| 1 | Python dependencies unpinned; no lock | Exact direct pins in `pyproject.toml`, compatibility requirements, and canonical `uv.lock`; `scripts/check_dependency_contract.py` and CI `uv lock --check` | Remediated | Lock portability still requires CI on supported Python/OS combinations |
| 2 | No enforced lint configuration | Ruff correctness baseline in `pyproject.toml`; CI checks backend, scripts, and tests | Remediated incrementally | The baseline intentionally does not enforce broad style rules across legacy code |
| 3 | CI lacks lint/typecheck | Static-quality job runs Ruff, incremental mypy, frontend ESLint, and TypeScript no-emit | Remediated | Mypy covers five high-value boundary modules, not the entire backend |
| 4 | CI runs a curated test subset | Added complete offline pytest job with network/provider isolation conventions; final local run: 1,772 passed, 0 failed | Remediated | Full suite is long-running and still exposes SQLite fixture cleanup debt |
| 5 | No backend coverage threshold | Branch coverage enabled with a 35% initial hard floor; final run measured 73.04% statement, 58.09% branch, and 69.91% combined coverage | Remediated | Floor is intentionally lower than current measured coverage and must rise gradually |
| 6 | Large mixed-responsibility modules | Extracted admin observability router and MLE statistics diagnostics with contract tests | Partially remediated | `support_chat_agent.py`, `rag_evidence_envelope.py`, and a large admin React section remain |
| 7 | No structured logging detected | Canonical JSON logger with schema, request/trace IDs, component/severity fields, redaction, and safe exception metadata; middleware/runtime adoption | Remediated for highest-value paths | Legacy modules still contain ad hoc logging and need staged migration |
| 8 | No health endpoint detected | Cheap `/health` liveness and dependency-aware `/ready` with stable nonclinical boundary fields | Remediated | Readiness proves service dependencies only, not healthcare readiness |
| 9 | `.env.example` incomplete | Expanded inventory, `docs/environment_configuration.md`, source/example contract check and tests | Remediated | Dynamic or third-party configuration may still require future inventory updates |
| 10 | No CI dependency audit | Python exported-lock audit, npm high-severity audit, `pip check`, and documented finding policy | Remediated and locally verified | Remote advisory-service availability remains an external dependency |
| 11 | No contributing guide or changelog | Added `CONTRIBUTING.md` and an honest `CHANGELOG.md` beginning at Unreleased | Remediated | Future entries must remain tied to real changes |
| 12 | No tagged release | Added blocked `v0.1.0` development-release draft and tag procedure | Intentionally blocked | DEP-001, six canonical release-gate failures, licensing, and clean-worktree requirements remain |
| 13 | Low mineable task capacity | Adopted focused-change guidance and paired new runtime/refactor behavior with tests | Prospective only | History was not rewritten; this improves only through future genuine work |

## Quality categories

### Dependency health and experiment reproducibility

`uv.lock` is the resolver of record. Direct pins, lock drift, environment
documentation, and install consistency are checked independently. Model and
evaluation claims remain tied to explicit artifacts and contamination status.

### Code cleanliness and architecture

Ruff catches high-confidence runtime errors without a repository-wide rewrite.
Two responsibility-driven module splits landed. Safety-critical orchestrators
were deliberately left intact until equivalence and fresh independent evidence
can support a lower-risk refactor.

### Test coverage and CI/CD maturity

CI now separates static quality, full offline tests with branch coverage,
dependency security, frontend quality, and the existing RAG/ML/safety gates.
Coverage does not override functional failures.

### Documentation and onboarding

README first-screen positioning, uv quickstart, environment inventory,
contribution rules, dependency-security policy, release draft, changelog, and
provenance audit now provide one reproducible onboarding path.

### Security hygiene

Logs redact secrets and sensitive fields, health probes avoid expensive work,
dependency audits are policy-enforced, and existing safety/release gates remain
unchanged. Fresh frozen-graph Python and npm audits reported no known
vulnerabilities locally; CI repeats them.

### History, maintenance, and licensing

No history, contributors, tags, or evidence were fabricated. The repository has
no root code license, and third-party KB/data redistribution is not cleared.
The licensing audit therefore blocks a distributable release rather than
inventing permission.

## Release verdict

`BLOCKED_RELEASE`. Repository engineering posture is materially stronger and
the complete offline suite is green, but the canonical release gate still
reports 6 failures across 257 artifacts, with 25 appendix warnings. `v0.1.0`
must not be tagged until those decision failures pass on independently valid
evidence, the worktree is clean and reviewable, DEP-001 is no longer
behaviorally blocked, and licensing/provenance blockers are resolved.

## Final verification snapshot

Recorded on 2026-08-20 against the pinned project environment:

- `uv lock --check`, dependency-contract validation, environment-contract
  validation (83 variables), and `pip check`: passed.
- Ruff and incremental mypy over the declared boundary: passed.
- Frontend clean install, ESLint, TypeScript no-emit, 60 Vitest tests, and
  production build: passed.
- Complete offline Python suite: 1,772 passed, 0 failed, 150 warnings.
- Backend coverage: 73.04% statements, 58.09% branches, 69.91% combined;
  35% CI floor passed.
- Frozen Python production dependency audit and npm audit: no known
  vulnerabilities reported; secret scan passed.
- Workflow YAML and `git diff --check`: passed (Windows line-ending notices
  only).
- Canonical release gate: failed with 6 decision failures and 25 appendix
  warnings across 257 artifacts. Existing RAG and ML core gates were largely
  green; DEP-001 behavioral evidence and stale critical artifacts remain red.

Primary machine-readable evidence is in `reports/coverage.json` and
`reports/coverage.xml`. The release-gate policy source is
`config/release_gate_thresholds.yaml`; the human-readable current result is
recorded here because the gate CLI reports to stdout and does not persist a new
artifact.
