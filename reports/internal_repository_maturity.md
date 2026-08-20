# Internal repository maturity estimate

Status: verified internal engineering estimate; release remains blocked.

These are internal engineering estimates only. They are not a third-party
score, clinical validation, a security certification, or production-healthcare
readiness.

| Dimension | Estimate / 10 | Evidence | Main cap |
|---|---:|---|---|
| Architecture | 8.0 | Modular FastAPI/React system; two responsibility-driven splits | Safety-critical orchestrators remain large |
| Code quality | 7.5 | Ruff correctness gate and focused contract tests | Legacy style and dynamic typing debt |
| Maintainability | 7.5 | Environment/dependency contracts and module ownership docs | Large long-running suite; broad service surface |
| Testing | 8.5 | 1,772-test complete offline run passed; 73.04% statement and 58.09% branch coverage; frontend and API contracts pass | SQLite resource warnings and a conservative 35% CI floor remain |
| Documentation | 8.5 | Setup, environment, security, contribution, release, provenance reports | Many historical docs still compete for authority |
| Security | 7.5 | Redacted structured logs, clean frozen-graph audits, fail-closed safety gates | DEP-001 remains behaviorally blocked |
| Dependency health | 8.0 | Exact pins, cross-platform uv lock, drift checks | Ecosystem advisories require successful remote audit |
| Reproducibility | 8.0 | Frozen lock, pinned direct requirements, deterministic commands | Optional model/data assets retain external dependencies |
| CI/CD | 8.0 | Static, full-suite, coverage, audit, frontend, and existing gates | Canonical release gate has 6 failures and 25 appendix warnings; no clean tag |
| Operational maturity | 7.5 | JSON logs, request IDs, liveness/readiness, runtime evidence | No sustained staging load or production traffic |
| Repository hygiene | 6.5 | Changelog, contributing guide, focused-change policy | Large dirty worktree and historical mixed changes |
| Licensing clarity | 5.0 | Explicit provenance and distribution boundary audit | No root license; KB/data terms unresolved |

Internal overall repository maturity: **7.7/10**. The full-suite result raises
testing credibility, while the estimate remains capped by blocked independent
safety evidence, six canonical release-gate failures, licensing uncertainty,
SQLite lifecycle warnings, and the absence of a clean tagged development
release.
