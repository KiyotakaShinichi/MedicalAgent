# External Quality Remediation Baseline

Recorded: 2026-08-20

This baseline describes repository engineering posture only. NLCare remains a
synthetic-only, nonclinical engineering prototype. Nothing in this sprint is
clinical validation or production-healthcare readiness.

## Repository state

- Branch: `main`.
- History: 132 commits by one recorded author over the current repository
  history; no tags or releases.
- Worktree: dirty before this sprint with an extensive uncommitted DEP-001
  safety/evidence remediation. Those consumed-bank artifacts must not be
  modified or rerun.
- Tracked files: 2,447.
- `CONTRIBUTING.md`, `CHANGELOG.md`, a repository license, and a licensing and
  provenance inventory were absent.

## Current test and build commands

- Backend curated CI tests: selected pytest modules plus
  `python -m unittest tests/test_breast_monitoring.py`.
- Frontend: `npm ci`, `npm run build`, `npm run test`, and a Playwright smoke
  test.
- Cross-platform ship gate: `python scripts/ship.py`.
- Existing RAG, safety, MLE, OpenAPI drift, ingestion, API smoke, and release
  checks run in `.github/workflows/ci.yml`.
- The CI workflow did not run the complete offline pytest suite, Ruff, a Python
  type checker, backend coverage, `pip check`, or dependency audits.

## Dependency setup

- `requirements.txt` and `requirements-serving.txt` listed unpinned direct
  dependencies and were the files installed by CI and Docker paths.
- `requirements-lock.txt` pinned direct dependencies only.
- `requirements-lock-py314-win.txt` contained a 109-package local Windows
  Python 3.14 transitive snapshot, but it was environment-specific, unhashed,
  and not consumed by Linux/Python 3.11 CI.
- The frontend had a committed `package-lock.json` and CI correctly used
  `npm ci`.
- There was no canonical `pyproject.toml`, dependency update workflow, or CI
  validation that the declared and installed Python environments agree.

## CI gates

Present: secret scan, Python compile, curated backend tests, legacy monitoring
regression, frontend build/tests, OpenAPI drift, RAG and MLE checks, release
gate, Playwright smoke, API smoke, and Docker build.

Missing or implicit: full offline pytest, Python lint, incremental Python type
checking, explicit frontend lint, standalone TypeScript no-emit gate, backend
coverage threshold, Python dependency consistency, and vulnerability policy.

## Observability

- `backend/services/structured_logging.py` existed and emitted JSON strings,
  but only one service imported it.
- No canonical application logging configuration or request middleware was
  detected.
- Request/correlation IDs were available in parts of the application, but were
  not consistently attached to runtime HTTP logs.
- Logging safety requirements were not centrally enforced; raw patient content,
  prompts, tokens, and secrets need explicit redaction policy.

## Health and readiness

- `/health` and `/ready` already existed.
- `/health` performed a database query, so it was a readiness check rather than
  a cheap process-liveness probe.
- `/ready` checked database reachability and retrieval readiness and preserved
  nonclinical claim boundaries, but dependency exceptions were not normalized
  into a stable readiness response.

## Environment documentation gaps

`.env.example` documented major database, LLM, RAG, vector-store, Redis, n8n,
and automation settings. Source inspection found additional runtime controls
for request limits, rate limits, auth/chat/upload limits, RRF candidate limits,
clinical-summary LLM behavior, and legacy compatibility names. There was no
automated source-to-example drift check.

## Largest maintainability surfaces

Generated OpenAPI types (9,988 lines) are intentionally generated and excluded
from refactor decisions. The largest authored modules were:

| File | Lines | Initial assessment |
|---|---:|---|
| `tests/test_breast_monitoring.py` | 2,172 | Large legacy integration suite; split only by fixture/domain ownership. |
| `scripts/generate_project_guide_pdf.py` | 1,468 | Artifact generator; low runtime risk. |
| `backend/api/routers/admin_eval.py` | 1,409 | High-value router responsibility split candidate. |
| `backend/services/mle_readiness.py` | 1,251 | Mixed evaluation/report assembly candidate. |
| `backend/services/support_chat_agent.py` | 1,213 | Safety-critical orchestrator; refactor only with strong characterization tests. |
| `backend/services/rag_evidence_envelope.py` | 1,126 | Safety-critical evidence boundary; avoid cosmetic splitting. |
| `frontend-react/src/pages/admin/sections/SafetyCenterSection.tsx` | 1,075 | UI composition split candidate. |
| `backend/services/agent_rag.py` | 1,055 | RAG orchestration; boundary already partly modularized. |

The first sprint should prefer `admin_eval.py` and one low-risk, coherent
reporting/UI boundary, not the safety-critical release path.

## Release state

- No tags or release notes exist.
- The release gate is intentionally failing on unresolved DEP-001 behavioral
  evidence. A first tag is therefore blocked even if repository quality checks
  become green.
- Release preparation must not weaken the safety gate or relabel internal
  evidence as external validation.

## Baseline decision

Proceed with dependency, CI, observability, onboarding, and provenance
hardening. Do not tag `v0.1.0` while DEP-001 remains blocked or while the full
offline quality stack has unresolved failures.
