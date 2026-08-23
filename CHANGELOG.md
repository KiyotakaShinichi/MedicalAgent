# Changelog

All notable repository changes will be recorded here. This project has not yet
published a tagged release; historical entries are intentionally not invented.

## Unreleased

### Added

- Responsibility-named modules for the SaaS control plane, governance
  credibility artifacts, admin analytics, and the synthetic perturbation
  retrain evaluation, each behind its original module as a compatibility
  facade.
- A shrink-only backend service size ratchet
  (`scripts/check_file_size.py`) that freezes pre-existing debt and blocks new
  oversized modules.
- A source-plus-tests discipline check
  (`scripts/check_change_discipline.py`) with a structural, AST-based
  documentation-only exemption.
- Full-suite fresh-clone verification with hermetic accounting: every test file
  executed, live credentials stripped, non-loopback egress blocked, and any
  skip attributable to a missing network or credential treated as a failure.
- An empty-model-cache proof that the semantic safety runtimes fail *closed*
  rather than open when their encoder is unavailable.
- `backend/app_logging.py` as the canonical logging entrypoint, with an
  explicit python-json-logger import.
- Retrieval index state on `GET /health`, alongside database connectivity and
  the running version.
- Explicit HTTP 422 request-validation evidence for the patient write
  boundaries.
- Canonical `pyproject.toml` and `uv.lock` dependency workflow.
- Ruff, incremental mypy, frontend lint/typecheck, full offline coverage, and
  dependency-audit CI gates.
- Canonical structured JSON logging and bounded liveness/readiness probes.
- Source-enforced environment configuration documentation.
- Contributor, dependency-security, licensing/provenance, and repository
  hardening documentation.
- Complete offline backend coverage execution with a hard CI floor.
- Focused regression tests for privacy-process education and isolated demo-auth
  fixtures.

### Changed

- `GET /health` now reports `status`, `service`, `version`, database
  reachability, and whether the retrieval index is loaded in this process. All
  dependency fields are informational: the endpoint returns 200 while the
  process is alive, and answering it never loads an index or blocks on a hung
  database. `/ready` remains the authoritative, fail-closed readiness signal.
- Dependency profiles are uv dependency groups rather than separate manifests;
  `uv sync --frozen --no-default-groups` installs the minimal serving runtime.
- Dependabot groups Python updates by stack (ml, reporting, dev tooling,
  runtime) so a model-library bump is reviewed apart from a web-framework bump.
- Pinned direct Python dependencies and updated vulnerable Pillow, Torch, and
  Setuptools versions in the canonical lock inputs.
- Split admin observability routes and MLE statistical diagnostics into
  responsibility-named modules without changing their public contracts.
- Corrected privacy-process routing so consent education remains answerable
  while mixed disclosure demands and prompt-injection instructions fail closed.

### Security

- Request logging redacts sensitive identifiers and prompt/token/password-like
  fields and avoids request bodies.
- Dependency auditing is release-blocking according to the documented policy.
- Patient demo authentication tests now create only their own synthetic fixture
  instead of depending on persistent local database state.

### Known limitations

- The semantic safety encoder is a ~480 MB multilingual transformer that must
  be provisioned once with network access. It cannot be replaced by a small
  bundled fixture: the tokenizer's 250k-token vocabulary alone accounts for
  ~384 MB of embedding weights, and a randomly-initialised stand-in would
  satisfy the loading contract while invalidating the classifier heads trained
  on the real encoder's embedding space.
- DEP-001 remains blocked by official behavioral evidence.
- No real patient data, clinical validation, clinician sign-off, IRB approval,
  healthcare compliance claim, or production healthcare readiness.
- The source code does not yet have an explicit root license, and some external
  knowledge/data assets have noncommercial, unknown, or dataset-specific terms.
