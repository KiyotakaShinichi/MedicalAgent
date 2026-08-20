# Changelog

All notable repository changes will be recorded here. This project has not yet
published a tagged release; historical entries are intentionally not invented.

## Unreleased

### Added

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

- DEP-001 remains blocked by official behavioral evidence.
- No real patient data, clinical validation, clinician sign-off, IRB approval,
  healthcare compliance claim, or production healthcare readiness.
- The source code does not yet have an explicit root license, and some external
  knowledge/data assets have noncommercial, unknown, or dataset-specific terms.
