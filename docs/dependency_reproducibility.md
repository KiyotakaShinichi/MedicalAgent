# Dependency reproducibility

`pyproject.toml` and `uv.lock` are the canonical cross-platform dependency
contract. Direct runtime and development dependencies are exact-pinned in the
project manifest; the lock records the resolved transitive graph for supported
Python versions. CI must install with a pinned uv version and `uv sync --frozen`
so dependency resolution cannot drift during a build.

Update dependencies intentionally:

```powershell
uv lock --upgrade-package <package-name>
uv sync --frozen
python scripts/check_dependency_contract.py
python -m pip check
```

`requirements.txt` and `requirements-serving.txt` remain exact-pinned
compatibility manifests for legacy tooling and container profiles. They are not
the canonical transitive lock.

The generated `requirements-lock-py314-win.txt` remains a historical snapshot
of the known-good CPython 3.14 Windows engineering environment, together with
its interpreter/platform fingerprint. Refresh and audit it only from that
environment:

```powershell
.\.venv\Scripts\python.exe scripts\run_dependency_lock_audit.py --refresh-transitive-lock
```

The audit reports direct-lock drift, transitive graph drift, missing or extra
distributions, the lock SHA-256, and an environment-fingerprint mismatch.

The Windows snapshot is intentionally platform-scoped and is not the CI lock.
Run dependency and container audits separately; a reproducible graph is not a
security certification. Deployment still requires reviewed image digests,
SBOMs, transitive/container scans, signing, and isolated verification. None of
these controls establishes compliance or makes NLCare healthcare production
ready.
