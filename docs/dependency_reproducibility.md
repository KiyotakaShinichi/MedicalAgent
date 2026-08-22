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

They are kept because something real installs from them: the container image
builds with plain `pip` and never runs `uv` (`Dockerfile`,
`ARG REQUIREMENTS_FILE=requirements.txt`), and `requirements-serving.txt` is the
deliberately smaller runtime profile that omits the training, deep-learning,
SHAP, and evaluation stack. `requirements.txt` additionally serves pip-only
contributors, so it may carry development extras such as the test runner.

Because they are hand-maintained exports of a canonical source, they can drift
from it while staying perfectly pinned — two files can both be exact-pinned to
*disagreeing* versions, which ships one dependency set to the container and a
different one to every test. `scripts/check_dependency_contract.py` therefore
enforces agreement, not just pinning:

- every runtime dependency in `pyproject.toml` appears in `requirements.txt` at
  the same version;
- an extra entry that `pyproject.toml` also declares in a dependency group must
  match that group's pin;
- `requirements-serving.txt` is a subset of `requirements.txt` with matching
  versions.

CI runs this check in `static-quality`, and `tests/test_dependency_contract.py`
covers each drift mode. When you change a dependency, update `pyproject.toml`
first, re-lock, then mirror the pin into the compatibility manifests.

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
