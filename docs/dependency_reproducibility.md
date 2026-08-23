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

## Install profiles

There is no `requirements.txt` and no `requirements-serving.txt`. Profiles are
uv dependency groups, so there is exactly one place a version is written:

| profile | command | contents |
| --- | --- | --- |
| development / CI | `uv sync --frozen` | project dependencies + `dev`, `ml`, `reporting` |
| minimal serving | `uv sync --frozen --no-default-groups` | project dependencies only |

`[project].dependencies` is the minimal request-path runtime. The heavier
stacks live in groups:

- `ml` — torch, torchvision, sentence-transformers, shap;
- `reporting` — matplotlib, reportlab;
- `dev` — pytest, ruff, mypy, pip-audit, and friends.

`[tool.uv] default-groups` lists all three, so a plain `uv sync --frozen` still
installs exactly what it installed when these packages were flat runtime
dependencies. The groups exist so the *serving image* can opt out, not so the
default environment can shrink. `scripts/check_dependency_contract.py` fails if
a declared group is missing from `default-groups`, because that would quietly
change what every developer and CI job installs without any file appearing to
move.

The container resolves the same lockfile rather than a manifest maintained
beside it. `Dockerfile` takes `ARG PYTHON_PROFILE` (`full` or `serving`) and
runs the matching `uv export`; `docker-compose.synthetic-staging.yml` builds
with `PYTHON_PROFILE: serving`.

## Why the manifests were removed

They were hand-maintained exports of a canonical source, and being exact-pinned
never made them correct: two files can both be perfectly pinned to
*disagreeing* versions, which ships one dependency set to the container and a
different one to every test. That failure was only detectable by a contract
that compared them. The contract is now simply that they do not exist —
`scripts/check_dependency_contract.py` fails if either path reappears, and says
what to do instead. CI runs that check in `static-quality`, and
`tests/test_dependency_contract.py` covers each failure mode.

To add or change a dependency: edit `pyproject.toml`, run `uv lock`, and commit
the lockfile. There is no second file to mirror it into.


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
