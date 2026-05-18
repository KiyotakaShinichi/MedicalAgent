# OncoTrack pre-commit integration gate

## What it does

Runs `tests/test_breast_monitoring.py` before every commit and refuses the
commit on any failure. This is the repo's cheapest cross-module integration
suite: chat, RAG, citations, safety scope, synthetic dataset loading, and the
agent regression benchmark all pass through it.

The hook sets `RAG_FORCE_SPARSE=true` so local commits do not depend on a dense
embedding model download.

## Install

Either install the tracked hook path:

```bash
git config core.hooksPath .githooks
```

Or write the hook directly into `.git/hooks/pre-commit`:

```bash
python scripts/install_pre_commit.py
```

Or use the Python `pre-commit` framework:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The framework hook is defined in `.pre-commit-config.yaml` and calls
`scripts/precommit_integration_gate.py`.

## Run on demand

```bash
RAG_FORCE_SPARSE=true python -m pytest tests/test_breast_monitoring.py -q --tb=line
```

PowerShell:

```powershell
$env:RAG_FORCE_SPARSE='true'; python -m pytest tests\test_breast_monitoring.py -q --tb=line
```

## Bypass

Only bypass when the failure is unrelated and a follow-up fix is already
tracked:

```bash
SKIP_ONCOTRACK_GATE=1 git commit -m "..."
```
