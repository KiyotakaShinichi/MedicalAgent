# How To Use The Pre-commit Gate

OncoTrack has two supported local commit gates. Both run the same core
integration check:

```bash
RAG_FORCE_SPARSE=true python -m pytest tests/test_breast_monitoring.py -q --tb=line
```

This catches chat/RAG/citation/cache/safety regressions before an AI-assisted
edit is called "done."

## Option A: Git hooks path

```bash
git config core.hooksPath .githooks
```

## Option B: Install into `.git/hooks`

```bash
python scripts/install_pre_commit.py
```

## Option C: Python `pre-commit` package

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Bypass Policy

Only bypass when the failure is unrelated and a follow-up fix is already
tracked:

```bash
SKIP_ONCOTRACK_GATE=1 git commit -m "..."
```

Do not bypass the gate to ship broken RAG/citation/cache behavior.
