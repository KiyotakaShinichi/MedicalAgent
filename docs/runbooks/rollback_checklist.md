# Rollback Checklist

Use this when a release or demo branch regresses safety, retrieval, or core
dashboard behavior.

- Identify the commit or artifact version that introduced the regression.
- Preserve failing artifacts for review.
- Restore the last green KB index, model artifact, or code commit.
- Rerun `python scripts/ship.py`.
- Regenerate release-gate explanation.
- Update demo notes with what was rolled back and why.

Rollback does not erase the failure. Keep the evidence so the next fix can be
reviewed honestly.
