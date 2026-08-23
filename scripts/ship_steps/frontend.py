"""Frontend unit, e2e, lint, and production-build steps.

Extracted from ``scripts/ship.py`` as part of splitting a 477-line
``_build_steps``. Step definitions are relocated verbatim: the command, working
directory, environment, and timeout of every step are unchanged, and the order
within and across these modules reproduces the original list exactly.
"""

from __future__ import annotations

from scripts.ship_steps.common import FRONTEND, Step, npm_cmd

__all__ = ["frontend_steps"]


def frontend_steps() -> list[Step]:
    return [
                Step(
                    name="Frontend Vitest unit tests",
                    command=npm_cmd("run", "test"),
                    cwd=FRONTEND,
                ),
                Step(
                    name="Frontend Playwright smoke",
                    command=npm_cmd("run", "test:e2e", "--", "tests/e2e/smoke.spec.ts"),
                    cwd=FRONTEND,
                ),
                Step(
                    name="Frontend lint",
                    command=npm_cmd("run", "lint"),
                    cwd=FRONTEND,
                ),
                Step(
                    name="Frontend production build",
                    command=npm_cmd("run", "build"),
                    cwd=FRONTEND,
                ),
    ]
