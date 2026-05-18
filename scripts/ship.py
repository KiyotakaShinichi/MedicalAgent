"""Cross-platform OncoTrack ship gate.

Runs the same checks as ``make ship`` without requiring GNU Make. The script
stops on the first failed command and returns that command's exit code.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend-react"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    cwd: Path = ROOT
    env: dict[str, str] | None = None


def _npm_cmd(*args: str) -> list[str]:
    executable = "npm.cmd" if os.name == "nt" else "npm"
    return [executable, *args]


def _run(step: Step) -> None:
    env = os.environ.copy()
    if step.env:
        env.update(step.env)
    print(f"\n[ship] {step.name}")
    print(f"[ship] cwd={step.cwd}")
    print(f"[ship] cmd={' '.join(step.command)}")
    subprocess.run(step.command, cwd=step.cwd, env=env, check=True)


def main() -> int:
    steps = [
        Step(
            name="Backend breast-monitoring integration tests",
            command=[sys.executable, "-m", "pytest", "tests/test_breast_monitoring.py", "-q"],
            env={"RAG_FORCE_SPARSE": "true"},
        ),
        Step(
            name="Frontend Vitest unit tests",
            command=_npm_cmd("run", "test"),
            cwd=FRONTEND,
        ),
        Step(
            name="Frontend Playwright smoke",
            command=_npm_cmd("run", "test:e2e", "--", "tests/e2e/smoke.spec.ts"),
            cwd=FRONTEND,
        ),
        Step(
            name="Frontend lint",
            command=_npm_cmd("run", "lint"),
            cwd=FRONTEND,
        ),
        Step(
            name="Frontend production build",
            command=_npm_cmd("run", "build"),
            cwd=FRONTEND,
        ),
        Step(
            name="Release artifact gate",
            command=[sys.executable, "scripts/run_release_gate.py"],
        ),
    ]
    for step in steps:
        try:
            _run(step)
        except subprocess.CalledProcessError as exc:
            print(f"\n[ship] FAILED: {step.name} exited {exc.returncode}", file=sys.stderr)
            return int(exc.returncode or 1)
    print("\n[ship] PASSED: all gates green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
