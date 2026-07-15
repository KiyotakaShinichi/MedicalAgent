"""Cross-platform pre-commit integration gate for NLCare."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env = os.environ.copy()
    env["RAG_FORCE_SPARSE"] = "true"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_breast_monitoring.py",
        "-q",
        "--tb=line",
    ]
    print("[oncotrack] running breast-monitoring integration gate")
    print("[oncotrack] " + " ".join(command))
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
