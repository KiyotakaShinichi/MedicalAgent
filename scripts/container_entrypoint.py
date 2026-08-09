"""Shell-free backend startup for the distroless serving image."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    profile = str(
        os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV") or "development"
    ).strip().lower()
    validation = [sys.executable, "scripts/validate_deployment_env.py"]
    if profile in {"staging", "production", "prod"}:
        validation.append("--strict")
    subprocess.run(validation, check=True)
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8017",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
