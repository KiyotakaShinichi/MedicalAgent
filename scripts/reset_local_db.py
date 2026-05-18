"""Reset the local SQLite demo database.

This script is intentionally limited to SQLite database URLs. It will refuse
to touch Postgres/MySQL style URLs so a developer cannot accidentally wipe a
shared database while trying to refresh the local demo.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_URL = "sqlite:///./medical_agent.db"


def _sqlite_path(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError(
            "reset_local_db.py only supports sqlite:/// URLs. "
            f"Refusing to reset {database_url!r}."
        )
    parsed = urlparse(database_url)
    raw_path = parsed.path
    if database_url.startswith("sqlite:///./"):
        raw_path = database_url.removeprefix("sqlite:///")
    if raw_path in {"", ":memory:"}:
        raise ValueError("Refusing to reset an empty or in-memory SQLite URL.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset the local OncoTrack SQLite demo database.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL))
    parser.add_argument("--no-seed", action="store_true", help="Create schema only; skip demo data seeding.")
    args = parser.parse_args(argv)

    db_path = _sqlite_path(args.database_url)
    if db_path.exists():
        db_path.unlink()
        print(f"[reset-db] removed {db_path}")
    else:
        print(f"[reset-db] no existing database at {db_path}")

    env = os.environ.copy()
    env["DATABASE_URL"] = args.database_url
    print("[reset-db] applying Alembic migrations")
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)

    if not args.no_seed:
        print("[reset-db] seeding demo patient data")
        subprocess.run([sys.executable, "seed_db.py"], cwd=ROOT, env=env, check=True)

    print("[reset-db] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
