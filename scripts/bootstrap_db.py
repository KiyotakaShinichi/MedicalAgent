"""Bootstrap (or reset) the local NLCare SQLite database.

The on-disk ``medical_agent.db`` is **not** tracked in git — see
``docs/local_db.md`` for the why and how.  This script creates a fresh
copy from the SQLAlchemy schema + Alembic migration history so a new
checkout can start the backend without hand-copying a binary.

Usage
~~~~~
    python scripts/bootstrap_db.py            # create if missing, otherwise no-op
    python scripts/bootstrap_db.py --reset    # delete + recreate

Honors the ``DATABASE_URL`` env var.  For non-sqlite URLs the ``--reset``
flag is refused (we won't drop a Postgres DB by accident).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sqlite_path_from_url(url: str) -> Path | None:
    """Return the on-disk path for a ``sqlite:///`` URL, else None."""
    if not url.startswith("sqlite:///"):
        return None
    raw = url[len("sqlite:///") :]
    if raw.startswith("/"):
        return Path(raw)
    return (ROOT / raw).resolve()


def _create_schema() -> None:
    """Create every table from the SQLAlchemy metadata + run schema migrations."""
    from backend.database import Base, engine
    from backend import models  # noqa: F401 — registers tables on Base.metadata
    from backend.schema_migrations import ensure_schema

    Base.metadata.create_all(bind=engine)
    ensure_schema()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap or reset the NLCare local DB")
    parser.add_argument("--reset", action="store_true", help="Delete the DB file (sqlite only) before recreating.")
    args = parser.parse_args(argv)

    url = os.environ.get("DATABASE_URL", "sqlite:///./medical_agent.db")
    sqlite_path = _sqlite_path_from_url(url)

    if args.reset:
        if sqlite_path is None:
            print(f"[bootstrap_db] refusing --reset on non-sqlite URL: {url}", file=sys.stderr)
            return 2
        if sqlite_path.exists():
            sqlite_path.unlink()
            print(f"[bootstrap_db] deleted {sqlite_path}")

    if sqlite_path is not None and sqlite_path.exists():
        print(f"[bootstrap_db] DB already exists at {sqlite_path} — nothing to do.")
        print("[bootstrap_db] pass --reset to recreate.")
        return 0

    _create_schema()
    if sqlite_path is not None:
        print(f"[bootstrap_db] created {sqlite_path}")
    else:
        print(f"[bootstrap_db] applied schema to {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
