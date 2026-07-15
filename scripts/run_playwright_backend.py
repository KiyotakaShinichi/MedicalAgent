"""Run Playwright against a disposable, seeded SQLite database.

This process intentionally uses ports and storage separate from the developer
demo stack. Browser tests may write records, but they must never mutate the
live P001 demo database.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DB = ROOT / "Data" / "runtime" / "playwright_test.db"
DATABASE_URL = "sqlite:///./Data/runtime/playwright_test.db"
API_PORT = 8117
FRONTEND_ORIGIN = "http://127.0.0.1:5273"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    TEST_DB.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DATABASE_URL"] = DATABASE_URL
    env["ONCOTRACK_CORS_ORIGINS"] = FRONTEND_ORIGIN
    env["ENVIRONMENT"] = "test"

    subprocess.run(
        [
            sys.executable,
            "scripts/reset_local_db.py",
            "--database-url",
            DATABASE_URL,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    os.environ.update(
        {
            "DATABASE_URL": DATABASE_URL,
            "ONCOTRACK_CORS_ORIGINS": FRONTEND_ORIGIN,
            "ENVIRONMENT": "test",
        }
    )
    import uvicorn

    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=API_PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
