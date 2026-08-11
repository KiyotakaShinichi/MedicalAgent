"""Provision local synthetic demo accounts without resetting user-entered data."""

from __future__ import annotations

import json
import os

from backend.database import SessionLocal
from backend.services.synthetic_demo_bootstrap import (
    ensure_synthetic_demo_data,
    synthetic_demo_bootstrap_allowed,
)


def main() -> int:
    if not synthetic_demo_bootstrap_allowed(os.environ):
        raise RuntimeError(
            "Synthetic demo bootstrap requires an explicit nonproduction synthetic profile."
        )

    db = SessionLocal()
    try:
        result = ensure_synthetic_demo_data(db)
    finally:
        db.close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

