"""Build the synthetic data quality (generator-proxy) report.

Writes ``Data/evals/realism/latest_synthetic_data_quality.json``.

Usage::

    python scripts/run_synthetic_data_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.synthetic_data_quality import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    DEFAULT_ROWS_PATH,
    write_quality_report,
)


def main() -> int:
    out = write_quality_report(DEFAULT_ROWS_PATH, DEFAULT_OUTPUT_PATH)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
