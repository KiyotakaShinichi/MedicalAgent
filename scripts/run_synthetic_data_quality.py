"""Build the synthetic data quality (generator-proxy) report.

Writes ``Data/evals/realism/latest_synthetic_data_quality.json`` and the
compatibility alias ``Data/evals/models/latest_synthetic_data_quality_report.json``.

Usage::

    python scripts/run_synthetic_data_quality.py
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.synthetic_data_quality import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    DEFAULT_MODELS_OUTPUT_PATH,
    DEFAULT_ROWS_PATH,
    write_quality_report,
)


def main() -> int:
    out = write_quality_report(DEFAULT_ROWS_PATH, DEFAULT_OUTPUT_PATH)
    DEFAULT_MODELS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, DEFAULT_MODELS_OUTPUT_PATH)
    print(f"wrote: {out}")
    print(f"wrote: {DEFAULT_MODELS_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
