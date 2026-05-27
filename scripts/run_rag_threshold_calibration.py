"""Run the RAG-agent threshold sensitivity sweep.

Writes ``Data/evals/rag/latest_rag_threshold_calibration.json``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.rag_threshold_calibration import (  # noqa: E402
    OUTPUT_PATH,
    build_calibration_report,
    write_calibration_report,
)


def main() -> int:
    out = write_calibration_report(OUTPUT_PATH)
    report = build_calibration_report()
    print(f"wrote: {out}")
    for name, block in report["constants"].items():
        verdict = block.get("verdict")
        print(f"  {name:32s} default={block['default']}  verdict={verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
