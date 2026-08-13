#!/usr/bin/env python
"""Run the frozen DEP-001 final-output safety assurance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The final bank must pass with the optional LLM adjudicator unavailable.
os.environ["ONCOTRACK_FAST_MODE"] = "1"
os.environ["LLM_ADJUDICATION_ENABLED"] = "false"
os.environ["RAG_FORCE_SPARSE"] = "1"

from backend.services.dep001_safety_evaluation import (  # noqa: E402
    DEFAULT_FINAL_BANK,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    evaluate_dep001_bank,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=DEFAULT_FINAL_BANK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_dep001_bank(args.bank, manifest_path=args.manifest, output_path=args.output)
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2))
    return 0 if not report["deployment_blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
