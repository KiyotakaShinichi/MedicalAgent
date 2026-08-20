from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_official_evaluation import run_dep001d_official_internal_once


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--blind-bank-id", required=True)
    args = parser.parse_args()
    result = run_dep001d_official_internal_once(
        candidate_id=args.candidate_id,
        blind_bank_id=args.blind_bank_id,
    )
    print(json.dumps({
        "run_id": result["run_id"],
        "status": result["status"],
        "candidate_id": result["candidate_id"],
        "blind_bank_id": result["blind_bank_id"],
        "cases_evaluated": result["cases_evaluated"],
        "metrics": result["metrics"],
        "integrity_valid": result["integrity_valid"],
        "ready_for_new_external_holdout": result["ready_for_new_external_holdout"],
        "dep001_status": result["dep001_status"],
    }, indent=2))
