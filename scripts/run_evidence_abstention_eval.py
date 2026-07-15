"""Run the evidence-abstention sweep and write the artifact.

Exits non-zero when overall_status is `needs_attention` so CI can gate on
the abstention contract not silently regressing.
"""
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.evidence_abstention_eval import run_evidence_abstention_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=600,
        help=(
            "Bounded sample size for routine release runs. Use 0 for the full "
            "synthetic table; full sweeps can take several minutes on CPU."
        ),
    )
    args = parser.parse_args()
    payload = run_evidence_abstention_eval(
        sample_size=None if args.sample_size == 0 else args.sample_size,
    )
    summary = payload.get("summary", {})
    print(json.dumps({
        "output_path": "Data/evals/models/latest_evidence_abstention_eval.json",
        "rows_evaluated": payload.get("rows_evaluated"),
        "status": payload.get("status"),
        "full_data_coverage_rate": summary.get("full_data_coverage_rate"),
        "full_data_covered_accuracy": summary.get("full_data_covered_accuracy"),
        "demographics_only_abstention_rate": summary.get("demographics_only_abstention_rate"),
        "scenario_count": summary.get("scenario_count"),
    }, indent=2))
    sys.exit(0 if payload.get("status") in {"strong", "acceptable"} else 1)
