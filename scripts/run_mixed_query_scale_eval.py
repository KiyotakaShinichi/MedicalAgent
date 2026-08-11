from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.mixed_query_scale_eval import (  # noqa: E402
    DEFAULT_PER_FAMILY_N,
    DEFAULT_REAL_PIPELINE_SAMPLE_N,
    run_mixed_query_scale_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mixed KB, garbage, and dangerous-query stress evaluation.")
    parser.add_argument("--per-family-n", type=int, default=DEFAULT_PER_FAMILY_N)
    parser.add_argument("--real-pipeline-sample-n", type=int, default=DEFAULT_REAL_PIPELINE_SAMPLE_N)
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()
    payload = run_mixed_query_scale_eval(
        per_family_n=args.per_family_n,
        real_pipeline_sample_n=args.real_pipeline_sample_n,
        run_retrieval=not args.skip_retrieval,
    )
    print(json.dumps({
        "status": payload["status"],
        "query_count": payload["query_count"],
        "query_family_counts": payload["query_family_counts"],
        "summary": payload["summary"],
        "clinical_validation": payload["clinical_validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
