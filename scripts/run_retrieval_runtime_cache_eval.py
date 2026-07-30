from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.retrieval_runtime_cache_eval import write_retrieval_runtime_cache_eval


if __name__ == "__main__":
    result = write_retrieval_runtime_cache_eval()
    comparison = result["comparison"]["retrieval_p95_ms"]
    print(
        "retrieval cache eval:",
        result["status"],
        f"p95 {comparison['baseline_ms']}ms -> {comparison['current_ms']}ms",
    )
