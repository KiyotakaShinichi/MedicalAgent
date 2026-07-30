from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_feature_policy import (
    write_synthetic_feature_policy,
)


if __name__ == "__main__":
    result = write_synthetic_feature_policy()
    policy = result["canonical_promotion_policy"]
    print(
        f"status={result['status']} policy_id={result['policy_id']} "
        f"features={len(policy['numeric_features'])} "
        f"removed={policy['removed_features']}"
    )
