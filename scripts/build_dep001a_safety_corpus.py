from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001a_safety_corpus import build_corpora


if __name__ == "__main__":
    manifest = build_corpora()
    print(json.dumps({
        "dataset_version": manifest["dataset_version"],
        "development_n": manifest["development"]["n"],
        "validation_n": manifest["validation"]["n"],
        "contains_final_holdout_examples": manifest["provenance"]["contains_final_holdout_examples"],
    }, indent=2))
