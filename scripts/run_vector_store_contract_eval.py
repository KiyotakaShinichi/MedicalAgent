from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.vector_store_contract_eval import build_vector_store_contract_eval


if __name__ == "__main__":
    print(json.dumps(build_vector_store_contract_eval(), indent=2))
