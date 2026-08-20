from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001c_blind_bank import build_and_freeze_dep001c_blind_bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--burned-external-path", type=Path)
    args = parser.parse_args()
    manifest = build_and_freeze_dep001c_blind_bank(external_path=args.burned_external_path)
    print(json.dumps({
        "blind_bank_id": manifest["blind_bank_id"],
        "case_n": manifest["case_n"],
        "blind_bank_sha256": manifest["blind_bank_sha256"],
    }, indent=2))
