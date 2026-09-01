"""Refresh buyer integrity digests from immutable Git blobs, never checkout bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.buyer.contracts import load_json, tracked_files_bytes  # noqa: E402


def main() -> int:
    path = ROOT / "config/buyer/protected_evidence_manifest.json"
    payload = load_json(path)
    paths = [entry["path"] for entry in payload["files"]]
    committed = tracked_files_bytes(paths)
    for entry in payload["files"]:
        entry["sha256"] = hashlib.sha256(committed[entry["path"]]).hexdigest()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(paths)} canonical Git-blob hashes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
