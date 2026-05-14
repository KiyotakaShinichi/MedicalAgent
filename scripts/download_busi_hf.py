from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/convert the small BUSI ultrasound mirror from Hugging Face.")
    parser.add_argument("--repo-id", default="Angelou0516/BUSI")
    parser.add_argument("--filename", default="data/train-00000-of-00001.parquet")
    parser.add_argument("--output-root", default="Datasets/BUSI")
    parser.add_argument("--cache-dir", default="Datasets/_hf_cache/Angelou0516_BUSI")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for a quick smoke download. 0 means all rows.")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Missing dependency. Run: .\\.venv\\Scripts\\python.exe -m pip install huggingface_hub pyarrow"
        ) from exc

    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=args.filename,
        local_dir=args.cache_dir,
    )
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    manifest_rows = []
    for row in rows:
        label = str(row.get("class_label") or "unknown").strip().lower()
        image_id = str(row.get("image_id") or f"{label}_{counts.get(label, 0) + 1}")
        safe_id = _safe_name(image_id)
        label_dir = output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        image_payload = row.get("image") or {}
        mask_payload = row.get("mask") or {}
        image_path = label_dir / f"{safe_id}.png"
        mask_path = label_dir / f"{safe_id}_mask.png"
        image_path.write_bytes(image_payload.get("bytes") or b"")
        mask_path.write_bytes(mask_payload.get("bytes") or b"")

        counts[label] = counts.get(label, 0) + 1
        manifest_rows.append({
            "image_id": image_id,
            "class_label": label,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "has_multifocal_lesion": bool(row.get("has_multifocal_lesion")),
        })

    manifest_path = output_root / "busi_manifest.json"
    manifest = {
        "schema_version": "busi_hf_mirror_v1",
        "source_repo": args.repo_id,
        "source_file": args.filename,
        "output_root": str(output_root),
        "row_count": len(manifest_rows),
        "class_counts": counts,
        "license_note": (
            "Hugging Face mirror states the original BUSI authors did not publish an explicit license. "
            "Treat as research-use with citation; do not redistribute blindly."
        ),
        "citation": "Al-Dhabyani et al. Dataset of breast ultrasound images. Data in Brief, 2020.",
        "rows": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "downloaded",
        "source": args.repo_id,
        "output_root": str(output_root),
        "row_count": len(manifest_rows),
        "class_counts": counts,
        "manifest_path": str(manifest_path),
    }, indent=2))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "image"


if __name__ == "__main__":
    main()
