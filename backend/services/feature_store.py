import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE_STORE_DIR = "Data/feature_store"
FEATURE_STORE_SCHEMA_VERSION = "local_feature_store_v1"


def materialize_feature_store(
    source_csv,
    output_dir=DEFAULT_FEATURE_STORE_DIR,
    entity_column="patient_id",
    exclude_columns=None,
):
    source_path = Path(source_csv)
    if not source_path.exists():
        raise FileNotFoundError(f"Feature source CSV not found: {source_csv}")
    frame = pd.read_csv(source_path)
    if entity_column not in frame.columns:
        raise ValueError(f"entity_column={entity_column} not found in {source_csv}")

    exclude = set(exclude_columns or [])
    feature_columns = [column for column in frame.columns if column not in exclude]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    offline_path = output_path / "offline_features.csv"
    manifest_path = output_path / "feature_store_manifest.json"
    frame[feature_columns].to_csv(offline_path, index=False)

    manifest = {
        "schema_version": FEATURE_STORE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_path),
        "source_sha256": _sha256(source_path),
        "offline_features_path": str(offline_path),
        "offline_features_sha256": _sha256(offline_path),
        "entity_column": entity_column,
        "row_count": int(len(frame)),
        "entity_count": int(frame[entity_column].nunique()),
        "feature_columns": feature_columns,
        "dtypes": {column: str(frame[column].dtype) for column in feature_columns},
        "missing_rates": {
            column: round(float(frame[column].isna().mean()), 6)
            for column in feature_columns
        },
        "purpose": "Local account-free feature store manifest for training/serving feature consistency.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_feature_store_manifest(manifest_path=None, output_dir=DEFAULT_FEATURE_STORE_DIR):
    path = Path(manifest_path) if manifest_path else Path(output_dir) / "feature_store_manifest.json"
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    offline_path = Path(payload.get("offline_features_path") or "")
    current_hash = _sha256(offline_path) if offline_path.exists() else None
    status = "current" if current_hash == payload.get("offline_features_sha256") else "stale"
    return {
        **payload,
        "status": status,
        "manifest_path": str(path),
        "current_offline_features_sha256": current_hash,
    }


def load_feature_row(entity_id, manifest_path=None, output_dir=DEFAULT_FEATURE_STORE_DIR):
    manifest = load_feature_store_manifest(manifest_path=manifest_path, output_dir=output_dir)
    if manifest.get("status") not in {"current", "stale"}:
        raise FileNotFoundError(f"Feature store manifest not available: {manifest.get('path')}")
    frame = pd.read_csv(manifest["offline_features_path"])
    entity_column = manifest["entity_column"]
    row = frame[frame[entity_column] == entity_id]
    if row.empty:
        raise ValueError(f"No feature row found for {entity_column}={entity_id}")
    return row.head(1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
