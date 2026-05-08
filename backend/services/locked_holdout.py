import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_TRAINING_ROWS_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_DIR = "Data/complete_synthetic_training/locked_holdout"


def create_locked_holdout_manifest(
    training_rows_path=DEFAULT_TRAINING_ROWS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    holdout_size=0.20,
    seed=314159,
    target="treatment_success_binary",
):
    rows = pd.read_csv(training_rows_path)
    if "patient_id" not in rows.columns or target not in rows.columns:
        raise ValueError(f"Expected patient_id and {target} columns in {training_rows_path}")

    patient_labels = (
        rows.groupby("patient_id", as_index=False)[target]
        .max()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    train_ids, holdout_ids = train_test_split(
        patient_labels["patient_id"],
        test_size=holdout_size,
        random_state=seed,
        stratify=patient_labels[target].astype(int),
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    locked_rows = rows[rows["patient_id"].isin(set(holdout_ids))].copy()
    development_rows = rows[rows["patient_id"].isin(set(train_ids))].copy()
    locked_rows_path = output_path / "locked_holdout_rows.csv"
    development_rows_path = output_path / "development_rows.csv"
    locked_rows.to_csv(locked_rows_path, index=False)
    development_rows.to_csv(development_rows_path, index=False)

    payload = {
        "schema_version": "locked_holdout_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_rows_path": training_rows_path,
        "target": target,
        "seed": seed,
        "holdout_size": holdout_size,
        "dataset_hash": _sha256(training_rows_path),
        "development_patients": int(len(train_ids)),
        "locked_holdout_patients": int(len(holdout_ids)),
        "development_rows": int(len(development_rows)),
        "locked_holdout_rows": int(len(locked_rows)),
        "development_positive_rate": float(patient_labels[patient_labels["patient_id"].isin(set(train_ids))][target].mean()),
        "locked_holdout_positive_rate": float(patient_labels[patient_labels["patient_id"].isin(set(holdout_ids))][target].mean()),
        "locked_holdout_patient_ids_sha256": hashlib.sha256(
            "\n".join(sorted(holdout_ids)).encode("utf-8")
        ).hexdigest(),
        "files": {
            "locked_holdout_rows_csv": str(locked_rows_path),
            "development_rows_csv": str(development_rows_path),
            "manifest_json": str(output_path / "locked_holdout_manifest.json"),
        },
        "usage_policy": (
            "Keep this split frozen for model comparison. Do not tune thresholds, features, or generator logic "
            "against the locked holdout."
        ),
        "claim_boundary": "Locked synthetic holdout supports engineering discipline only; it is not external clinical validation.",
    }
    Path(payload["files"]["manifest_json"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
