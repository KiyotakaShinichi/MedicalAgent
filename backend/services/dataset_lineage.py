import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RESPONSE_REGRESSION_TARGET


DEFAULT_DATASET_DIR = "Data/complete_synthetic_breast_journeys"
DEFAULT_OUTPUT_PATH = "Data/lineage/complete_synthetic_lineage.json"


def build_complete_synthetic_lineage(
    dataset_dir=DEFAULT_DATASET_DIR,
    output_path=DEFAULT_OUTPUT_PATH,
):
    dataset_path = Path(dataset_dir)
    files = []
    for csv_path in sorted(dataset_path.glob("*.csv")):
        files.append(_csv_fingerprint(csv_path))

    summary = _load_json(dataset_path / "summary.json")
    data_dictionary = _load_json(dataset_path / "data_dictionary.json")
    feature_lineage = _feature_lineage(data_dictionary)
    combined_hash = _combined_hash(files)

    payload = {
        "schema_version": "dataset_lineage_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_path),
        "dataset_hash": combined_hash,
        "source": summary.get("source", "synthetic_complete_breast_journey"),
        "generation_options": summary.get("generation_options", {}),
        "patients_created": summary.get("patients_created"),
        "cycles_per_patient": summary.get("cycles_per_patient"),
        "table_counts": summary.get("table_counts", {}),
        "files": files,
        "feature_contract": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "response_regression_target": RESPONSE_REGRESSION_TARGET,
            "label_columns": [
                "treatment_success_binary",
                "final_response_category",
                "final_cancer_status",
                "toxicity_risk_binary",
                "support_intervention_needed",
                "urgent_intervention_needed",
                "cycle_response_trend_class",
            ],
        },
        "feature_lineage": feature_lineage,
        "claim_boundary": "Synthetic data lineage for engineering reproducibility only; not clinical provenance.",
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _csv_fingerprint(path):
    frame = pd.read_csv(path, nrows=25)
    return {
        "table": path.stem,
        "path": str(path),
        "sha256": _sha256(path),
        "row_count": _row_count(path),
        "columns": frame.columns.tolist(),
        "schema_signature": _schema_signature(frame),
    }


def _row_count(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_signature(frame):
    schema = "|".join(f"{column}:{str(dtype)}" for column, dtype in frame.dtypes.items())
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _combined_hash(files):
    digest = hashlib.sha256()
    for item in files:
        digest.update(item["table"].encode("utf-8"))
        digest.update(item["sha256"].encode("utf-8"))
        digest.update(item["schema_signature"].encode("utf-8"))
    return digest.hexdigest()


def _load_json(path):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _feature_lineage(data_dictionary):
    return {
        "cycle": "treatment_sessions.cycle and temporal_ml_rows.cycle",
        "demographics": "patients.age",
        "diagnosis_profile": "diagnoses.stage, diagnoses.molecular_subtype, diagnoses.receptor_status",
        "regimen": "treatment_sessions.regimen",
        "pre_cycle_labs": "labs rows where lab_timepoint=pre_cycle",
        "nadir_labs": "labs rows where lab_timepoint=post_cycle_nadir",
        "recovery_labs": "labs rows where lab_timepoint=recovery",
        "mri_response_features": "mri_reports tumor size and percent_change_from_baseline available up to current cycle",
        "symptom_features": "symptoms aggregated inside the current treatment cycle",
        "intervention_features": "interventions aggregated inside the current treatment cycle",
        "outcome_labels": "outcomes final response labels attached after journey generation and excluded from model features",
        "table_purposes": data_dictionary,
    }
