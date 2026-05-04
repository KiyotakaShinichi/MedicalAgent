import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from backend.models import ModelRegistry, PredictionAuditLog
from backend.services.breastdcedl_baseline import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    _logistic_regression_pipeline,
)
from backend.services.breastdcedl_xai import load_patient_shap_explanation


DEFAULT_MODEL_NAME = "breastdcedl_pcr_logreg"
DEFAULT_MODEL_VERSION = "v1"
DEFAULT_FEATURES_CSV_PATH = "Data/breastdcedl_spy1_features.csv"
DEFAULT_METRICS_PATH = "Data/breastdcedl_spy1_baseline_metrics.json"
DEFAULT_SHAP_PATH = "Data/breastdcedl_spy1_shap_explanations.json"
DEFAULT_ARTIFACT_DIR = "Data/models"
DEFAULT_COMPLETE_SYNTHETIC_METRICS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
DEFAULT_COMPLETE_SYNTHETIC_TRAINING_DATA_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_COMPLETE_SYNTHETIC_ARTIFACT_DIR = "Data/complete_synthetic_training"


def train_and_register_breastdcedl_model(
    db,
    version: str = DEFAULT_MODEL_VERSION,
    features_csv_path: str = DEFAULT_FEATURES_CSV_PATH,
    metrics_path: str = DEFAULT_METRICS_PATH,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
):
    features = _load_training_features(features_csv_path)
    X = features[FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    y = features["pcr_label"].astype(int)

    model = _logistic_regression_pipeline()
    model.fit(X, y)

    artifact_path = Path(artifact_dir) / f"{DEFAULT_MODEL_NAME}_{_safe_version(version)}.joblib"
    metadata_path = Path(artifact_dir) / f"{DEFAULT_MODEL_NAME}_{_safe_version(version)}_metadata.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = _load_json_if_exists(metrics_path)
    metadata = {
        "model_name": DEFAULT_MODEL_NAME,
        "model_version": version,
        "task": "BreastDCEDL pCR treatment-response classification",
        "target": "pCR positive / pathologic complete response",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(features)),
        "positive_pcr": int(y.sum()),
        "negative_pcr": int((y == 0).sum()),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "metrics": metrics,
        "training_dataset_hash": _file_sha256(features_csv_path),
        "metrics_hash": _file_sha256(metrics_path),
        "promotion_status": "candidate",
        "promotion_reason": (
            "Registered as a candidate PoC artifact. Promote only after evaluation report review, "
            "calibration checks, and clinician-loop acceptance review."
        ),
        "registry_schema_version": "model_registry_metadata_v2",
        "warning": "Exploratory PoC model only. Not clinically validated.",
    }
    bundle = {
        "model": model,
        "metadata": metadata,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
    }
    joblib.dump(bundle, artifact_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    registry_row = _upsert_model_registry(
        db=db,
        model_name=DEFAULT_MODEL_NAME,
        version=version,
        task=metadata["task"],
        artifact_path=str(artifact_path),
        metrics_path=metrics_path,
        training_data_path=features_csv_path,
        metadata=metadata,
    )

    return {
        "message": "Final BreastDCEDL logistic regression artifact trained and registered.",
        "model_registry_id": registry_row.id,
        "model_name": DEFAULT_MODEL_NAME,
        "model_version": version,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "training_rows": metadata["training_rows"],
        "positive_pcr": metadata["positive_pcr"],
        "negative_pcr": metadata["negative_pcr"],
        "warning": metadata["warning"],
    }


def predict_breastdcedl_patient(
    db,
    patient_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    features_csv_path: str = DEFAULT_FEATURES_CSV_PATH,
    shap_json_path: str = DEFAULT_SHAP_PATH,
):
    registry_row = _get_model_registry_row(db, model_name, model_version)
    if registry_row is None:
        raise FileNotFoundError(
            f"No registered model found for {model_name} {model_version}. "
            "Run /models/breastdcedl/train-final first."
        )

    artifact_path = Path(registry_row.artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Registered model artifact is missing: {artifact_path}")

    bundle = joblib.load(artifact_path)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    categorical_columns = bundle.get("categorical_feature_columns", CATEGORICAL_FEATURE_COLUMNS)
    patient_row = _load_patient_feature_row(features_csv_path, patient_id)
    X = patient_row[feature_columns + categorical_columns]

    pcr_probability = float(model.predict_proba(X)[:, 1][0])
    predicted_label = int(pcr_probability >= 0.5)
    explanation = load_patient_shap_explanation(patient_id, shap_json_path)
    prediction_payload = {
        "patient_id": patient_id,
        "model_name": model_name,
        "model_version": model_version,
        "task": "BreastDCEDL pCR treatment-response classification",
        "pcr_probability": round(pcr_probability, 6),
        "predicted_label": predicted_label,
        "actual_pcr_label": _safe_int(patient_row.iloc[0].get("pcr_label")),
        "model_interpretation": _interpret_pcr_probability(pcr_probability),
        "safety_note": (
            "This is an exploratory treatment-response model signal from a PoC dataset. "
            "It is not diagnosis, treatment advice, or a clinically validated score."
        ),
    }
    input_reference = {
        "features_csv_path": features_csv_path,
        "patient_id": patient_id,
        "feature_columns": feature_columns + categorical_columns,
    }
    audit_log = PredictionAuditLog(
        patient_id=patient_id,
        model_name=model_name,
        model_version=model_version,
        input_reference=json.dumps(input_reference),
        prediction_json=json.dumps(prediction_payload),
        explanation_json=json.dumps(explanation) if explanation else None,
    )
    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    prediction_payload["audit_log_id"] = audit_log.id
    prediction_payload["xai"] = explanation
    return prediction_payload


def list_registered_models(db):
    rows = (
        db.query(ModelRegistry)
        .order_by(ModelRegistry.created_at.desc(), ModelRegistry.id.desc())
        .all()
    )
    return [_registry_row_to_dict(row) for row in rows]


def register_complete_synthetic_champion(
    db,
    version: str = "synthetic-v1",
    metrics_path: str = DEFAULT_COMPLETE_SYNTHETIC_METRICS_PATH,
    training_data_path: str = DEFAULT_COMPLETE_SYNTHETIC_TRAINING_DATA_PATH,
    artifact_dir: str = DEFAULT_COMPLETE_SYNTHETIC_ARTIFACT_DIR,
    promotion_status: str = "candidate",
    promotion_reason: str | None = None,
):
    metrics = _load_json_if_exists(metrics_path)
    if not metrics:
        raise FileNotFoundError(f"Complete synthetic metrics file not found: {metrics_path}")

    model_name = metrics.get("best_model_by_patient_level_roc_auc")
    if not model_name:
        raise ValueError("Metrics file does not contain best_model_by_patient_level_roc_auc")
    target = metrics.get("task") or "treatment_success_binary"
    artifact_path = _complete_synthetic_artifact_path(artifact_dir, model_name, target)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Champion artifact not found: {artifact_path}")

    metadata = {
        "model_name": model_name,
        "model_version": version,
        "task": f"Complete synthetic longitudinal breast cancer {target}",
        "target": target,
        "model_family": (metrics.get("models") or {}).get(model_name, {}).get("model_type"),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source": "complete_synthetic_longitudinal_breast_cancer_journeys",
        "metrics": (metrics.get("models") or {}).get(model_name, {}),
        "training_rows": metrics.get("train_rows"),
        "test_rows": metrics.get("test_rows"),
        "patients": metrics.get("patients"),
        "training_dataset_hash": _file_sha256(training_data_path),
        "metrics_hash": _file_sha256(metrics_path),
        "artifact_hash": _file_sha256(artifact_path),
        "promotion_status": promotion_status,
        "promotion_reason": promotion_reason or "Registered as synthetic champion for engineering evaluation only.",
        "registry_schema_version": "model_registry_metadata_v2",
        "warning": "Synthetic-data champion only. This does not prove real clinical performance.",
    }

    row = _upsert_model_registry(
        db=db,
        model_name=f"complete_synthetic_{model_name}",
        version=version,
        task=metadata["task"],
        artifact_path=str(artifact_path),
        metrics_path=metrics_path,
        training_data_path=training_data_path,
        metadata=metadata,
    )
    return _registry_row_to_dict(row)


def get_prediction_audit_logs(db, patient_id: str | None = None, limit: int = 50):
    query = db.query(PredictionAuditLog)
    if patient_id:
        query = query.filter(PredictionAuditLog.patient_id == patient_id)
    rows = (
        query.order_by(PredictionAuditLog.created_at.desc(), PredictionAuditLog.id.desc())
        .limit(limit)
        .all()
    )
    return [_audit_row_to_dict(row) for row in rows]


def _load_training_features(features_csv_path):
    path = Path(features_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    features = pd.read_csv(path)
    features = features[features["pcr_label"].notna()].copy()
    if len(features) < 20:
        raise ValueError("Need at least 20 feature rows to train the final model artifact")
    return features


def _load_patient_feature_row(features_csv_path, patient_id):
    path = Path(features_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    features = pd.read_csv(path)
    row = features[features["patient_id"] == patient_id]
    if row.empty:
        raise ValueError(f"No BreastDCEDL feature row found for patient_id={patient_id}")
    return row.head(1)


def _upsert_model_registry(
    db,
    model_name,
    version,
    task,
    artifact_path,
    metrics_path,
    training_data_path,
    metadata,
):
    row = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_name == model_name)
        .filter(ModelRegistry.model_version == version)
        .first()
    )
    if row is None:
        row = ModelRegistry(model_name=model_name, model_version=version)
        db.add(row)

    row.task = task
    row.artifact_path = artifact_path
    row.metrics_path = metrics_path
    row.training_data_path = training_data_path
    row.model_metadata_json = json.dumps(metadata)
    row.status = "active"
    db.commit()
    db.refresh(row)
    return row


def _get_model_registry_row(db, model_name, model_version):
    return (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_name == model_name)
        .filter(ModelRegistry.model_version == model_version)
        .filter(ModelRegistry.status == "active")
        .first()
    )


def _registry_row_to_dict(row):
    return {
        "id": row.id,
        "model_name": row.model_name,
        "model_version": row.model_version,
        "task": row.task,
        "artifact_path": row.artifact_path,
        "metrics_path": row.metrics_path,
        "training_data_path": row.training_data_path,
        "metadata": _loads_or_none(row.model_metadata_json),
        "status": row.status,
        "created_at": str(row.created_at),
    }


def _audit_row_to_dict(row):
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "model_name": row.model_name,
        "model_version": row.model_version,
        "input_reference": _loads_or_none(row.input_reference),
        "prediction": _loads_or_none(row.prediction_json),
        "explanation": _loads_or_none(row.explanation_json),
        "created_at": str(row.created_at),
    }


def _load_json_if_exists(path):
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _loads_or_none(value):
    if not value:
        return None
    return json.loads(value)


def _safe_version(version):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in version)


def _safe_int(value):
    if pd.isna(value):
        return None
    return int(value)


def _file_sha256(path):
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_synthetic_artifact_path(artifact_dir, model_name, target):
    extension = ".pt" if model_name in {"temporal_1d_cnn", "temporal_gru"} else ".joblib"
    return Path(artifact_dir) / f"{model_name}_{target}{extension}"


def _interpret_pcr_probability(probability):
    if probability >= 0.70:
        band = "higher_pcr_probability"
        message = "The model leans toward pCR/favorable complete response for this dataset task."
    elif probability <= 0.40:
        band = "lower_pcr_probability"
        message = "The model leans away from pCR and toward non-pCR for this dataset task."
    else:
        band = "uncertain_middle_probability"
        message = "The model signal is mixed or uncertain."

    return {
        "band": band,
        "message": message,
        "threshold_note": "0.5 is the demo classification threshold; the probability itself is more informative.",
    }
