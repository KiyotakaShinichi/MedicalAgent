import json
from pathlib import Path

import joblib
import pandas as pd

from backend.models import ModelRegistry, PredictionAuditLog
from backend.services.app_logging import log_app_event
from backend.services.breastdcedl_baseline import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS
from backend.services.breastdcedl_xai import load_patient_shap_explanation


SERVICE_SCHEMA_VERSION = "local_model_inference_service_v1"


class LocalModelInferenceService:
    backend_name = "local_artifact_loader"

    def predict_breastdcedl_patient(
        self,
        db,
        patient_id: str,
        model_name: str,
        model_version: str,
        features_csv_path: str,
        shap_json_path: str,
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
            "inference_backend": self.backend_name,
            "service_schema_version": SERVICE_SCHEMA_VERSION,
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
            "artifact_path": str(artifact_path),
            "inference_backend": self.backend_name,
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
        log_app_event(
            db=db,
            event_type="prediction",
            patient_id=patient_id,
            route="/models/breastdcedl/predict/{patient_id}",
            status="ok",
            input_payload=input_reference,
            output_payload={
                "model_name": model_name,
                "model_version": model_version,
                "pcr_probability": prediction_payload["pcr_probability"],
                "predicted_label": predicted_label,
                "audit_log_id": audit_log.id,
                "inference_backend": self.backend_name,
            },
        )
        return prediction_payload


def get_inference_service():
    return LocalModelInferenceService()


def describe_inference_service():
    return {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "active_backend": LocalModelInferenceService.backend_name,
        "purpose": "Stable inference boundary between FastAPI routes and model artifact loading.",
        "future_backends": ["bentoml", "ray_serve", "triton", "remote_http_model_server"],
        "safety_note": "Inference outputs remain exploratory PoC signals unless clinically validated.",
    }


def _get_model_registry_row(db, model_name, model_version):
    return (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_name == model_name)
        .filter(ModelRegistry.model_version == model_version)
        .filter(ModelRegistry.status.in_(["active", "champion"]))
        .first()
    )


def _load_patient_feature_row(features_csv_path, patient_id):
    path = Path(features_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    features = pd.read_csv(path)
    row = features[features["patient_id"] == patient_id]
    if row.empty:
        raise ValueError(f"No BreastDCEDL feature row found for patient_id={patient_id}")
    return row.head(1)


def _safe_int(value):
    if pd.isna(value):
        return None
    return int(value)


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
