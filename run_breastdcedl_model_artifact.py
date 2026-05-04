import argparse
import json

from backend.database import SessionLocal
from backend.schema_migrations import ensure_schema
from backend.services.model_artifacts import (
    predict_breastdcedl_patient,
    train_and_register_breastdcedl_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train/register and test the BreastDCEDL model artifact.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--features", default="Data/breastdcedl_spy1_features.csv")
    parser.add_argument("--metrics", default="Data/breastdcedl_spy1_baseline_metrics.json")
    parser.add_argument("--artifact-dir", default="Data/models")
    parser.add_argument("--patient-id", default="ISPY1_1001")
    parser.add_argument("--shap", default="Data/breastdcedl_spy1_shap_explanations.json")
    args = parser.parse_args()

    ensure_schema()
    db = SessionLocal()
    try:
        training_result = train_and_register_breastdcedl_model(
            db=db,
            version=args.version,
            features_csv_path=args.features,
            metrics_path=args.metrics,
            artifact_dir=args.artifact_dir,
        )
        prediction_result = predict_breastdcedl_patient(
            db=db,
            patient_id=args.patient_id,
            model_version=args.version,
            features_csv_path=args.features,
            shap_json_path=args.shap,
        )
    finally:
        db.close()

    print(json.dumps({
        "training": training_result,
        "prediction": prediction_result,
    }, indent=2))


if __name__ == "__main__":
    main()
