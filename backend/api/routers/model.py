"""
Model router — registry, training, data generation, import, and prediction endpoints.

Covers /models/*, /train-*, /generate-*, /import-*, /build-*, /run-*, /index-*,
/preprocess-*, /inspect-*, and /prediction-audits.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import (
    get_admin_access_context,
    get_clinician_or_admin_context,
    get_db,
)

router = APIRouter(tags=["models"])


# ─── Request models ───────────────────────────────────────────────────────────

class CSVImportRequest(BaseModel):
    import_type: str
    dataset: str = "canonical"
    csv_text: str | None = None
    file_path: str | None = None


class QINBreast02ImportRequest(BaseModel):
    clinical_xlsx_path: str = "datasets/QIN-BREAST-02_clinicalData-Transformed-20191022-Revised20200210.xlsx"


class MRISeriesIndexRequest(BaseModel):
    root_path: str = "datasets/qin_breast_02"
    patient_id: str | None = None
    max_files: int | None = None


class MRIManifestRequest(BaseModel):
    clinical_xlsx_path: str | None = "Datasets/QIN-BREAST-02_clinicalData-Transformed-20191022-Revised20200210.xlsx"
    output_csv_path: str | None = "Data/qin_breast_02_mri_manifest.csv"


class MRIPreviewPreprocessRequest(BaseModel):
    manifest_csv_path: str = "Data/qin_breast_02_mri_manifest.csv"
    output_dir: str = "Data/qin_mri_previews"


class SyntheticJourneyRequest(BaseModel):
    count: int = 25
    seed: int = 42


class TemporalSyntheticJourneyRequest(BaseModel):
    count: int = 12
    seed: int = 2026
    cycles: int = 6


class CompleteSyntheticDatasetRequest(BaseModel):
    count: int = 60
    seed: int = 2027
    cycles: int = 6
    output_dir: str = "Data/complete_synthetic_breast_journeys"
    write_db: bool = True
    patient_prefix: str = "COMP-BRCA-"
    balanced_outcomes: bool = True
    balanced_subgroups: bool = True
    missing_rate: float = 0.04
    noise_level: float = 0.03


class CompleteSyntheticTrainingRequest(BaseModel):
    ml_csv_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
    output_dir: str = "Data/complete_synthetic_training"
    target: str = "treatment_success_binary"
    test_size: float = 0.25
    seed: int = 42
    cnn_epochs: int = 20
    cnn_batch_size: int = 16


class CompleteSyntheticXAIRequest(BaseModel):
    ml_csv_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
    model_path: str = "Data/complete_synthetic_training/logistic_regression_treatment_success_binary.joblib"
    predictions_csv_path: str = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
    output_json_path: str = "Data/complete_synthetic_training/synthetic_xai_explanations.json"
    top_n: int = 6


class CompleteSyntheticRegisterRequest(BaseModel):
    version: str = "synthetic-v1"
    metrics_path: str = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
    training_data_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
    artifact_dir: str = "Data/complete_synthetic_training"
    promotion_status: str = "candidate"
    promotion_reason: str | None = None


class ModelVersionActionRequest(BaseModel):
    reason: str | None = None


class BreastDCEDLInspectRequest(BaseModel):
    path: str = "Datasets/BreastDCEDL_spy1"


class BreastDCEDLManifestRequest(BaseModel):
    root_path: str = "Datasets/BreastDCEDL_spy1"
    output_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"


class BreastDCEDLBaselineRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    metrics_json_path: str = "Data/breastdcedl_spy1_baseline_metrics.json"
    predictions_csv_path: str = "Data/breastdcedl_spy1_model_predictions.csv"
    max_patients: int | None = None


class BreastDCEDLPreviewRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    output_dir: str = "Data/breastdcedl_previews"
    max_patients: int = 40


class BreastDCEDLImportRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    limit: int = 25


class BreastDCEDLCNNRequest(BaseModel):
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv"
    metrics_json_path: str = "Data/breastdcedl_spy1_cnn_metrics.json"
    max_patients: int = 120
    epochs: int = 4
    batch_size: int = 8


class BreastDCEDLXAIRequest(BaseModel):
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    output_json_path: str = "Data/breastdcedl_spy1_shap_explanations.json"
    top_n: int = 5


class BreastDCEDLModelTrainRequest(BaseModel):
    version: str = "v1"
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    metrics_path: str = "Data/breastdcedl_spy1_baseline_metrics.json"
    artifact_dir: str = "Data/models"


class BreastDCEDLModelPredictRequest(BaseModel):
    model_name: str = "breastdcedl_pcr_logreg"
    model_version: str = "v1"
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv"
    shap_json_path: str = "Data/breastdcedl_spy1_shap_explanations.json"


# ─── Import / schema ──────────────────────────────────────────────────────────

@router.get("/import-schema")
def get_import_schema():
    from backend.services.csv_importer import DATASET_ADAPTERS, SUPPORTED_IMPORT_TYPES

    return {
        "supported_import_types": sorted(SUPPORTED_IMPORT_TYPES),
        "supported_datasets": sorted(DATASET_ADAPTERS.keys()),
        "data_dictionary": "Data/breast_monitoring_data_dictionary.md",
    }


@router.post("/import-csv")
def import_csv_payload(
    payload: CSVImportRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.csv_importer import import_csv

    if not payload.csv_text and not payload.file_path:
        raise HTTPException(status_code=400, detail="Provide csv_text or file_path")

    try:
        result = import_csv(
            db=db,
            import_type=payload.import_type,
            csv_text=payload.csv_text,
            file_path=payload.file_path,
            dataset=payload.dataset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "CSV import completed",
        "import_type": payload.import_type,
        "dataset": payload.dataset,
        "result": result,
    }


@router.post("/import-qin-breast-02")
def import_qin_breast_02(payload: QINBreast02ImportRequest, db: Session = Depends(get_db)):
    from backend.services.csv_importer import import_qin_breast_02_clinical_xlsx

    try:
        result = import_qin_breast_02_clinical_xlsx(db=db, file_path=payload.clinical_xlsx_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "QIN-BREAST-02 clinical workbook import completed",
        "result": result,
    }


# ─── Synthetic data generation ────────────────────────────────────────────────

@router.post("/generate-qin-synthetic-cbc")
def generate_qin_synthetic_cbc(db: Session = Depends(get_db)):
    from backend.services.synthetic_cbc import generate_synthetic_cbc_for_qin_patients

    return {
        "message": "Synthetic QIN CBC generation completed",
        "result": generate_synthetic_cbc_for_qin_patients(db),
        "warning": "These CBC values are synthetic demo data and are not part of QIN-BREAST-02.",
    }


@router.post("/generate-synthetic-breast-journeys")
def generate_synthetic_breast_journeys(payload: SyntheticJourneyRequest, db: Session = Depends(get_db)):
    from backend.services.synthetic_journey import generate_synthetic_breast_cancer_journeys

    if payload.count < 1 or payload.count > 500:
        raise HTTPException(status_code=400, detail="count must be between 1 and 500")

    return {
        "message": "Synthetic breast cancer longitudinal journeys generated",
        "result": generate_synthetic_breast_cancer_journeys(db=db, count=payload.count, seed=payload.seed),
        "warning": "Synthetic journeys are for software testing and demos only, not model validation or clinical evidence.",
    }


@router.post("/generate-temporal-synthetic-breast-journeys")
def generate_temporal_synthetic_breast_journeys(payload: TemporalSyntheticJourneyRequest, db: Session = Depends(get_db)):
    from backend.services.synthetic_journey import generate_temporal_breast_cancer_journeys

    if payload.count < 1 or payload.count > 300:
        raise HTTPException(status_code=400, detail="count must be between 1 and 300")
    if payload.cycles < 2 or payload.cycles > 8:
        raise HTTPException(status_code=400, detail="cycles must be between 2 and 8")

    return {
        "message": "Temporal synthetic breast cancer journeys generated",
        "result": generate_temporal_breast_cancer_journeys(db=db, count=payload.count, seed=payload.seed, cycles=payload.cycles),
        "warning": "Synthetic temporal journeys are for engineering demos and model practice only, not clinical evidence.",
    }


@router.post("/generate-complete-synthetic-breast-dataset")
def generate_complete_synthetic_breast_dataset_endpoint(
    payload: CompleteSyntheticDatasetRequest,
    db: Session = Depends(get_db),
):
    from backend.services.complete_synthetic_dataset import generate_complete_synthetic_breast_dataset

    if payload.count < 1 or payload.count > 1000:
        raise HTTPException(status_code=400, detail="count must be between 1 and 1000")
    if payload.cycles < 2 or payload.cycles > 10:
        raise HTTPException(status_code=400, detail="cycles must be between 2 and 10")
    if payload.missing_rate < 0 or payload.missing_rate > 0.35:
        raise HTTPException(status_code=400, detail="missing_rate must be between 0 and 0.35")
    if payload.noise_level < 0 or payload.noise_level > 0.25:
        raise HTTPException(status_code=400, detail="noise_level must be between 0 and 0.25")
    if not payload.patient_prefix or len(payload.patient_prefix) > 32:
        raise HTTPException(status_code=400, detail="patient_prefix is required and must be 32 characters or less")

    return {
        "message": "Complete synthetic breast cancer treatment dataset generated",
        "result": generate_complete_synthetic_breast_dataset(
            db=db,
            count=payload.count,
            seed=payload.seed,
            cycles=payload.cycles,
            output_dir=payload.output_dir,
            write_db=payload.write_db,
            patient_prefix=payload.patient_prefix,
            balanced_outcomes=payload.balanced_outcomes,
            balanced_subgroups=payload.balanced_subgroups,
            missing_rate=payload.missing_rate,
            noise_level=payload.noise_level,
        ),
        "warning": "This dataset is fully synthetic and intended only for engineering demos and ML practice.",
    }


# ─── Model training ───────────────────────────────────────────────────────────

@router.post("/train-complete-synthetic-models")
def train_complete_synthetic_models_endpoint(
    payload: CompleteSyntheticTrainingRequest,
    db: Session = Depends(get_db),
):
    from backend.services.complete_synthetic_training import train_complete_synthetic_models
    from backend.services.mlops_tracking import finish_experiment_run, start_experiment_run

    allowed_targets = {
        "treatment_success_binary",
        "maintenance_needed",
        "toxicity_risk_binary",
        "support_intervention_needed",
        "urgent_intervention_needed",
    }
    if payload.target not in allowed_targets:
        raise HTTPException(status_code=400, detail=f"target must be one of {sorted(allowed_targets)}")
    if payload.test_size <= 0 or payload.test_size >= 0.5:
        raise HTTPException(status_code=400, detail="test_size must be greater than 0 and less than 0.5")
    if payload.cnn_epochs < 1 or payload.cnn_epochs > 200:
        raise HTTPException(status_code=400, detail="cnn_epochs must be between 1 and 200")

    run = start_experiment_run(
        db=db,
        experiment_name="complete_synthetic_training_api",
        run_name=payload.target,
        params=payload.dict(),
        tags={"entrypoint": "fastapi", "warning": "synthetic_data_only"},
    )

    try:
        result = train_complete_synthetic_models(
            ml_csv_path=payload.ml_csv_path,
            output_dir=payload.output_dir,
            target=payload.target,
            test_size=payload.test_size,
            seed=payload.seed,
            cnn_epochs=payload.cnn_epochs,
            cnn_batch_size=payload.cnn_batch_size,
        )
    except FileNotFoundError as exc:
        finish_experiment_run(db=db, run_id=run["run_id"], status="failed", error_message=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        finish_experiment_run(db=db, run_id=run["run_id"], status="failed", error_message=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        finish_experiment_run(db=db, run_id=run["run_id"], status="failed", error_message=str(exc))
        raise

    finish_experiment_run(
        db=db,
        run_id=run["run_id"],
        status="completed",
        metrics=result,
        artifacts=(result.get("artifacts") or {}),
        tags={"best_model": result.get("best_model_by_patient_level_roc_auc")},
    )
    return {
        "message": "Complete synthetic model training finished",
        "mlops_run_id": run["run_id"],
        "result": result,
    }


@router.post("/generate-complete-synthetic-xai")
def generate_complete_synthetic_xai_endpoint(payload: CompleteSyntheticXAIRequest):
    from backend.services.complete_synthetic_xai import generate_complete_synthetic_xai

    if payload.top_n < 1 or payload.top_n > 20:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 20")

    try:
        result = generate_complete_synthetic_xai(
            ml_csv_path=payload.ml_csv_path,
            model_path=payload.model_path,
            predictions_csv_path=payload.predictions_csv_path,
            output_json_path=payload.output_json_path,
            top_n=payload.top_n,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Complete synthetic model explanations generated",
        "result": result,
        "warning": "Synthetic XAI explains simulator-trained model behavior only, not clinical causality.",
    }


# ─── QIN MRI pipeline ─────────────────────────────────────────────────────────

@router.post("/index-qin-mri")
def index_qin_mri(payload: MRISeriesIndexRequest, db: Session = Depends(get_db)):
    from backend.services.mri_series_indexer import index_mri_series

    try:
        result = index_mri_series(db=db, root_path=payload.root_path, patient_id=payload.patient_id, max_files=payload.max_files)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"message": "MRI DICOM series index completed", "result": result}


@router.post("/build-qin-mri-manifest")
def build_qin_mri_manifest_endpoint(payload: MRIManifestRequest, db: Session = Depends(get_db)):
    from backend.services.mri_manifest import build_qin_mri_manifest

    return {
        "message": "QIN-BREAST-02 MRI modeling manifest built",
        "result": build_qin_mri_manifest(db=db, clinical_xlsx_path=payload.clinical_xlsx_path, output_csv_path=payload.output_csv_path),
    }


@router.post("/preprocess-qin-mri-previews")
def preprocess_qin_mri_previews(payload: MRIPreviewPreprocessRequest):
    from backend.services.mri_preprocessing import preprocess_mri_manifest_previews

    return {
        "message": "QIN-BREAST-02 MRI preview preprocessing completed",
        "result": preprocess_mri_manifest_previews(manifest_csv_path=payload.manifest_csv_path, output_dir=payload.output_dir),
    }


# ─── BreastDCEDL pipeline ─────────────────────────────────────────────────────

@router.post("/inspect-breastdcedl")
def inspect_breastdcedl(payload: BreastDCEDLInspectRequest):
    from backend.services.breastdcedl_inspector import inspect_breastdcedl_dataset

    try:
        result = inspect_breastdcedl_dataset(payload.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"message": "BreastDCEDL local dataset inspection completed", "result": result}


@router.post("/build-breastdcedl-manifest")
def build_breastdcedl_manifest_endpoint(payload: BreastDCEDLManifestRequest):
    from backend.services.breastdcedl_inspector import build_breastdcedl_manifest

    try:
        result = build_breastdcedl_manifest(root_path=payload.root_path, output_csv_path=payload.output_csv_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"message": "BreastDCEDL manifest built", "result": result}


@router.post("/run-breastdcedl-baseline")
def run_breastdcedl_baseline_endpoint(payload: BreastDCEDLBaselineRequest):
    from backend.services.breastdcedl_baseline import run_breastdcedl_baseline

    try:
        result = run_breastdcedl_baseline(
            manifest_csv_path=payload.manifest_csv_path,
            features_csv_path=payload.features_csv_path,
            metrics_json_path=payload.metrics_json_path,
            predictions_csv_path=payload.predictions_csv_path,
            max_patients=payload.max_patients,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"message": "BreastDCEDL baseline completed", "result": result}


@router.post("/generate-breastdcedl-previews")
def generate_breastdcedl_previews_endpoint(payload: BreastDCEDLPreviewRequest):
    from backend.services.breastdcedl_previews import generate_breastdcedl_previews

    return {
        "message": "BreastDCEDL preview generation completed",
        "result": generate_breastdcedl_previews(
            manifest_csv_path=payload.manifest_csv_path,
            output_dir=payload.output_dir,
            max_patients=payload.max_patients,
        ),
    }


@router.post("/import-breastdcedl-patients")
def import_breastdcedl_patients_endpoint(payload: BreastDCEDLImportRequest, db: Session = Depends(get_db)):
    from backend.services.breastdcedl_importer import import_breastdcedl_patients_to_dashboard

    return {
        "message": "BreastDCEDL patients imported to dashboard",
        "result": import_breastdcedl_patients_to_dashboard(db=db, manifest_csv_path=payload.manifest_csv_path, limit=payload.limit),
    }


@router.post("/run-breastdcedl-cnn")
def run_breastdcedl_cnn_endpoint(payload: BreastDCEDLCNNRequest):
    from backend.services.breastdcedl_cnn import run_breastdcedl_small_cnn

    try:
        result = run_breastdcedl_small_cnn(
            manifest_csv_path=payload.manifest_csv_path,
            metrics_json_path=payload.metrics_json_path,
            max_patients=payload.max_patients,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"message": "BreastDCEDL small CNN baseline completed", "result": result}


@router.post("/generate-breastdcedl-xai")
def generate_breastdcedl_xai_endpoint(payload: BreastDCEDLXAIRequest):
    from backend.services.breastdcedl_xai import generate_breastdcedl_shap_explanations

    return {
        "message": "BreastDCEDL SHAP explanations generated",
        "result": generate_breastdcedl_shap_explanations(
            features_csv_path=payload.features_csv_path,
            output_json_path=payload.output_json_path,
            top_n=payload.top_n,
        ),
    }


# ─── Model registry ───────────────────────────────────────────────────────────

@router.post("/models/breastdcedl/train-final")
def train_breastdcedl_final_model_endpoint(payload: BreastDCEDLModelTrainRequest, db: Session = Depends(get_db)):
    from backend.services.model_artifacts import train_and_register_breastdcedl_model

    try:
        result = train_and_register_breastdcedl_model(
            db=db,
            version=payload.version,
            features_csv_path=payload.features_csv_path,
            metrics_path=payload.metrics_path,
            artifact_dir=payload.artifact_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@router.get("/models")
def list_models_endpoint(
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.model_artifacts import list_registered_models

    return {"models": list_registered_models(db)}


@router.post("/models/complete-synthetic/register-champion")
def register_complete_synthetic_champion_endpoint(
    payload: CompleteSyntheticRegisterRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.model_artifacts import register_complete_synthetic_champion

    try:
        result = register_complete_synthetic_champion(
            db=db,
            version=payload.version,
            metrics_path=payload.metrics_path,
            training_data_path=payload.training_data_path,
            artifact_dir=payload.artifact_dir,
            promotion_status=payload.promotion_status,
            promotion_reason=payload.promotion_reason,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Complete synthetic champion registered.",
        "model": result,
        "warning": "Synthetic champion registration is for MLOps practice only, not clinical deployment.",
    }


@router.post("/models/{model_name}/{model_version}/promote")
def promote_model_version_endpoint(
    model_name: str,
    model_version: str,
    payload: ModelVersionActionRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.app_logging import log_app_event
    from backend.services.model_artifacts import promote_model_version

    try:
        model = promote_model_version(db, model_name=model_name, model_version=model_version, reason=payload.reason)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="model_lifecycle",
            actor_role=context.role,
            route="/models/{model_name}/{model_version}/promote",
            status="error",
            input_payload={"model_name": model_name, "model_version": model_version, "reason": payload.reason},
            error_message=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "message": "Model version promoted to champion.",
        "model": model,
        "safety_note": "Promotion is an MLOps registry action. It does not imply clinical validation.",
    }


@router.post("/models/{model_name}/{model_version}/rollback")
def rollback_model_version_endpoint(
    model_name: str,
    model_version: str,
    payload: ModelVersionActionRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.app_logging import log_app_event
    from backend.services.model_artifacts import rollback_model_version

    try:
        model = rollback_model_version(db, model_name=model_name, rollback_to_version=model_version, reason=payload.reason)
    except ValueError as exc:
        log_app_event(
            db=db,
            event_type="model_lifecycle",
            actor_role=context.role,
            route="/models/{model_name}/{model_version}/rollback",
            status="error",
            input_payload={"model_name": model_name, "model_version": model_version, "reason": payload.reason},
            error_message=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "message": "Model rollback completed.",
        "model": model,
        "safety_note": "Rollback changes serving registry status only. Clinical use still requires review.",
    }


@router.post("/models/breastdcedl/predict/{patient_id}")
def predict_breastdcedl_patient_endpoint(
    patient_id: str,
    payload: BreastDCEDLModelPredictRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.crud import get_patient
    from backend.services.app_logging import log_app_event
    from backend.services.model_artifacts import predict_breastdcedl_patient

    patient = get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        result = predict_breastdcedl_patient(
            db=db,
            patient_id=patient_id,
            model_name=payload.model_name,
            model_version=payload.model_version,
            features_csv_path=payload.features_csv_path,
            shap_json_path=payload.shap_json_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        log_app_event(
            db=db,
            event_type="prediction",
            patient_id=patient_id,
            route="/models/breastdcedl/predict/{patient_id}",
            status="error",
            input_payload=payload.dict(),
            error_message=str(exc),
        )
        status_code = 404 if isinstance(exc, FileNotFoundError) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return result


@router.get("/prediction-audits")
def list_prediction_audits_endpoint(
    patient_id: str | None = None,
    limit: int = 50,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.model_artifacts import get_prediction_audit_logs

    safe_limit = max(1, min(limit, 200))
    return {"prediction_audits": get_prediction_audit_logs(db, patient_id=patient_id, limit=safe_limit)}
