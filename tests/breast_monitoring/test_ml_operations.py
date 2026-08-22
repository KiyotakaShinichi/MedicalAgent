"""Model lifecycle, local MLOps, task, feature-store, and provider contracts."""

import os
import pandas as pd
from backend.models import AsyncTask, MLExperimentRun, ModelRegistry
from backend.services.inference_service import describe_inference_service, get_inference_service
from backend.services.feature_store import load_feature_row, load_feature_store_manifest, materialize_feature_store
from backend.services.local_llm import configured_llm_providers, describe_llm_adjudication
from backend.config import get_groq_config, get_groq_model
from backend.services.mlops_tracking import log_completed_run
from backend.services.model_artifacts import promote_model_version, rollback_model_version
from backend.services.task_queue import enqueue_task, run_task

from tests.breast_monitoring.support import (
    _make_temp_dir,
    _temp_db_session,
    _temp_root,
)


class MLOperationsTestsMixin:
    def test_model_lifecycle_promote_and_rollback_changes_champion(self):
        db = _temp_db_session()
        try:
            db.add(ModelRegistry(
                model_name="demo_response_model",
                model_version="v1",
                task="demo",
                artifact_path="Data/models/demo_v1.joblib",
                model_metadata_json='{"promotion_status": "candidate"}',
                status="active",
            ))
            db.add(ModelRegistry(
                model_name="demo_response_model",
                model_version="v2",
                task="demo",
                artifact_path="Data/models/demo_v2.joblib",
                model_metadata_json='{"promotion_status": "candidate"}',
                status="active",
            ))
            db.commit()

            promoted = promote_model_version(db, "demo_response_model", "v2", reason="better calibration")
            rolled_back = rollback_model_version(db, "demo_response_model", "v1", reason="v2 drift alert")

            self.assertEqual(promoted["status"], "champion")
            self.assertEqual(rolled_back["status"], "champion")
            rows = {row.model_version: row.status for row in db.query(ModelRegistry).all()}
            self.assertEqual(rows["v1"], "champion")
            self.assertEqual(rows["v2"], "rolled_back")
        finally:
            db.close()
            db.bind.dispose()

    def test_local_mlops_tracking_records_run_and_artifact_hash(self):
        db = _temp_db_session()
        artifact_dir = _make_temp_dir(_temp_root()) / "mlops"
        artifact_dir.mkdir(parents=True)
        artifact_path = artifact_dir / "metrics.json"
        artifact_path.write_text('{"roc_auc": 0.91}', encoding="utf-8")
        try:
            run = log_completed_run(
                db=db,
                experiment_name="unit_test_experiment",
                run_name="candidate-v1",
                params={"seed": 42},
                metrics={"roc_auc": 0.91},
                artifacts={"metrics": str(artifact_path)},
                tags={"source": "unit_test"},
                tracking_dir=str(artifact_dir),
            )
            row = db.query(MLExperimentRun).first()

            self.assertEqual(row.status, "completed")
            self.assertEqual(row.experiment_name, "unit_test_experiment")
            self.assertEqual(run["artifact_hashes"][0]["exists"], True)
            self.assertIsNotNone(run["artifact_hashes"][0]["sha256"])
            self.assertEqual(db.query(MLExperimentRun).count(), 1)
        finally:
            db.close()
            db.bind.dispose()

    def test_inference_service_boundary_reports_backend_and_missing_model(self):
        db = _temp_db_session()
        try:
            description = describe_inference_service()
            self.assertEqual(description["active_backend"], "local_artifact_loader")
            with self.assertRaises(FileNotFoundError):
                get_inference_service().predict_breastdcedl_patient(
                    db=db,
                    patient_id="NO-PATIENT",
                    model_name="missing_model",
                    model_version="v1",
                    features_csv_path="missing.csv",
                    shap_json_path="missing.json",
                )
        finally:
            db.close()
            db.bind.dispose()

    def test_local_task_queue_runs_rag_index_job(self):
        db = _temp_db_session()
        index_path = _make_temp_dir(_temp_root()) / "queued_rag_index.joblib"
        try:
            queued = enqueue_task(
                db=db,
                task_type="build_rag_index",
                payload={"index_path": str(index_path)},
                created_by="unit_test",
            )
            completed = run_task(db, queued["id"])

            self.assertEqual(completed["status"], "completed")
            self.assertTrue(index_path.exists())
            self.assertEqual(db.query(AsyncTask).count(), 1)
            self.assertGreaterEqual(completed["result"]["document_count"], 1)
        finally:
            db.close()
            db.bind.dispose()

    def test_local_feature_store_materializes_manifest_and_rows(self):
        test_dir = _make_temp_dir(_temp_root()) / "feature_store"
        test_dir.mkdir(parents=True)
        source_csv = test_dir / "features.csv"
        pd.DataFrame([
            {"patient_id": "P1", "cycle": 1, "wbc": 4.2, "label": 1},
            {"patient_id": "P2", "cycle": 1, "wbc": 3.8, "label": 0},
        ]).to_csv(source_csv, index=False)

        manifest = materialize_feature_store(source_csv=str(source_csv), output_dir=str(test_dir))
        loaded = load_feature_store_manifest(output_dir=str(test_dir))
        row = load_feature_row("P1", output_dir=str(test_dir))

        self.assertEqual(manifest["row_count"], 2)
        self.assertEqual(loaded["status"], "current")
        self.assertEqual(row.iloc[0]["patient_id"], "P1")

    def test_llm_adjudication_prefers_groq_then_ollama(self):
        managed_keys = [
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "OLLAMA_MODEL",
            "LOCAL_LLM_MODEL",
            "LLM_ADJUDICATION_ENABLED",
        ]
        original = {key: os.environ.get(key) for key in managed_keys}
        try:
            os.environ["GROQ_API_KEY"] = "test-key"
            os.environ["GROQ_MODEL"] = "test-groq-model"
            os.environ["OLLAMA_MODEL"] = "test-ollama-model"
            os.environ.pop("LOCAL_LLM_MODEL", None)
            os.environ["LLM_ADJUDICATION_ENABLED"] = "true"

            providers = configured_llm_providers()
            status = describe_llm_adjudication()

            self.assertEqual([provider["provider"] for provider in providers], ["groq", "ollama"])
            self.assertEqual(status["primary_provider"], "groq")

            os.environ["LLM_ADJUDICATION_ENABLED"] = "false"
            self.assertEqual(configured_llm_providers(), [])
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_groq_answer_and_router_models_are_split(self):
        managed_keys = [
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "GROQ_ANSWER_MODEL",
            "GROQ_ROUTER_MODEL",
            "GROQ_ADJUDICATION_MODEL",
            "LLM_ADJUDICATION_ENABLED",
        ]
        original = {key: os.environ.get(key) for key in managed_keys}
        try:
            os.environ["GROQ_API_KEY"] = "test-key"
            os.environ.pop("GROQ_MODEL", None)
            os.environ["GROQ_ANSWER_MODEL"] = "openai/gpt-oss-120b"
            os.environ["GROQ_ROUTER_MODEL"] = "llama-3.3-70b-versatile"
            os.environ["LLM_ADJUDICATION_ENABLED"] = "true"

            self.assertEqual(get_groq_model(), "openai/gpt-oss-120b")
            self.assertEqual(get_groq_config()["model"], "llama-3.3-70b-versatile")
            self.assertEqual(configured_llm_providers()[0]["model"], "llama-3.3-70b-versatile")
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
