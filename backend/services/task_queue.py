import json
from datetime import datetime, timezone

from backend.models import AsyncTask


SUPPORTED_TASK_TYPES = {
    "build_rag_index",
    "agent_regression",
    "safety_red_team",
    "safety_red_team_live",
    "rag_eval",
    "rag_eval_live",
    "rag_ablation",
    "drift_report",
    "noise_eval",
    "temporal_eval",
    "mle_readiness",
    "evaluation_report",
    "materialize_feature_store",
    "public_data_manifest",
    "public_imaging_manifest",
    "ultrasound_baseline",
    "ultrasound_transfer_baseline",
    "ultrasound_segmentation_baseline",
    "ct_lesion_workflow",
    "sim_to_public_imaging",
    "chat_latency_report",
    "ai_ml_narrative_report",
    "demo_storyline",
    "realism_calibrated_dataset",
    "current_vs_realism_candidate",
    "multilingual_refusal_eval",
    "llm_judge_eval",
    "benchmark_registry",
}


def enqueue_task(db, task_type, payload=None, created_by=None):
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type={task_type}. Supported: {sorted(SUPPORTED_TASK_TYPES)}")
    row = AsyncTask(
        task_type=task_type,
        status="queued",
        payload_json=json.dumps(payload or {}, default=str),
        created_by=created_by,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return task_to_dict(row)


def run_task(db, task_id):
    row = db.query(AsyncTask).filter(AsyncTask.id == task_id).first()
    if row is None:
        raise ValueError(f"Task not found: {task_id}")
    if row.status == "running":
        raise ValueError(f"Task is already running: {task_id}")

    row.status = "running"
    row.started_at = datetime.now(timezone.utc)
    row.attempts = int(row.attempts or 0) + 1
    db.commit()
    db.refresh(row)

    try:
        result = _dispatch_task(db, row.task_type, _json_loads(row.payload_json) or {})
    except Exception as exc:
        row.status = "failed"
        row.error_message = str(exc)
        row.finished_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)
        return task_to_dict(row)

    row.status = "completed"
    row.result_json = json.dumps(result, default=str)
    row.error_message = None
    row.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return task_to_dict(row)


def run_next_queued_task(db):
    row = (
        db.query(AsyncTask)
        .filter(AsyncTask.status == "queued")
        .order_by(AsyncTask.queued_at.asc(), AsyncTask.id.asc())
        .first()
    )
    if row is None:
        return None
    return run_task(db, row.id)


def list_tasks(db, limit=50):
    rows = (
        db.query(AsyncTask)
        .order_by(AsyncTask.queued_at.desc(), AsyncTask.id.desc())
        .limit(limit)
        .all()
    )
    return [task_to_dict(row) for row in rows]


def task_to_dict(row):
    return {
        "id": row.id,
        "task_type": row.task_type,
        "status": row.status,
        "payload": _json_loads(row.payload_json) or {},
        "result": _json_loads(row.result_json),
        "error_message": row.error_message,
        "attempts": row.attempts,
        "created_by": row.created_by,
        "queued_at": str(row.queued_at),
        "started_at": str(row.started_at) if row.started_at else None,
        "finished_at": str(row.finished_at) if row.finished_at else None,
    }


def _dispatch_task(db, task_type, payload):
    if task_type == "build_rag_index":
        from backend.services.agent_rag import get_rag_corpus, knowledge_base_fingerprint
        from backend.services.rag_vector_index import DEFAULT_RAG_INDEX_PATH, build_rag_vector_index

        return build_rag_vector_index(
            corpus=get_rag_corpus(),
            index_path=payload.get("index_path") or DEFAULT_RAG_INDEX_PATH,
            knowledge_fingerprint=knowledge_base_fingerprint(),
        )

    if task_type == "agent_regression":
        from backend.services.agent_regression_eval import DEFAULT_AGENT_REGRESSION_PATH, run_agent_regression_suite

        return run_agent_regression_suite(output_path=payload.get("output_path") or DEFAULT_AGENT_REGRESSION_PATH)

    if task_type in {"safety_red_team", "safety_red_team_live"}:
        from backend.services.safety_red_team import DEFAULT_OUTPUT_PATH, DEFAULT_CSV_PATH, run_safety_red_team_suite

        return run_safety_red_team_suite(
            output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH,
            csv_path=payload.get("csv_path") or DEFAULT_CSV_PATH,
            live_agent=task_type.endswith("_live") or bool(payload.get("live_agent")),
        )

    if task_type in {"rag_eval", "rag_eval_live"}:
        from backend.services.rag_eval_suite import DEFAULT_CSV_PATH, DEFAULT_OUTPUT_PATH, run_rag_eval_suite

        return run_rag_eval_suite(
            output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH,
            csv_path=payload.get("csv_path") or DEFAULT_CSV_PATH,
            live_agent=task_type.endswith("_live") or bool(payload.get("live_agent")),
        )

    if task_type == "rag_ablation":
        from backend.services.rag_ablation import ABLATION_OUTPUT_PATH, run_rag_ablation

        return run_rag_ablation(output_path=payload.get("output_path") or ABLATION_OUTPUT_PATH)

    if task_type == "drift_report":
        from backend.services.drift_monitoring import DEFAULT_OUTPUT_PATH, build_drift_report

        return build_drift_report(output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH)

    if task_type == "noise_eval":
        from backend.services.noise_eval import DEFAULT_NOISE_EVAL_PATH, run_noise_eval

        return run_noise_eval(output_path=payload.get("output_path") or DEFAULT_NOISE_EVAL_PATH)

    if task_type == "temporal_eval":
        from backend.services.temporal_eval import DEFAULT_TEMPORAL_EVAL_PATH, run_temporal_eval

        return run_temporal_eval(output_path=payload.get("output_path") or DEFAULT_TEMPORAL_EVAL_PATH)

    if task_type == "mle_readiness":
        from backend.services.mle_readiness import DEFAULT_OUTPUT_PATH, build_mle_readiness_summary

        return build_mle_readiness_summary(db=db, output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH)

    if task_type == "evaluation_report":
        from backend.services.evaluation_reports import generate_versioned_evaluation_report

        return generate_versioned_evaluation_report(
            db=db,
            output_root=payload.get("output_root") or "Data/model_evaluation_reports",
            run_id=payload.get("run_id"),
        )

    if task_type == "materialize_feature_store":
        from backend.services.feature_store import DEFAULT_FEATURE_STORE_DIR, materialize_feature_store

        return materialize_feature_store(
            source_csv=payload.get("source_csv") or "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv",
            output_dir=payload.get("output_dir") or DEFAULT_FEATURE_STORE_DIR,
            entity_column=payload.get("entity_column") or "patient_id",
        )

    if task_type == "public_data_manifest":
        from backend.services.public_data_manifest import DEFAULT_OUTPUT_PATH, build_public_data_manifest

        return build_public_data_manifest(output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH)

    if task_type == "public_imaging_manifest":
        from backend.services.public_imaging_datasets import DEFAULT_OUTPUT_PATH, build_public_imaging_manifest

        return build_public_imaging_manifest(output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH)

    if task_type == "ultrasound_baseline":
        from backend.services.imaging_baseline_experiments import DEFAULT_ULTRASOUND_OUTPUT_PATH, run_ultrasound_baseline

        return run_ultrasound_baseline(output_path=payload.get("output_path") or DEFAULT_ULTRASOUND_OUTPUT_PATH)

    if task_type == "ultrasound_transfer_baseline":
        from backend.services.ultrasound_transfer_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_transfer_baseline

        return run_ultrasound_transfer_baseline(**{**payload, "output_path": payload.get("output_path") or DEFAULT_OUTPUT_PATH})

    if task_type == "ultrasound_segmentation_baseline":
        from backend.services.ultrasound_segmentation_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_segmentation_baseline

        return run_ultrasound_segmentation_baseline(**{**payload, "output_path": payload.get("output_path") or DEFAULT_OUTPUT_PATH})

    if task_type == "ct_lesion_workflow":
        from backend.services.imaging_baseline_experiments import DEFAULT_CT_WORKFLOW_PATH, build_ct_lesion_workflow_report

        return build_ct_lesion_workflow_report(output_path=payload.get("output_path") or DEFAULT_CT_WORKFLOW_PATH)

    if task_type == "sim_to_public_imaging":
        from backend.services.sim_to_public_imaging_report import DEFAULT_OUTPUT_PATH, build_sim_to_public_imaging_report

        return build_sim_to_public_imaging_report(output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH)

    if task_type == "chat_latency_report":
        from backend.services.chat_latency_report import DEFAULT_OUTPUT_PATH, build_chat_latency_report

        return build_chat_latency_report(
            db=db,
            output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH,
            limit=int(payload.get("limit") or 500),
        )

    if task_type == "ai_ml_narrative_report":
        from backend.services.evaluation_narrative_report import DEFAULT_OUTPUT_DIR, build_ai_ml_narrative_report

        return build_ai_ml_narrative_report(output_dir=payload.get("output_dir") or DEFAULT_OUTPUT_DIR)

    if task_type == "demo_storyline":
        from backend.services.demo_storyline import DEFAULT_OUTPUT_DIR, build_demo_storyline

        return build_demo_storyline(
            patient_id=payload.get("patient_id") or "P001",
            output_dir=payload.get("output_dir") or DEFAULT_OUTPUT_DIR,
        )

    if task_type == "realism_calibrated_dataset":
        from backend.services.complete_synthetic_dataset import generate_complete_synthetic_breast_dataset
        from backend.services.synthetic_realism_report import build_synthetic_realism_report

        output_dir = payload.get("output_dir") or "Data/complete_synthetic_breast_journeys_realism_v2"
        report_path = payload.get("report_path") or "Data/mle_monitoring/synthetic_realism_candidate_report.json"
        summary = generate_complete_synthetic_breast_dataset(
            db=None,
            count=int(payload.get("count") or 240),
            seed=int(payload.get("seed") or 2031),
            cycles=int(payload.get("cycles") or 6),
            output_dir=output_dir,
            write_db=False,
            patient_prefix=payload.get("patient_prefix") or "REALISM-BRCA-",
            missing_rate=float(payload.get("missing_rate") or 0.08),
            noise_level=float(payload.get("noise_level") or 0.05),
            realism_profile="external_calibrated",
            toxicity_profile="realistic",
            missingness_mode="ehr_like",
        )
        report = build_synthetic_realism_report(
            training_csv=f"{output_dir}/temporal_ml_rows.csv",
            output_path=report_path,
        )
        return {"summary": summary, "realism_report": report}

    if task_type == "current_vs_realism_candidate":
        from backend.services.candidate_model_comparison import (
            DEFAULT_OUTPUT_PATH,
            build_current_vs_candidate_report,
        )

        return build_current_vs_candidate_report(output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH)

    if task_type == "multilingual_refusal_eval":
        from backend.services.multilingual_refusal_eval import DEFAULT_OUTPUT_PATH, run_multilingual_refusal_eval

        return run_multilingual_refusal_eval(output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH)

    if task_type == "llm_judge_eval":
        from backend.services.llm_judge_eval import DEFAULT_OUTPUT_PATH, run_llm_judge_eval

        return run_llm_judge_eval(
            rag_eval_path=payload.get("rag_eval_path") or "Data/evals/rag/latest_rag_gold_eval.json",
            output_path=payload.get("output_path") or DEFAULT_OUTPUT_PATH,
            max_cases=int(payload.get("max_cases") or 30),
        )

    if task_type == "benchmark_registry":
        from backend.services.benchmark_registry import DEFAULT_JSON_PATH, build_benchmark_registry

        return build_benchmark_registry(output_path=payload.get("output_path") or DEFAULT_JSON_PATH)

    raise ValueError(f"No dispatcher for task_type={task_type}")


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
