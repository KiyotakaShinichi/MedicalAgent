import json
from datetime import datetime, timezone

from backend.models import AsyncTask


SUPPORTED_TASK_TYPES = {
    "build_rag_index",
    "agent_regression",
    "mle_readiness",
    "evaluation_report",
    "materialize_feature_store",
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

    raise ValueError(f"No dispatcher for task_type={task_type}")


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
