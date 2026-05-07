import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.models import MLExperimentRun


DEFAULT_TRACKING_DIR = "Data/mlruns_local"
TRACKING_SCHEMA_VERSION = "local_mlops_run_v1"


def start_experiment_run(
    db=None,
    experiment_name="default",
    run_name=None,
    params=None,
    tags=None,
    tracking_dir=DEFAULT_TRACKING_DIR,
):
    started_at = datetime.now(timezone.utc)
    run_id = _new_run_id(experiment_name, started_at)
    payload = {
        "schema_version": TRACKING_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "status": "running",
        "params": params or {},
        "metrics": {},
        "artifacts": {},
        "artifact_hashes": [],
        "tags": tags or {},
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "error_message": None,
    }
    _write_run_payload(payload, tracking_dir=tracking_dir)

    if db is not None:
        row = MLExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name,
            status="running",
            params_json=_json_dumps(params or {}),
            metrics_json=_json_dumps({}),
            artifacts_json=_json_dumps({}),
            tags_json=_json_dumps(tags or {}),
            started_at=started_at,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

    return payload


def finish_experiment_run(
    db=None,
    run_id=None,
    status="completed",
    metrics=None,
    artifacts=None,
    tags=None,
    error_message=None,
    tracking_dir=DEFAULT_TRACKING_DIR,
):
    if not run_id:
        raise ValueError("run_id is required")
    payload = load_experiment_run(run_id, tracking_dir=tracking_dir) or {
        "schema_version": TRACKING_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": "unknown",
        "params": {},
        "started_at": None,
    }
    ended_at = datetime.now(timezone.utc)
    merged_tags = {**(payload.get("tags") or {}), **(tags or {})}
    artifact_payload = artifacts or payload.get("artifacts") or {}
    payload.update({
        "status": status,
        "metrics": metrics or payload.get("metrics") or {},
        "artifacts": artifact_payload,
        "artifact_hashes": artifact_hashes(artifact_payload),
        "tags": merged_tags,
        "ended_at": ended_at.isoformat(),
        "error_message": error_message,
    })
    _write_run_payload(payload, tracking_dir=tracking_dir)

    if db is not None:
        row = db.query(MLExperimentRun).filter(MLExperimentRun.run_id == run_id).first()
        if row is None:
            row = MLExperimentRun(
                run_id=run_id,
                experiment_name=payload.get("experiment_name") or "unknown",
                run_name=payload.get("run_name"),
                started_at=_parse_datetime(payload.get("started_at")) or ended_at,
            )
            db.add(row)
        row.status = status
        row.metrics_json = _json_dumps(payload.get("metrics") or {})
        row.artifacts_json = _json_dumps({
            "artifacts": artifact_payload,
            "artifact_hashes": payload["artifact_hashes"],
            "tracking_file": str(_run_path(run_id, tracking_dir)),
        })
        row.tags_json = _json_dumps(merged_tags)
        row.error_message = error_message
        row.ended_at = ended_at
        db.commit()
        db.refresh(row)

    return payload


def log_completed_run(
    db=None,
    experiment_name="default",
    run_name=None,
    params=None,
    metrics=None,
    artifacts=None,
    tags=None,
    tracking_dir=DEFAULT_TRACKING_DIR,
):
    run = start_experiment_run(
        db=db,
        experiment_name=experiment_name,
        run_name=run_name,
        params=params,
        tags=tags,
        tracking_dir=tracking_dir,
    )
    return finish_experiment_run(
        db=db,
        run_id=run["run_id"],
        status="completed",
        metrics=metrics,
        artifacts=artifacts,
        tags=tags,
        tracking_dir=tracking_dir,
    )


def list_experiment_runs(db=None, tracking_dir=DEFAULT_TRACKING_DIR, limit=50):
    if db is not None:
        rows = (
            db.query(MLExperimentRun)
            .order_by(MLExperimentRun.started_at.desc(), MLExperimentRun.id.desc())
            .limit(limit)
            .all()
        )
        return [_row_to_dict(row) for row in rows]

    root = Path(tracking_dir)
    files = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in files[:limit]
    ]


def load_experiment_run(run_id, tracking_dir=DEFAULT_TRACKING_DIR):
    path = _run_path(run_id, tracking_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_hashes(artifacts):
    rows = []
    for name, value in (artifacts or {}).items():
        if isinstance(value, dict):
            for child_name, child_value in value.items():
                rows.extend(artifact_hashes({f"{name}.{child_name}": child_value}))
            continue
        path = Path(str(value))
        rows.append({
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() and path.is_file() else None,
        })
    return rows


def _row_to_dict(row):
    return {
        "id": row.id,
        "run_id": row.run_id,
        "experiment_name": row.experiment_name,
        "run_name": row.run_name,
        "status": row.status,
        "params": _json_loads(row.params_json) or {},
        "metrics": _json_loads(row.metrics_json) or {},
        "artifacts": _json_loads(row.artifacts_json) or {},
        "tags": _json_loads(row.tags_json) or {},
        "error_message": row.error_message,
        "started_at": str(row.started_at),
        "ended_at": str(row.ended_at) if row.ended_at else None,
    }


def _write_run_payload(payload, tracking_dir):
    path = _run_path(payload["run_id"], tracking_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _run_path(run_id, tracking_dir):
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(run_id))
    return Path(tracking_dir) / f"{safe}.json"


def _new_run_id(experiment_name, started_at):
    prefix = "".join(char if char.isalnum() else "-" for char in experiment_name.lower()).strip("-")[:32] or "run"
    return f"{prefix}-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_dumps(value):
    return json.dumps(value, default=str)


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
