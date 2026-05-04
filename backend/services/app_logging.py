import json
from datetime import datetime, timezone

from backend.models import AgentResponseCache, AppEventLog, PredictionAuditLog


def log_app_event(
    db,
    event_type,
    actor_role=None,
    patient_id=None,
    route=None,
    status="ok",
    input_payload=None,
    output_payload=None,
    error_message=None,
):
    row = AppEventLog(
        event_type=event_type,
        actor_role=actor_role,
        patient_id=patient_id,
        route=route,
        status=status,
        input_json=_json_dumps(input_payload),
        output_json=_json_dumps(output_payload),
        error_message=str(error_message) if error_message else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return app_event_to_dict(row)


def build_app_monitoring_summary(db, recent_limit=10):
    events = db.query(AppEventLog).all()
    prediction_count = db.query(PredictionAuditLog).count()
    total_events = len(events)
    failed_events = [event for event in events if event.status in {"error", "failed"}]
    failure_rate = len(failed_events) / total_events if total_events else 0.0

    event_type_counts = {}
    for event in events:
        event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

    confidence_distribution = _confidence_distribution(db)
    cache_rows = db.query(AgentResponseCache).all()
    cache_hit_count = sum(int(row.hit_count or 0) for row in cache_rows)
    recent_errors = (
        db.query(AppEventLog)
        .filter(AppEventLog.status.in_(["error", "failed"]))
        .order_by(AppEventLog.created_at.desc(), AppEventLog.id.desc())
        .limit(recent_limit)
        .all()
    )

    return {
        "status": _monitoring_status(failure_rate, prediction_count),
        "purpose": "Operational telemetry for app usage, prediction behavior, and failure monitoring.",
        "total_events": total_events,
        "prediction_count": prediction_count,
        "failed_event_count": len(failed_events),
        "failure_rate": round(failure_rate, 3),
        "event_type_counts": event_type_counts,
        "confidence_distribution": confidence_distribution,
        "agent_cache": {
            "cache_entry_count": len(cache_rows),
            "cache_hit_count": cache_hit_count,
            "purpose": "Tracks exact/semantic reuse for low-risk educational agent answers only.",
        },
        "recent_errors": [app_event_to_dict(row) for row in recent_errors],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def app_event_to_dict(row):
    return {
        "id": row.id,
        "event_type": row.event_type,
        "actor_role": row.actor_role,
        "patient_id": row.patient_id,
        "route": row.route,
        "status": row.status,
        "input": _json_loads(row.input_json),
        "output": _json_loads(row.output_json),
        "error_message": row.error_message,
        "created_at": str(row.created_at),
    }


def _confidence_distribution(db):
    rows = db.query(PredictionAuditLog).all()
    probabilities = []
    for row in rows:
        payload = _json_loads(row.prediction_json) or {}
        value = _extract_probability(payload)
        if value is not None:
            probabilities.append(float(value))

    bins = [
        {"range": "0.0-0.2", "count": 0},
        {"range": "0.2-0.4", "count": 0},
        {"range": "0.4-0.6", "count": 0},
        {"range": "0.6-0.8", "count": 0},
        {"range": "0.8-1.0", "count": 0},
    ]
    for probability in probabilities:
        index = min(4, max(0, int(probability * 5)))
        bins[index]["count"] += 1

    return {
        "sample_count": len(probabilities),
        "mean_probability": round(sum(probabilities) / len(probabilities), 3) if probabilities else None,
        "bins": bins,
        "purpose": "Shows whether live prediction outputs cluster near uncertain middle values or extreme confidence bands.",
    }


def _extract_probability(payload):
    for key in [
        "response_probability",
        "pcr_probability",
        "logistic_regression_probability",
        "gradient_boosting_probability",
        "random_forest_probability",
        "extra_trees_probability",
        "temporal_1d_cnn_probability",
        "temporal_gru_probability",
    ]:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _monitoring_status(failure_rate, prediction_count):
    if prediction_count == 0:
        return "unavailable"
    if failure_rate >= 0.15:
        return "failed"
    if failure_rate >= 0.05:
        return "unideal"
    return "passed"


def _json_dumps(value):
    if value is None:
        return None
    return json.dumps(value, default=str)


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
