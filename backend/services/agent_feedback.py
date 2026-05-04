import json

from backend.models import AgentResponseFeedback, ChatMessage


def create_agent_response_feedback(
    db,
    patient_id,
    rating,
    thumbs_up=None,
    chat_message_id=None,
    feedback_text=None,
    feedback_context=None,
):
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        raise ValueError("rating must be an integer from 1 to 5.")
    if feedback_text is not None and len(feedback_text) > 1000:
        raise ValueError("feedback_text must be 1000 characters or less.")
    if chat_message_id is not None:
        message = (
            db.query(ChatMessage)
            .filter(ChatMessage.id == chat_message_id)
            .filter(ChatMessage.patient_id == patient_id)
            .filter(ChatMessage.role == "assistant")
            .first()
        )
        if message is None:
            raise ValueError("chat_message_id must reference one of this patient's assistant messages.")

    row = AgentResponseFeedback(
        patient_id=patient_id,
        chat_message_id=chat_message_id,
        rating=rating,
        thumbs_up=_bool_to_int(thumbs_up),
        feedback_text=feedback_text,
        feedback_json=json.dumps(feedback_context or {}, default=str),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return feedback_to_dict(row)


def build_agent_feedback_summary(db, limit=10):
    rows = (
        db.query(AgentResponseFeedback)
        .order_by(AgentResponseFeedback.created_at.desc(), AgentResponseFeedback.id.desc())
        .all()
    )
    if not rows:
        return {
            "status": "unavailable",
            "purpose": "Patient-rated feedback on support-agent answers.",
            "feedback_count": 0,
            "average_rating": None,
            "thumbs_up_rate": None,
            "rating_distribution": {str(value): 0 for value in range(1, 6)},
            "recent_feedback": [],
        }
    ratings = [int(row.rating) for row in rows]
    thumbs = [row.thumbs_up for row in rows if row.thumbs_up is not None]
    distribution = {str(value): 0 for value in range(1, 6)}
    for rating in ratings:
        distribution[str(rating)] += 1
    average_rating = sum(ratings) / len(ratings)
    thumbs_up_rate = sum(1 for value in thumbs if int(value) == 1) / len(thumbs) if thumbs else None
    return {
        "status": _feedback_status(average_rating),
        "purpose": "Patient-rated feedback on support-agent answer usefulness. This is UX/workflow evidence, not clinical correctness.",
        "feedback_count": len(rows),
        "average_rating": round(average_rating, 2),
        "thumbs_up_rate": round(thumbs_up_rate, 3) if thumbs_up_rate is not None else None,
        "rating_distribution": distribution,
        "recent_feedback": [feedback_to_dict(row) for row in rows[:limit]],
    }


def list_agent_feedback(db, patient_id=None, limit=50):
    query = db.query(AgentResponseFeedback)
    if patient_id:
        query = query.filter(AgentResponseFeedback.patient_id == patient_id)
    rows = (
        query.order_by(AgentResponseFeedback.created_at.desc(), AgentResponseFeedback.id.desc())
        .limit(max(1, min(limit, 200)))
        .all()
    )
    return [feedback_to_dict(row) for row in rows]


def feedback_to_dict(row):
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "chat_message_id": row.chat_message_id,
        "rating": row.rating,
        "thumbs_up": None if row.thumbs_up is None else bool(row.thumbs_up),
        "feedback_text": row.feedback_text,
        "feedback_context": _loads_or_empty(row.feedback_json),
        "created_at": str(row.created_at),
    }


def _feedback_status(average_rating):
    if average_rating >= 4.2:
        return "strong"
    if average_rating >= 3.5:
        return "acceptable"
    if average_rating >= 2.8:
        return "unideal"
    return "failed"


def _bool_to_int(value):
    if value is None:
        return None
    return 1 if bool(value) else 0


def _loads_or_empty(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}
