from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


DEFAULT_TTL_SECONDS = 30 * 60


@dataclass
class PatientConversationState:
    patient_id: str
    updated_at: float = field(default_factory=time.time)
    messages: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))
    saved_actions: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=3))
    pending_actions: dict[str, dict[str, Any]] = field(default_factory=dict)


_STATE: dict[str, PatientConversationState] = {}


def get_patient_conversation_state(patient_id: str) -> PatientConversationState:
    _prune_expired()
    state = _STATE.get(patient_id)
    if state is None:
        state = PatientConversationState(patient_id=patient_id)
        _STATE[patient_id] = state
    state.updated_at = time.time()
    return state


def remember_turn(patient_id: str, role: str, message: str, *, actions: list[dict[str, Any]] | None = None) -> None:
    state = get_patient_conversation_state(patient_id)
    state.messages.append({
        "role": role,
        "message": str(message or "")[:2000],
        "timestamp": time.time(),
    })
    for action in actions or []:
        if str(action.get("type", "")).startswith("saved_"):
            state.saved_actions.append(action)
    state.updated_at = time.time()


def set_pending_action(patient_id: str, key: str, payload: dict[str, Any]) -> None:
    state = get_patient_conversation_state(patient_id)
    state.pending_actions[key] = {
        **payload,
        "created_at": time.time(),
    }
    state.updated_at = time.time()


def get_pending_action(patient_id: str, key: str) -> dict[str, Any] | None:
    state = get_patient_conversation_state(patient_id)
    pending = state.pending_actions.get(key)
    if not pending:
        return None
    if time.time() - float(pending.get("created_at") or 0) > DEFAULT_TTL_SECONDS:
        state.pending_actions.pop(key, None)
        return None
    return pending


def clear_pending_action(patient_id: str, key: str) -> None:
    state = get_patient_conversation_state(patient_id)
    state.pending_actions.pop(key, None)
    state.updated_at = time.time()


def state_snapshot(patient_id: str) -> dict[str, Any]:
    state = get_patient_conversation_state(patient_id)
    return {
        "message_count": len(state.messages),
        "recent_messages": list(state.messages),
        "recent_saved_actions": list(state.saved_actions),
        "pending_actions": {
            key: {k: v for k, v in value.items() if k != "created_at"}
            for key, value in state.pending_actions.items()
        },
    }


def _prune_expired() -> None:
    now = time.time()
    expired = [
        patient_id
        for patient_id, state in _STATE.items()
        if now - state.updated_at > DEFAULT_TTL_SECONDS
    ]
    for patient_id in expired:
        _STATE.pop(patient_id, None)
