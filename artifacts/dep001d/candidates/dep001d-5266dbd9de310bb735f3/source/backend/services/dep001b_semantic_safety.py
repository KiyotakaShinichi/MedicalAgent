"""Fail-closed runtime for DEP-001B safety routing and utility calibration."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import yaml
from scipy.sparse import csr_matrix, hstack

from backend.services.safety_policy_action import (
    IntentFamily,
    PolicyAction,
    PolicyDecision,
    select_policy_action,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "Data/evals/safety/dep001b"
CONFIG_PATH = ROOT / "config/dep001b_semantic_safety.yaml"
MAX_CONTEXT_TURNS = 4
MAX_TURN_CHARACTERS = 4000
STATE_VERSION = "dep001b_independent_signal_state_v1"


@dataclass(frozen=True)
class DEP001BSafetyPrediction:
    unsafe_probability: float
    urgent_probability: float
    intent_family: str
    intent_family_confidence: float
    uncertainty: float
    policy_action: str
    policy_reason: str
    model_version: str
    unsafe_route_threshold: float
    urgent_route_threshold: float
    urgent_independent_threshold: float
    selected_turn_offset: int
    context_turn_count: int
    state_version: str = STATE_VERSION
    failure_reason: str | None = None

    @property
    def requires_safe_route(self) -> bool:
        return self.policy_action in {
            PolicyAction.SAFE_REDIRECT.value,
            PolicyAction.REFUSE_ACTIONABLE.value,
            PolicyAction.URGENT_ESCALATION.value,
            PolicyAction.FAIL_CLOSED.value,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requires_safe_route"] = self.requires_safe_route
        return payload


@dataclass(frozen=True)
class _Runtime:
    encoder: Any
    model: dict[str, Any]
    calibration: dict[str, Any]
    thresholds: dict[str, Any]


def classify_dep001b_safety(
    query: str,
    *,
    previous_user_messages: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
) -> DEP001BSafetyPrediction:
    """Classify bounded turns and deterministically select a policy action."""
    if os.getenv("NLCARE_DEP001B_SAFETY_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return _fail_closed("dep001b_safety_disabled")
    if not isinstance(query, str) or not query.strip():
        return _fail_closed("malformed_or_empty_query")
    if len(query) > MAX_TURN_CHARACTERS:
        return _fail_closed("query_exceeds_safety_limit")
    history = list(previous_user_messages or ())
    if any(not isinstance(turn, str) or not turn.strip() or len(turn) > MAX_TURN_CHARACTERS for turn in history):
        return _fail_closed("malformed_patient_context")
    history = history[-MAX_CONTEXT_TURNS:]
    directory = Path(
        artifact_dir
        or os.getenv("NLCARE_DEP001B_ARTIFACT_DIR", "")
        or DEFAULT_ARTIFACT_DIR
    ).resolve()
    try:
        runtime = _load_runtime(str(directory))
        turns = history + [query]
        signals = _predict_turns(runtime, turns)
        return _aggregate(runtime, signals, len(history))
    except Exception as exc:
        return _fail_closed(f"dep001b_runtime_error:{type(exc).__name__}")


def clear_dep001b_runtime_cache() -> None:
    _load_runtime.cache_clear()


@lru_cache(maxsize=4)
def _load_runtime(artifact_dir: str) -> _Runtime:
    directory = Path(artifact_dir)
    snapshot_config = directory / "dep001b_semantic_safety.yaml"
    config_path = snapshot_config if snapshot_config.is_file() else CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = directory / "semantic_safety_model_manifest.json"
    model_path = directory / "semantic_safety_model.joblib"
    calibration_path = directory / "semantic_safety_calibration.joblib"
    thresholds_path = directory / "semantic_safety_thresholds.json"
    manifest = _read_json(manifest_path)
    thresholds = _read_json(thresholds_path)
    expected_model = str(config["model_version"])
    expected_dataset = str(config["dataset_version"])
    for payload, name in ((manifest, "manifest"), (thresholds, "thresholds")):
        if payload.get("model_version") != expected_model:
            raise ValueError(f"stale_{name}_model_version")
        if payload.get("dataset_version") != expected_dataset:
            raise ValueError(f"stale_{name}_dataset_version")
    generated = datetime.fromisoformat(str(manifest["generated_at"]).replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds() / 86400
    if age_days > int(config.get("artifact_max_age_days", 180)):
        raise ValueError("stale_dep001b_artifact")
    records = manifest["artifacts"]
    _verify_hash(model_path, str(records["model"]["sha256"]))
    _verify_hash(calibration_path, str(records["calibration"]["sha256"]))
    _verify_hash(thresholds_path, str(records["thresholds"]["sha256"]))
    if thresholds.get("model_sha256") != _sha256(model_path):
        raise ValueError("threshold_model_hash_mismatch")
    if thresholds.get("calibration_sha256") != _sha256(calibration_path):
        raise ValueError("threshold_calibration_hash_mismatch")
    model = joblib.load(model_path)
    calibration = joblib.load(calibration_path)
    for bundle, name in ((model, "model"), (calibration, "calibration")):
        if bundle.get("model_version") != expected_model or bundle.get("dataset_version") != expected_dataset:
            raise ValueError(f"unpaired_{name}_version")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(model["base_encoder"]), local_files_only=True)
    if int(encoder.get_sentence_embedding_dimension()) != int(model["embedding_dimension"]):
        raise ValueError("embedding_dimension_mismatch")
    return _Runtime(encoder=encoder, model=model, calibration=calibration, thresholds=thresholds)


def _predict_turns(runtime: _Runtime, turns: list[str]) -> list[dict[str, Any]]:
    embeddings = runtime.encoder.encode(
        turns,
        batch_size=min(16, len(turns)),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    spec = runtime.model["feature_spec"]
    vectors = runtime.model["vectorizers"]
    features = hstack((
        csr_matrix(embeddings * float(spec["semantic_weight"])),
        vectors["word"].transform(turns) * float(spec["word_weight"]),
        vectors["character"].transform(turns) * float(spec["character_weight"]),
    ), format="csr")
    unsafe = _calibrated(runtime.model["unsafe_head"], runtime.calibration["unsafe_calibrator"], features)
    urgent = _calibrated(runtime.model["urgent_head"], runtime.calibration["urgent_calibrator"], features)
    family_prob = runtime.model["intent_family_head"].predict_proba(features)
    labels = np.asarray(runtime.model["intent_family_head"].classes_)
    output = []
    for index in range(len(turns)):
        distribution = np.asarray(family_prob[index], dtype=float)
        output.append({
            "unsafe": float(unsafe[index]),
            "urgent": float(urgent[index]),
            "family": str(labels[int(np.argmax(distribution))]),
            "family_confidence": float(np.max(distribution)),
            "family_uncertainty": _normalized_entropy(distribution),
        })
    return output


def _aggregate(runtime: _Runtime, turns: list[dict[str, Any]], history_count: int) -> DEP001BSafetyPrediction:
    thresholds = runtime.thresholds
    last_index = len(turns) - 1
    weighted = []
    for index, turn in enumerate(turns):
        age = last_index - index
        decay = 0.92 ** age
        weighted.append({
            **turn,
            "unsafe": turn["unsafe"] * decay,
            "urgent": turn["urgent"] * decay,
            "offset": -age,
        })
    selected = max(
        weighted,
        key=lambda item: max(
            item["unsafe"] / max(float(thresholds["unsafe_route_threshold"]), 1e-6),
            item["urgent"] / max(float(thresholds["urgent_route_threshold"]), 1e-6),
        ),
    )
    unsafe_probability = max(float(item["unsafe"]) for item in weighted)
    urgent_probability = max(float(item["urgent"]) for item in weighted)
    uncertainty = max(_binary_entropy(unsafe_probability), float(selected["family_uncertainty"]))
    decision: PolicyDecision = select_policy_action(
        unsafe_probability=unsafe_probability,
        urgent_probability=urgent_probability,
        intent_family=str(selected["family"]),
        intent_family_confidence=float(selected["family_confidence"]),
        uncertainty=uncertainty,
        thresholds=thresholds,
    )
    return DEP001BSafetyPrediction(
        unsafe_probability=decision.unsafe_probability,
        urgent_probability=decision.urgent_probability,
        intent_family=decision.intent_family,
        intent_family_confidence=decision.intent_family_confidence,
        uncertainty=decision.uncertainty,
        policy_action=decision.action.value,
        policy_reason=decision.reason,
        model_version=str(runtime.model["model_version"]),
        unsafe_route_threshold=float(thresholds["unsafe_route_threshold"]),
        urgent_route_threshold=float(thresholds["urgent_route_threshold"]),
        urgent_independent_threshold=float(thresholds["urgent_independent_threshold"]),
        selected_turn_offset=int(selected["offset"]),
        context_turn_count=history_count,
    )


def _fail_closed(reason: str) -> DEP001BSafetyPrediction:
    config: dict[str, Any] = {}
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return DEP001BSafetyPrediction(
        unsafe_probability=1.0,
        urgent_probability=0.0,
        intent_family=IntentFamily.UNKNOWN_HIGH_RISK.value,
        intent_family_confidence=0.0,
        uncertainty=1.0,
        policy_action=PolicyAction.FAIL_CLOSED.value,
        policy_reason="safety_signal_failure",
        model_version=str(config.get("model_version") or "unavailable"),
        unsafe_route_threshold=0.0,
        urgent_route_threshold=0.0,
        urgent_independent_threshold=0.0,
        selected_turn_offset=0,
        context_turn_count=0,
        failure_reason=reason,
    )


def _calibrated(head: Any, calibrator: Any, features: Any) -> np.ndarray:
    scores = np.asarray(head.decision_function(features), dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(scores)[:, 1]


def _binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return float(-(value * math.log2(value) + (1.0 - value) * math.log2(1.0 - value)))


def _normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.sum(values * np.log2(values)) / math.log2(len(values)))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("malformed_json_artifact")
    return payload


def _verify_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"artifact_hash_mismatch:{path.name}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["DEP001BSafetyPrediction", "classify_dep001b_safety", "clear_dep001b_runtime_cache"]
