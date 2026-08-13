"""Fail-closed multilingual semantic safety recognition for patient-agent turns.

The frozen encoder and calibrated linear heads classify each turn separately.
Conversation history is represented as bounded, decayed risk state; arbitrary
turn text is never concatenated into a hidden prompt. This is an engineering
safety control, not a clinical classifier.
"""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "Data/evals/safety/dep001a"
CONFIG_PATH = ROOT / "config/dep001a_semantic_safety.yaml"
MAX_CONTEXT_TURNS = 4
MAX_TURN_CHARACTERS = 4000
RISK_STATE_VERSION = "dep001a_structured_risk_state_v3"
URGENT_CATEGORIES = {"urgent_deterioration", "emergency_symptoms", "urgent_natural_language"}
UNSAFE_CATEGORIES = {
    "unsafe_actionable_medical_request", "treatment_modification", "medication_dose_request",
    "contraindication_interactions", "urgent_deterioration", "emergency_symptoms",
    "urgent_natural_language", "refusal_bypass_attempt", "role_play_unsafe_request",
    "hypothetical_unsafe_request", "authority_impersonation", "rag_conditioned_unsafe_request",
    "multi_turn_escalation", "indirect_euphemistic_request",
}
CATEGORY_UNSAFE_CONFIDENCE = 0.60
CATEGORY_URGENT_CONFIDENCE = 0.12
URGENT_PROBABILITY_OVERRIDE = 0.50


@dataclass(frozen=True)
class SemanticSafetyPrediction:
    risk_category: str
    unsafe_probability: float
    urgent_probability: float
    uncertainty: float
    model_version: str
    status: str
    unsafe_route_threshold: float
    urgent_route_threshold: float
    action_target: str
    selected_turn_offset: int
    context_turn_count: int
    state_version: str = RISK_STATE_VERSION
    failure_reason: str | None = None

    @property
    def requires_safe_route(self) -> bool:
        return self.status in {"unsafe", "urgent", "uncertain", "fail_closed"}

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


def classify_multilingual_safety(
    query: str,
    *,
    previous_user_messages: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
) -> SemanticSafetyPrediction:
    """Return calibrated risk state, failing closed on any control-plane fault."""
    if os.getenv("NLCARE_SEMANTIC_SAFETY_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return _fail_closed("semantic_safety_disabled")
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
        or os.getenv("NLCARE_DEP001A_ARTIFACT_DIR", "")
        or DEFAULT_ARTIFACT_DIR
    ).resolve()
    try:
        runtime = _load_runtime(str(directory))
        turns = history + [query]
        raw = _predict_turns(runtime, turns)
        return _aggregate_turn_state(runtime, raw, len(history))
    except Exception as exc:  # Safety availability failures must not continue to RAG/generation.
        return _fail_closed(f"semantic_runtime_error:{type(exc).__name__}")


def clear_semantic_safety_runtime_cache() -> None:
    _load_runtime.cache_clear()


@lru_cache(maxsize=4)
def _load_runtime(artifact_dir: str) -> _Runtime:
    directory = Path(artifact_dir)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = directory / "semantic_safety_model_manifest.json"
    model_path = directory / "semantic_safety_model.joblib"
    calibration_path = directory / "semantic_safety_calibration.joblib"
    thresholds_path = directory / "semantic_safety_thresholds.json"
    manifest = _read_json(manifest_path)
    thresholds = _read_json(thresholds_path)
    expected_model_version = str(config["model_version"])
    expected_dataset_version = str(config["dataset_version"])
    for payload, name in ((manifest, "manifest"), (thresholds, "thresholds")):
        if payload.get("model_version") != expected_model_version:
            raise ValueError(f"stale_{name}_model_version")
        if payload.get("dataset_version") != expected_dataset_version:
            raise ValueError(f"stale_{name}_dataset_version")
    generated = datetime.fromisoformat(str(manifest["generated_at"]).replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds() / 86400
    if age_days > int(config.get("artifact_max_age_days", 180)):
        raise ValueError("stale_semantic_safety_artifact")
    records = manifest.get("artifacts") or {}
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
        if bundle.get("model_version") != expected_model_version:
            raise ValueError(f"unpaired_{name}_model_version")
        if bundle.get("dataset_version") != expected_dataset_version:
            raise ValueError(f"unpaired_{name}_dataset_version")
    if model.get("base_encoder") != config.get("base_encoder"):
        raise ValueError("encoder_version_mismatch")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(model["base_encoder"]), local_files_only=True)
    dimension = int(encoder.get_sentence_embedding_dimension())
    if dimension != int(model["embedding_dimension"]):
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
    unsafe = _calibrated_probability(
        runtime.model["unsafe_head"], runtime.calibration["unsafe_calibrator"], embeddings
    )
    urgent = _calibrated_probability(
        runtime.model["urgent_head"], runtime.calibration["urgent_calibrator"], embeddings
    )
    category_prob = runtime.model["category_head"].predict_proba(embeddings)
    labels = runtime.model["category_head"].classes_
    output = []
    for index in range(len(turns)):
        distribution = np.asarray(category_prob[index], dtype=float)
        output.append({
            "unsafe": float(unsafe[index]),
            "urgent": float(urgent[index]),
            "category": str(labels[int(np.argmax(distribution))]),
            "category_confidence": float(np.max(distribution)),
            "category_uncertainty": _normalized_entropy(distribution),
        })
    return output


def _aggregate_turn_state(
    runtime: _Runtime,
    turns: list[dict[str, Any]],
    history_count: int,
) -> SemanticSafetyPrediction:
    unsafe_threshold = float(runtime.thresholds["unsafe_route_threshold"])
    urgent_threshold = float(runtime.thresholds["urgent_route_threshold"])
    weighted: list[dict[str, Any]] = []
    last_index = len(turns) - 1
    for index, turn in enumerate(turns):
        age = last_index - index
        decay = 0.92 ** age
        urgent_score = (
            turn["urgent"]
            if turn["category"] in URGENT_CATEGORIES
            and (
                float(turn["category_confidence"]) >= CATEGORY_URGENT_CONFIDENCE
                or float(turn["urgent"]) >= URGENT_PROBABILITY_OVERRIDE
            )
            else 0.0
        )
        weighted.append({
            **turn,
            "unsafe": turn["unsafe"] * decay,
            "urgent": urgent_score * decay,
            "offset": -age,
        })
    selected = max(weighted, key=lambda item: max(item["unsafe"] / unsafe_threshold, item["urgent"] / urgent_threshold))
    unsafe_probability = max(item["unsafe"] for item in weighted)
    urgent_probability = max(item["urgent"] for item in weighted)
    binary_uncertainty = _binary_entropy(unsafe_probability)
    high_confidence_safe = float(runtime.thresholds["high_confidence_safe_threshold"])
    uncertainty = (
        binary_uncertainty
        if unsafe_probability <= high_confidence_safe and urgent_probability < urgent_threshold
        else max(binary_uncertainty, float(selected["category_uncertainty"]))
    )
    category = str(selected["category"])
    if urgent_probability >= urgent_threshold:
        status = "urgent"
        category = category if category in URGENT_CATEGORIES else "urgent_deterioration"
    elif unsafe_probability >= unsafe_threshold or (
        category in UNSAFE_CATEGORIES
        and float(selected["category_confidence"]) >= CATEGORY_UNSAFE_CONFIDENCE
    ):
        status = "unsafe"
    elif uncertainty >= float(runtime.thresholds["uncertainty_route_threshold"]):
        status = "uncertain"
        category = "ambiguous"
    else:
        status = "safe"
    return SemanticSafetyPrediction(
        risk_category=category,
        unsafe_probability=round(unsafe_probability, 6),
        urgent_probability=round(urgent_probability, 6),
        uncertainty=round(uncertainty, 6),
        model_version=str(runtime.model["model_version"]),
        status=status,
        unsafe_route_threshold=unsafe_threshold,
        urgent_route_threshold=urgent_threshold,
        action_target=_action_target(category, status),
        selected_turn_offset=int(selected["offset"]),
        context_turn_count=history_count,
    )


def _fail_closed(reason: str) -> SemanticSafetyPrediction:
    config: dict[str, Any] = {}
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return SemanticSafetyPrediction(
        risk_category="classifier_unavailable",
        unsafe_probability=1.0,
        urgent_probability=0.0,
        uncertainty=1.0,
        model_version=str(config.get("model_version") or "unavailable"),
        status="fail_closed",
        unsafe_route_threshold=0.0,
        urgent_route_threshold=0.0,
        action_target="safe_clarification_or_human_review",
        selected_turn_offset=0,
        context_turn_count=0,
        failure_reason=reason,
    )


def _calibrated_probability(head: Any, calibrator: Any, embeddings: np.ndarray) -> np.ndarray:
    scores = head.decision_function(embeddings).reshape(-1, 1)
    return calibrator.predict_proba(scores)[:, 1]


def _binary_entropy(probability: float) -> float:
    probability = min(max(float(probability), 1e-9), 1 - 1e-9)
    return float(-(probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability)))


def _normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(probabilities * np.log2(probabilities)) / math.log2(len(probabilities)))


def _action_target(category: str, status: str) -> str:
    if status == "urgent":
        return "urgent_human_escalation"
    if category in {"contraindication_interactions"}:
        return "pharmacist_or_clinician_review"
    if category in {"refusal_bypass_attempt", "authority_impersonation", "rag_conditioned_unsafe_request"}:
        return "security_block"
    if status in {"unsafe", "uncertain", "fail_closed"}:
        return "safe_refusal_or_clarification"
    return "education_or_tracking"


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


__all__ = [
    "SemanticSafetyPrediction",
    "classify_multilingual_safety",
    "clear_semantic_safety_runtime_cache",
]
