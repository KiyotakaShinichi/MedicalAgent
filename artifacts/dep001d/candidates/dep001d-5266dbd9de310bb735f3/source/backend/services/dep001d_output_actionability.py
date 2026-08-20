"""Fail-closed semantic validation for patient-facing response actionability."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from scipy.sparse import csr_matrix, hstack


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "Data/evals/safety/dep001d/runtime"
CONFIG_PATH = ROOT / "config/dep001d_semantic_safety.yaml"


@dataclass(frozen=True)
class OutputActionabilityDecision:
    decision: str
    actionable_probability: float
    uncertainty: float
    threshold: float
    uncertainty_threshold: float
    model_version: str
    reason: str
    failure_reason: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision == "blocked"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocked"] = self.blocked
        return payload


@dataclass(frozen=True)
class _Runtime:
    encoder: Any
    model: dict[str, Any]
    calibration: dict[str, Any]
    thresholds: dict[str, Any]


def classify_output_actionability(
    reply: str,
    *,
    artifact_dir: Path | None = None,
) -> OutputActionabilityDecision:
    """Block actionable, uncertain, malformed, or unvalidated output."""
    if not isinstance(reply, str) or not reply.strip():
        return _fail_closed("malformed_or_empty_output")
    if len(reply) > 16000:
        return _fail_closed("output_exceeds_safety_limit")
    if os.getenv("NLCARE_DEP001D_OUTPUT_GUARD_ENABLED", "true").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return _fail_closed("output_actionability_guard_disabled")
    directory = Path(
        artifact_dir
        or os.getenv("NLCARE_DEP001D_ARTIFACT_DIR", "")
        or DEFAULT_ARTIFACT_DIR
    ).resolve()
    try:
        runtime = _load_runtime(str(directory))
        embedding = runtime.encoder.encode(
            [reply], batch_size=1, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")
        spec = runtime.model["feature_spec"]
        vectors = runtime.model["vectorizers"]
        features = hstack((
            csr_matrix(embedding * float(spec["semantic_weight"])),
            vectors["word"].transform([reply]) * float(spec["word_weight"]),
            vectors["character"].transform([reply]) * float(spec["character_weight"]),
        ), format="csr")
        score = np.asarray(
            runtime.model["head"].decision_function(features), dtype=float,
        ).reshape(-1, 1)
        probability = float(runtime.calibration["calibrator"].predict_proba(score)[0, 1])
        uncertainty = _binary_entropy(probability)
        threshold = float(runtime.thresholds["actionability_threshold"])
        uncertainty_threshold = float(runtime.thresholds["uncertainty_threshold"])
        actionable = probability >= threshold
        uncertain = uncertainty >= uncertainty_threshold
        return OutputActionabilityDecision(
            decision="blocked" if actionable or uncertain else "allowed",
            actionable_probability=round(probability, 6),
            uncertainty=round(uncertainty, 6),
            threshold=threshold,
            uncertainty_threshold=uncertainty_threshold,
            model_version=str(runtime.model["model_version"]),
            reason=(
                "material_personalized_actionability"
                if actionable else
                "uncertain_output_actionability"
                if uncertain else
                "non_actionable_output"
            ),
        )
    except Exception as exc:
        return _fail_closed(f"output_actionability_runtime_error:{type(exc).__name__}")


def clear_output_actionability_cache() -> None:
    _load_runtime.cache_clear()


@lru_cache(maxsize=4)
def _load_runtime(artifact_dir: str) -> _Runtime:
    directory = Path(artifact_dir)
    config = yaml.safe_load((directory / "dep001d_semantic_safety.yaml").read_text(encoding="utf-8"))
    manifest_path = directory / "output_actionability_manifest.json"
    model_path = directory / "output_actionability_model.joblib"
    calibration_path = directory / "output_actionability_calibration.joblib"
    thresholds_path = directory / "output_actionability_thresholds.json"
    manifest = _read_json(manifest_path)
    thresholds = _read_json(thresholds_path)
    expected_model = str(config["output_model_version"])
    expected_dataset = str(config["dataset_version"])
    for payload, name in ((manifest, "manifest"), (thresholds, "thresholds")):
        if payload.get("model_version") != expected_model:
            raise ValueError(f"stale_{name}_model_version")
        if payload.get("dataset_version") != expected_dataset:
            raise ValueError(f"stale_{name}_dataset_version")
    generated = datetime.fromisoformat(str(manifest["generated_at"]).replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds() / 86400
    if age_days > int(config.get("artifact_max_age_days", 180)):
        raise ValueError("stale_output_actionability_artifact")
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


def _fail_closed(reason: str) -> OutputActionabilityDecision:
    model_version = "unavailable"
    try:
        model_version = str(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["output_model_version"])
    except Exception:
        pass
    return OutputActionabilityDecision(
        decision="blocked",
        actionable_probability=1.0,
        uncertainty=1.0,
        threshold=0.0,
        uncertainty_threshold=0.0,
        model_version=model_version,
        reason="output_actionability_validation_unavailable",
        failure_reason=reason,
    )


def _binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return float(-(value * math.log2(value) + (1.0 - value) * math.log2(1.0 - value)))


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
    "OutputActionabilityDecision",
    "classify_output_actionability",
    "clear_output_actionability_cache",
]
