"""Train and calibrate DEP-001D input-risk and output-actionability models."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import yaml
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from backend.services.dep001d_safety_corpus import CONFIG_PATH, OUTPUT_DIR
from backend.services.safety_policy_action import PolicyAction, select_policy_action


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "Data/evals/safety/dep001d"
RUNTIME_DIR = ARTIFACT_DIR / "runtime"
TRAINING_EVAL_PATH = ARTIFACT_DIR / "latest_training_evaluation.json"
MODEL_MANIFEST_PATH = RUNTIME_DIR / "semantic_safety_model_manifest.json"
OUTPUT_MANIFEST_PATH = RUNTIME_DIR / "output_actionability_manifest.json"


def train_dep001d_models(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_rows = {
        split: _jsonl(OUTPUT_DIR / f"{split}_input_safety.jsonl")
        for split in ("train", "calibration", "validation", "internal_test")
    }
    input_rows = {
        split: _runtime_turn_rows(rows)
        for split, rows in input_rows.items()
    }
    output_rows = {
        split: _jsonl(OUTPUT_DIR / f"{split}_output_actionability.jsonl")
        for split in ("train", "calibration", "validation", "internal_test")
    }
    _assert_disjoint(input_rows)
    _assert_disjoint(output_rows)

    encoder = SentenceTransformer(str(config["base_encoder"]), local_files_only=True)
    input_order = [row for split in input_rows.values() for row in split]
    output_order = [row for split in output_rows.values() for row in split]
    combined_text = [str(row["text"]) for row in input_order + output_order]
    combined_embedding = encoder.encode(
        combined_text, batch_size=96, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")
    input_embedding = combined_embedding[:len(input_order)]
    output_embedding = combined_embedding[len(input_order):]
    input_embeddings = _split_embeddings(input_embedding, input_rows)
    output_embeddings = _split_embeddings(output_embedding, output_rows)

    input_model, input_calibration = _fit_input_model(config, input_rows, input_embeddings)
    input_signals = {
        split: _input_signals(input_model, input_calibration, input_embeddings[split], [str(row["text"]) for row in rows])
        for split, rows in input_rows.items()
    }
    input_thresholds = _select_input_thresholds(
        input_rows["validation"], input_signals["validation"], config["development_targets"]
    )

    output_model, output_calibration = _fit_output_model(config, output_rows, output_embeddings)
    output_probabilities = {
        split: _output_probabilities(output_model, output_calibration, output_embeddings[split], [str(row["text"]) for row in rows])
        for split, rows in output_rows.items()
    }
    output_thresholds = _select_output_thresholds(
        output_rows["validation"], output_probabilities["validation"], config["development_targets"]
    )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    input_model_path = RUNTIME_DIR / "semantic_safety_model.joblib"
    input_calibration_path = RUNTIME_DIR / "semantic_safety_calibration.joblib"
    input_threshold_path = RUNTIME_DIR / "semantic_safety_thresholds.json"
    output_model_path = RUNTIME_DIR / "output_actionability_model.joblib"
    output_calibration_path = RUNTIME_DIR / "output_actionability_calibration.joblib"
    output_threshold_path = RUNTIME_DIR / "output_actionability_thresholds.json"
    joblib.dump(input_model, input_model_path)
    joblib.dump(input_calibration, input_calibration_path)
    joblib.dump(output_model, output_model_path)
    joblib.dump(output_calibration, output_calibration_path)

    input_thresholds.update({
        "schema_version": "dep001d_semantic_thresholds_v1",
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "model_sha256": _sha256(input_model_path),
        "calibration_sha256": _sha256(input_calibration_path),
        "selection_policy": "validation-only safety floors before utility maximization",
        "dep001c_consumed_bank_used": False,
    })
    input_threshold_path.write_text(json.dumps(input_thresholds, indent=2), encoding="utf-8")
    output_thresholds.update({
        "schema_version": "dep001d_output_actionability_thresholds_v1",
        "model_version": config["output_model_version"],
        "dataset_version": config["dataset_version"],
        "model_sha256": _sha256(output_model_path),
        "calibration_sha256": _sha256(output_calibration_path),
        "selection_policy": "multilingual containment floors, safe-output utility, and a predeclared conservative threshold cap",
        "dep001c_consumed_bank_used": False,
    })
    output_threshold_path.write_text(json.dumps(output_thresholds, indent=2), encoding="utf-8")
    shutil.copy2(config_path, RUNTIME_DIR / "dep001b_semantic_safety.yaml")
    shutil.copy2(config_path, RUNTIME_DIR / "dep001d_semantic_safety.yaml")

    generated_at = datetime.now(timezone.utc).isoformat()
    input_manifest = {
        "schema_version": "dep001d_semantic_model_manifest_v1",
        "generated_at": generated_at,
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "base_encoder": config["base_encoder"],
        "embedding_dimension": int(encoder.get_sentence_embedding_dimension()),
        "artifacts": {
            "model": _record(input_model_path),
            "calibration": _record(input_calibration_path),
            "thresholds": _record(input_threshold_path),
        },
        "training_inputs": _corpus_records("input"),
        "consumed_dep001c_bank_loaded": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
    }
    MODEL_MANIFEST_PATH.write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    output_manifest = {
        "schema_version": "dep001d_output_actionability_manifest_v1",
        "generated_at": generated_at,
        "model_version": config["output_model_version"],
        "dataset_version": config["dataset_version"],
        "base_encoder": config["base_encoder"],
        "embedding_dimension": int(encoder.get_sentence_embedding_dimension()),
        "artifacts": {
            "model": _record(output_model_path),
            "calibration": _record(output_calibration_path),
            "thresholds": _record(output_threshold_path),
        },
        "training_inputs": _corpus_records("output"),
        "consumed_dep001c_bank_loaded": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
    }
    OUTPUT_MANIFEST_PATH.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")

    evaluation = {
        "schema_version": "dep001d_training_evaluation_v1",
        "generated_at": generated_at,
        "status": "passed" if _development_model_targets_pass(
            _input_metrics(input_rows["validation"], input_signals["validation"], input_thresholds),
            _output_metrics(output_rows["validation"], output_probabilities["validation"], output_thresholds),
            config["development_targets"],
        ) else "needs_attention",
        "input_safety": {
            split: _input_metrics(rows, input_signals[split], input_thresholds)
            for split, rows in input_rows.items()
        },
        "output_actionability": {
            split: _output_metrics(rows, output_probabilities[split], output_thresholds)
            for split, rows in output_rows.items()
        },
        "thresholds": {
            "input": input_thresholds,
            "output": output_thresholds,
        },
        "development_targets": config["development_targets"],
        "consumed_dep001c_bank_evaluated": False,
        "blind_bank_evaluated": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
    }
    TRAINING_EVAL_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return evaluation


def recalibrate_dep001d_output_thresholds(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Re-select only the output threshold from the dedicated validation split."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = {
        split: _jsonl(OUTPUT_DIR / f"{split}_output_actionability.jsonl")
        for split in ("train", "calibration", "validation", "internal_test")
    }
    model_path = RUNTIME_DIR / "output_actionability_model.joblib"
    calibration_path = RUNTIME_DIR / "output_actionability_calibration.joblib"
    threshold_path = RUNTIME_DIR / "output_actionability_thresholds.json"
    model = joblib.load(model_path)
    calibration = joblib.load(calibration_path)
    encoder = SentenceTransformer(str(model["base_encoder"]), local_files_only=True)
    ordered = [row for split_rows in rows.values() for row in split_rows]
    embedding = encoder.encode(
        [str(row["text"]) for row in ordered], batch_size=96,
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    split_embedding = _split_embeddings(embedding, rows)
    probabilities = {
        split: _output_probabilities(
            model, calibration, split_embedding[split],
            [str(row["text"]) for row in split_rows],
        )
        for split, split_rows in rows.items()
    }
    thresholds = _select_output_thresholds(
        rows["validation"], probabilities["validation"], config["development_targets"],
    )
    thresholds.update({
        "schema_version": "dep001d_output_actionability_thresholds_v1",
        "model_version": config["output_model_version"],
        "dataset_version": config["dataset_version"],
        "model_sha256": _sha256(model_path),
        "calibration_sha256": _sha256(calibration_path),
        "selection_policy": "multilingual containment floors, safe-output utility, and a predeclared conservative threshold cap",
        "dep001c_consumed_bank_used": False,
    })
    threshold_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    manifest = json.loads(OUTPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["artifacts"]["thresholds"] = _record(threshold_path)
    manifest["threshold_recalibrated_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    evaluation = json.loads(TRAINING_EVAL_PATH.read_text(encoding="utf-8"))
    evaluation["output_actionability"] = {
        split: _output_metrics(split_rows, probabilities[split], thresholds)
        for split, split_rows in rows.items()
    }
    evaluation["thresholds"]["output"] = thresholds
    evaluation["output_threshold_recalibrated_at"] = datetime.now(timezone.utc).isoformat()
    validation_input = evaluation["input_safety"]["validation"]
    validation_output = evaluation["output_actionability"]["validation"]
    evaluation["status"] = "passed" if _development_model_targets_pass(
        validation_input, validation_output, config["development_targets"],
    ) else "needs_attention"
    TRAINING_EVAL_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return evaluation


def _fit_input_model(config: dict[str, Any], rows: dict[str, list[dict[str, Any]]], embeddings: dict[str, np.ndarray]):
    train_text = [str(row["text"]) for row in rows["train"]]
    vectorizers = _fit_vectorizers(train_text)
    spec = {"semantic_weight": 1.0, "word_weight": 0.85, "character_weight": 0.95}
    train_features = _features(embeddings["train"], train_text, vectorizers, spec)
    calibration_text = [str(row["text"]) for row in rows["calibration"]]
    calibration_features = _features(embeddings["calibration"], calibration_text, vectorizers, spec)
    unsafe_head = _binary_head(train_features, [int(row["unsafe_expected"]) for row in rows["train"]], 48121)
    urgent_head = _binary_head(train_features, [int(row["urgent_expected"]) for row in rows["train"]], 48122)
    family_head = LogisticRegression(
        max_iter=3000, class_weight="balanced", C=2.5, random_state=48123,
    ).fit(train_features, [str(row["intent_family"]) for row in rows["train"]])
    calibration = {
        "schema_version": "dep001d_semantic_calibration_v1",
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "method": "platt_scaling_on_dedicated_calibration_partition",
        "unsafe_calibrator": _calibrator(unsafe_head, calibration_features, [int(row["unsafe_expected"]) for row in rows["calibration"]]),
        "urgent_calibrator": _calibrator(urgent_head, calibration_features, [int(row["urgent_expected"]) for row in rows["calibration"]]),
        "calibration_case_n": len(rows["calibration"]),
    }
    model = {
        "schema_version": "dep001d_semantic_model_v1",
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "base_encoder": config["base_encoder"],
        "embedding_dimension": int(embeddings["train"].shape[1]),
        "unsafe_head": unsafe_head,
        "urgent_head": urgent_head,
        "intent_family_head": family_head,
        "intent_families": [str(value) for value in family_head.classes_],
        "vectorizers": vectorizers,
        "feature_spec": spec,
    }
    return model, calibration


def _fit_output_model(config: dict[str, Any], rows: dict[str, list[dict[str, Any]]], embeddings: dict[str, np.ndarray]):
    train_text = [str(row["text"]) for row in rows["train"]]
    vectorizers = _fit_vectorizers(train_text)
    spec = {"semantic_weight": 1.15, "word_weight": 0.75, "character_weight": 0.85}
    train_features = _features(embeddings["train"], train_text, vectorizers, spec)
    calibration_text = [str(row["text"]) for row in rows["calibration"]]
    calibration_features = _features(embeddings["calibration"], calibration_text, vectorizers, spec)
    head = _binary_head(train_features, [int(row["actionable_expected"]) for row in rows["train"]], 58121)
    calibrator = _calibrator(head, calibration_features, [int(row["actionable_expected"]) for row in rows["calibration"]])
    model = {
        "schema_version": "dep001d_output_actionability_model_v1",
        "model_version": config["output_model_version"],
        "dataset_version": config["dataset_version"],
        "base_encoder": config["base_encoder"],
        "embedding_dimension": int(embeddings["train"].shape[1]),
        "head": head,
        "vectorizers": vectorizers,
        "feature_spec": spec,
    }
    calibration = {
        "schema_version": "dep001d_output_actionability_calibration_v1",
        "model_version": config["output_model_version"],
        "dataset_version": config["dataset_version"],
        "method": "platt_scaling_on_dedicated_calibration_partition",
        "calibrator": calibrator,
        "calibration_case_n": len(rows["calibration"]),
    }
    return model, calibration


def _select_input_thresholds(rows: list[dict[str, Any]], signals: dict[str, np.ndarray], targets: dict[str, Any]) -> dict[str, float]:
    urgent_candidates = _threshold_candidates(signals["urgent"], 0.005, 0.50)
    unsafe_candidates = _threshold_candidates(signals["unsafe"], 0.01, 0.60)
    best: tuple[tuple[float, ...], dict[str, float]] | None = None
    for urgent_threshold in urgent_candidates:
        for unsafe_threshold in unsafe_candidates:
            for family_threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
                for uncertainty_threshold in (0.70, 0.80, 0.90):
                    candidate = {
                        "unsafe_route_threshold": float(unsafe_threshold),
                        "urgent_route_threshold": float(urgent_threshold),
                        "urgent_independent_threshold": float(urgent_threshold),
                        "intent_family_confidence_threshold": family_threshold,
                        "urgent_family_confidence_threshold": family_threshold,
                        "uncertainty_route_threshold": uncertainty_threshold,
                        "urgent_family_support_floor": float(urgent_threshold),
                    }
                    metrics = _policy_metrics(rows, signals, candidate)
                    feasible = _input_targets_pass(metrics, targets)
                    score = (
                        1.0 if feasible else 0.0,
                        metrics["safe_educational_acceptance_rate"],
                        metrics["urgent_escalation_recall"],
                        metrics["unsafe_intent_recall"],
                        -metrics["maximum_language_gap"],
                        -metrics["over_refusal_rate"],
                        float(unsafe_threshold),
                        float(urgent_threshold),
                    )
                    if best is None or score > best[0]:
                        best = (score, candidate)
    if best is None or best[0][0] != 1.0:
        raise RuntimeError("dep001d_no_input_threshold_meets_development_targets")
    return best[1]


def _select_output_thresholds(rows: list[dict[str, Any]], probabilities: np.ndarray, targets: dict[str, Any]) -> dict[str, float]:
    labels = np.asarray([bool(row["actionable_expected"]) for row in rows])
    best = None
    threshold_cap = float(targets.get("output_actionability_threshold_cap", 0.10))
    for threshold in _threshold_candidates(probabilities, 0.01, threshold_cap):
        for uncertainty_threshold in (0.95, 0.975, 0.99, 0.999):
            uncertain = np.asarray([_binary_entropy(value) >= uncertainty_threshold for value in probabilities])
            blocked = (probabilities >= threshold) | uncertain
            recall = _rate(np.sum(blocked & labels), np.sum(labels))
            safe_acceptance = _rate(np.sum(~blocked & ~labels), np.sum(~labels))
            language_recalls = []
            for language in ("english", "taglish", "filipino_heavy"):
                language_mask = np.asarray([row["language"] == language for row in rows]) & labels
                language_recalls.append(_rate(np.sum(blocked & language_mask), np.sum(language_mask)))
            feasible = (
                recall >= float(targets["output_actionability_containment_recall"])
                and safe_acceptance >= float(targets["output_safe_acceptance"])
                and min(language_recalls) >= float(targets["output_actionability_containment_recall"])
            )
            # The cap prevents an easy synthetic split from selecting an
            # unsafe high threshold. Within that safety envelope, preserve
            # bounded education before choosing the highest feasible threshold.
            score = (
                1.0 if feasible else 0.0,
                safe_acceptance,
                recall,
                min(language_recalls),
                float(threshold),
                uncertainty_threshold,
            )
            if best is None or score > best[0]:
                best = (score, {"actionability_threshold": float(threshold), "uncertainty_threshold": uncertainty_threshold})
    if best is None or best[0][0] != 1.0:
        raise RuntimeError("dep001d_no_output_threshold_meets_development_targets")
    return best[1]


def _runtime_turn_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror per-turn runtime inference while retaining split isolation."""
    expanded: list[dict[str, Any]] = []
    seen_context: set[str] = set()
    for row in rows:
        turns = [str(value) for value in (row.get("turns") or [row["text"]])]
        for index, turn in enumerate(turns[:-1]):
            normalized = " ".join(turn.lower().split())
            if normalized in seen_context:
                continue
            seen_context.add(normalized)
            context = dict(row)
            context.update({
                "case_id": f"{row['case_id']}::context-{index + 1}",
                "source_case_id": row["case_id"],
                "text": turn,
                "turns": [turn],
                "intent_family": "EDUCATIONAL_GENERAL",
                "unsafe_expected": False,
                "urgent_expected": False,
                "expected_policy_action": PolicyAction.ALLOW_EDUCATIONAL.value,
                "multi_turn": False,
                "rag_conditioned": False,
                "runtime_turn_role": "prior_context",
            })
            expanded.append(context)
        current = dict(row)
        current.update({
            "case_id": f"{row['case_id']}::current",
            "source_case_id": row["case_id"],
            "text": turns[-1],
            "turns": [turns[-1]],
            "runtime_turn_role": "current",
        })
        expanded.append(current)
    return expanded


def _input_metrics(rows: list[dict[str, Any]], signals: dict[str, np.ndarray], thresholds: dict[str, Any]) -> dict[str, Any]:
    unsafe_labels = np.asarray([int(row["unsafe_expected"]) for row in rows])
    urgent_labels = np.asarray([int(row["urgent_expected"]) for row in rows])
    head_unsafe = signals["unsafe"] >= float(thresholds["unsafe_route_threshold"])
    head_urgent = signals["urgent"] >= float(thresholds["urgent_independent_threshold"])
    return {
        "n": len(rows),
        "unsafe_head": _binary_metrics(unsafe_labels, signals["unsafe"], head_unsafe),
        "urgent_head": _binary_metrics(urgent_labels, signals["urgent"], head_urgent),
        "intent_family_accuracy": round(float(np.mean(signals["family"] == np.asarray([row["intent_family"] for row in rows]))), 6),
        "language_calibration": _language_calibration(rows, signals),
        "policy": _policy_metrics(rows, signals, thresholds),
    }


def _output_metrics(rows: list[dict[str, Any]], probabilities: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray([int(row["actionable_expected"]) for row in rows])
    entropy = np.asarray([_binary_entropy(value) for value in probabilities])
    blocked = (probabilities >= float(thresholds["actionability_threshold"])) | (entropy >= float(thresholds["uncertainty_threshold"]))
    base = _binary_metrics(labels, probabilities, blocked)
    language = {}
    for name in ("english", "taglish", "filipino_heavy"):
        index = np.asarray([row["language"] == name for row in rows])
        language[name] = _binary_metrics(labels[index], probabilities[index], blocked[index])
    base.update({
        "containment_recall": base["recall"],
        "safe_output_acceptance": _rate(np.sum(~blocked & (labels == 0)), np.sum(labels == 0)),
        "language": language,
    })
    return base


def _policy_metrics(rows: list[dict[str, Any]], signals: dict[str, np.ndarray], thresholds: dict[str, Any]) -> dict[str, Any]:
    actions = []
    for index in range(len(rows)):
        actions.append(select_policy_action(
            unsafe_probability=float(signals["unsafe"][index]),
            urgent_probability=float(signals["urgent"][index]),
            intent_family=str(signals["family"][index]),
            intent_family_confidence=float(signals["family_confidence"][index]),
            uncertainty=float(signals["uncertainty"][index]),
            thresholds=thresholds,
        ).action.value)
    routed = {PolicyAction.SAFE_REDIRECT.value, PolicyAction.REFUSE_ACTIONABLE.value, PolicyAction.URGENT_ESCALATION.value, PolicyAction.FAIL_CLOSED.value}
    allowed = {PolicyAction.ALLOW_EDUCATIONAL.value, PolicyAction.ALLOW_WITH_BOUNDARY.value}
    unsafe = [i for i, row in enumerate(rows) if row["unsafe_expected"]]
    urgent = [i for i, row in enumerate(rows) if row["urgent_expected"]]
    safe = [i for i, row in enumerate(rows) if not row["unsafe_expected"]]
    language = {}
    for name in ("english", "taglish", "filipino_heavy"):
        indices = [i for i in unsafe if rows[i]["language"] == name]
        language[name] = _rate(sum(actions[i] in routed for i in indices), len(indices))
    multi = [i for i in unsafe if rows[i]["multi_turn"]]
    rag = [i for i in unsafe if rows[i]["rag_conditioned"]]
    return {
        "unsafe_intent_recall": _rate(sum(actions[i] in routed for i in unsafe), len(unsafe)),
        "urgent_escalation_recall": _rate(sum(actions[i] == PolicyAction.URGENT_ESCALATION.value for i in urgent), len(urgent)),
        "safe_educational_acceptance_rate": _rate(sum(actions[i] in allowed for i in safe), len(safe)),
        "over_refusal_rate": _rate(sum(actions[i] not in allowed for i in safe), len(safe)),
        "english_unsafe_recall": language["english"],
        "taglish_unsafe_recall": language["taglish"],
        "filipino_heavy_unsafe_recall": language["filipino_heavy"],
        "maximum_language_gap": round(max(language.values()) - min(language.values()), 6),
        "multi_turn_unsafe_recall": _rate(sum(actions[i] in routed for i in multi), len(multi)),
        "rag_conditioned_unsafe_recall": _rate(sum(actions[i] in routed for i in rag), len(rag)),
        "policy_action_accuracy": _rate(sum(actions[i] == rows[i]["expected_policy_action"] for i in range(len(rows))), len(rows)),
        "action_counts": dict(sorted(Counter(actions).items())),
    }


def _input_targets_pass(metrics: dict[str, Any], targets: dict[str, Any]) -> bool:
    return bool(
        metrics["unsafe_intent_recall"] >= float(targets["unsafe_intent_recall"])
        and metrics["urgent_escalation_recall"] >= float(targets["urgent_escalation_recall"])
        and metrics["safe_educational_acceptance_rate"] >= float(targets["safe_educational_acceptance_rate"])
        and metrics["over_refusal_rate"] <= float(targets["over_refusal_rate"])
        and min(metrics[f"{name}_unsafe_recall"] for name in ("english", "taglish", "filipino_heavy")) >= float(targets["language_unsafe_recall"])
        and metrics["maximum_language_gap"] <= float(targets["maximum_language_gap"])
        and metrics["multi_turn_unsafe_recall"] >= float(targets["multi_turn_unsafe_recall"])
        and metrics["rag_conditioned_unsafe_recall"] >= float(targets["rag_conditioned_unsafe_recall"])
    )


def _development_model_targets_pass(input_metrics: dict[str, Any], output_metrics: dict[str, Any], targets: dict[str, Any]) -> bool:
    return bool(
        _input_targets_pass(input_metrics["policy"], targets)
        and output_metrics["containment_recall"] >= float(targets["output_actionability_containment_recall"])
        and output_metrics["safe_output_acceptance"] >= float(targets["output_safe_acceptance"])
    )


def _input_signals(model: dict[str, Any], calibration: dict[str, Any], embeddings: np.ndarray, texts: list[str]) -> dict[str, np.ndarray]:
    features = _features(embeddings, texts, model["vectorizers"], model["feature_spec"])
    unsafe = _calibrated(model["unsafe_head"], calibration["unsafe_calibrator"], features)
    urgent = _calibrated(model["urgent_head"], calibration["urgent_calibrator"], features)
    family_probability = model["intent_family_head"].predict_proba(features)
    classes = np.asarray(model["intent_family_head"].classes_)
    family = classes[np.argmax(family_probability, axis=1)]
    family_confidence = np.max(family_probability, axis=1)
    family_uncertainty = np.asarray([_normalized_entropy(value) for value in family_probability])
    return {
        "unsafe": unsafe,
        "urgent": urgent,
        "family": family,
        "family_confidence": family_confidence,
        "uncertainty": np.maximum(np.asarray([_binary_entropy(value) for value in unsafe]), family_uncertainty),
    }


def _output_probabilities(model: dict[str, Any], calibration: dict[str, Any], embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
    features = _features(embeddings, texts, model["vectorizers"], model["feature_spec"])
    return _calibrated(model["head"], calibration["calibrator"], features)


def _fit_vectorizers(texts: list[str]) -> dict[str, Any]:
    word = TfidfVectorizer(
        ngram_range=(1, 3), min_df=2, max_features=36000,
        sublinear_tf=True, strip_accents="unicode",
    ).fit(texts)
    character = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_features=48000,
        sublinear_tf=True, strip_accents="unicode",
    ).fit(texts)
    return {"word": word, "character": character}


def _features(embeddings: np.ndarray, texts: list[str], vectorizers: dict[str, Any], spec: dict[str, float]):
    return hstack((
        csr_matrix(embeddings * float(spec["semantic_weight"])),
        vectorizers["word"].transform(texts) * float(spec["word_weight"]),
        vectorizers["character"].transform(texts) * float(spec["character_weight"]),
    ), format="csr")


def _binary_head(features: Any, labels: list[int], seed: int):
    return LogisticRegression(max_iter=2500, class_weight="balanced", C=2.0, random_state=seed).fit(features, labels)


def _calibrator(head: Any, features: Any, labels: list[int]):
    score = np.asarray(head.decision_function(features), dtype=float).reshape(-1, 1)
    return LogisticRegression(max_iter=1500, class_weight="balanced", random_state=68121).fit(score, labels)


def _calibrated(head: Any, calibrator: Any, features: Any) -> np.ndarray:
    score = np.asarray(head.decision_function(features), dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(score)[:, 1]


def _threshold_candidates(probabilities: np.ndarray, low: float, high: float) -> np.ndarray:
    # A bounded grid keeps threshold selection reproducible and prevents the
    # validation sweep from becoming an accidental brute-force optimizer.
    quantiles = np.quantile(probabilities, np.linspace(0.02, 0.98, 9))
    anchors = np.linspace(low, high, 7)
    return np.unique(np.round(
        np.clip(np.concatenate((anchors, quantiles)), low, high),
        6,
    ))


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.astype(int), average="binary", zero_division=0)
    return {
        "auroc": round(float(roc_auc_score(labels, probabilities)), 6),
        "auprc": round(float(average_precision_score(labels, probabilities)), 6),
        "recall": round(float(recall), 6),
        "precision": round(float(precision), 6),
        "f1": round(float(f1), 6),
        "brier": round(float(brier_score_loss(labels, probabilities)), 6),
        "ece": round(_ece(labels, probabilities), 6),
    }


def _language_calibration(rows: list[dict[str, Any]], signals: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for name in ("english", "taglish", "filipino_heavy"):
        index = np.asarray([row["language"] == name for row in rows])
        result[name] = {
            "unsafe_brier": round(float(brier_score_loss([int(row["unsafe_expected"]) for row in np.asarray(rows, dtype=object)[index]], signals["unsafe"][index])), 6),
            "unsafe_ece": round(_ece([int(row["unsafe_expected"]) for row in np.asarray(rows, dtype=object)[index]], signals["unsafe"][index]), 6),
            "urgent_brier": round(float(brier_score_loss([int(row["urgent_expected"]) for row in np.asarray(rows, dtype=object)[index]], signals["urgent"][index])), 6),
            "urgent_ece": round(_ece([int(row["urgent_expected"]) for row in np.asarray(rows, dtype=object)[index]], signals["urgent"][index]), 6),
        }
    return result


def _ece(labels: Iterable[int], probabilities: Iterable[float], bins: int = 10) -> float:
    labels_array = np.asarray(list(labels), dtype=float)
    probability_array = np.asarray(list(probabilities), dtype=float)
    total = len(labels_array)
    value = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (probability_array >= lower) & (probability_array < upper if upper < 1.0 else probability_array <= upper)
        if np.any(selected):
            value += float(np.sum(selected) / total) * abs(float(np.mean(labels_array[selected])) - float(np.mean(probability_array[selected])))
    return float(value)


def _binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return float(-(value * math.log2(value) + (1.0 - value) * math.log2(1.0 - value)))


def _normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.sum(values * np.log2(values)) / math.log2(len(values)))


def _rate(numerator: int | np.integer, denominator: int | np.integer) -> float:
    return round(float(numerator) / float(denominator), 6) if int(denominator) else 0.0


def _split_embeddings(values: np.ndarray, rows: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    output = {}
    cursor = 0
    for split, split_rows in rows.items():
        output[split] = values[cursor:cursor + len(split_rows)]
        cursor += len(split_rows)
    return output


def _assert_disjoint(rows: dict[str, list[dict[str, Any]]]) -> None:
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for split, split_rows in rows.items():
        for row in split_rows:
            case_id = str(row["case_id"])
            normalized = " ".join(str(row["text"]).lower().split())
            if case_id in seen_ids or normalized in seen_text:
                raise ValueError(f"dep001d_partition_overlap:{split}:{case_id}")
            seen_ids.add(case_id)
            seen_text.add(normalized)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _corpus_records(kind: str) -> dict[str, Any]:
    return {
        split: _record(OUTPUT_DIR / f"{split}_{kind}_{'safety' if kind == 'input' else 'actionability'}.jsonl")
        for split in ("train", "calibration", "validation", "internal_test")
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "ARTIFACT_DIR", "RUNTIME_DIR", "TRAINING_EVAL_PATH",
    "recalibrate_dep001d_output_thresholds", "train_dep001d_models",
]
