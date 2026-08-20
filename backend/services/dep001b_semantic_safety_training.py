"""Train calibrated DEP-001B safety signals and select policy thresholds."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack

from backend.services.dep001b_safety_corpus import (
    INTERNAL_TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from backend.services.safety_policy_action import PolicyAction, select_policy_action


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/dep001b_semantic_safety.yaml"
OUTPUT_DIR = ROOT / "Data/evals/safety/dep001b"
MODEL_PATH = OUTPUT_DIR / "semantic_safety_model.joblib"
CALIBRATION_PATH = OUTPUT_DIR / "semantic_safety_calibration.joblib"
THRESHOLDS_PATH = OUTPUT_DIR / "semantic_safety_thresholds.json"
MODEL_MANIFEST_PATH = OUTPUT_DIR / "semantic_safety_model_manifest.json"
EVAL_PATH = OUTPUT_DIR / "latest_training_evaluation.json"
FAILURES_PATH = OUTPUT_DIR / "latest_validation_failures.json"


def train_and_evaluate(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    random.seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))
    train_rows = _read_jsonl(TRAIN_PATH)
    validation_rows = _read_jsonl(VALIDATION_PATH)
    test_rows = _read_jsonl(INTERNAL_TEST_PATH)
    _assert_training_isolation(train_rows, validation_rows, test_rows)

    encoder = SentenceTransformer(str(config["base_encoder"]), local_files_only=True)
    all_text = [str(row["text"]) for row in train_rows + validation_rows + test_rows]
    all_embeddings = encoder.encode(
        all_text,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    train_n = len(train_rows)
    validation_n = len(validation_rows)
    train_embeddings = all_embeddings[:train_n]
    validation_embeddings = all_embeddings[train_n:train_n + validation_n]
    test_embeddings = all_embeddings[train_n + validation_n:]

    indices = np.arange(train_n)
    fit_indices, calibration_indices = train_test_split(
        indices,
        test_size=float(config["calibration_fraction"]),
        random_state=int(config["random_seed"]),
        stratify=[row["intent_family"] for row in train_rows],
    )
    fit_rows = [train_rows[int(index)] for index in fit_indices]
    calibration_rows = [train_rows[int(index)] for index in calibration_indices]
    fit_embeddings = train_embeddings[fit_indices]
    calibration_embeddings = train_embeddings[calibration_indices]

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=24000,
        sublinear_tf=True, strip_accents="unicode",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=32000,
        sublinear_tf=True, strip_accents="unicode",
    )
    fit_text = [str(row["text"]) for row in fit_rows]
    word_vectorizer.fit(fit_text)
    char_vectorizer.fit(fit_text)

    feature_spec = {
        "semantic_weight": 1.0,
        "word_weight": 0.75,
        "character_weight": 0.75,
    }
    vectorizers = {"word": word_vectorizer, "character": char_vectorizer}
    fit_features = _hybrid_features(fit_embeddings, fit_text, vectorizers, feature_spec)
    calibration_features = _hybrid_features(
        calibration_embeddings,
        [str(row["text"]) for row in calibration_rows],
        vectorizers,
        feature_spec,
    )

    unsafe_head = _binary_head(fit_features, [int(row["unsafe_expected"]) for row in fit_rows])
    urgent_head = _binary_head(fit_features, [int(row["urgent_expected"]) for row in fit_rows])
    family_head = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=2.0,
        random_state=int(config["random_seed"]),
    ).fit(fit_features, [row["intent_family"] for row in fit_rows])
    unsafe_calibrator = _fit_calibrator(
        unsafe_head, calibration_features, [int(row["unsafe_expected"]) for row in calibration_rows]
    )
    urgent_calibrator = _fit_calibrator(
        urgent_head, calibration_features, [int(row["urgent_expected"]) for row in calibration_rows]
    )

    model = {
        "schema_version": "dep001b_semantic_model_v1",
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "base_encoder": config["base_encoder"],
        "embedding_dimension": int(encoder.get_sentence_embedding_dimension()),
        "unsafe_head": unsafe_head,
        "urgent_head": urgent_head,
        "intent_family_head": family_head,
        "intent_families": [str(label) for label in family_head.classes_],
        "vectorizers": vectorizers,
        "feature_spec": feature_spec,
    }
    calibration = {
        "schema_version": "dep001b_semantic_calibration_v1",
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "method": "platt_scaling_on_disjoint_training_calibration_partition",
        "unsafe_calibrator": unsafe_calibrator,
        "urgent_calibrator": urgent_calibrator,
        "calibration_case_n": len(calibration_rows),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(calibration, CALIBRATION_PATH)

    validation_signals = _signals(model, calibration, validation_embeddings, [str(row["text"]) for row in validation_rows])
    thresholds = _select_thresholds(validation_rows, validation_signals, config)
    thresholds.update({
        "schema_version": "dep001b_semantic_thresholds_v1",
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "model_sha256": _sha256(MODEL_PATH),
        "calibration_sha256": _sha256(CALIBRATION_PATH),
        "selection_policy": (
            "urgent head selected independently; policy surface must meet safety recall "
            "floors before maximizing safe educational acceptance"
        ),
        "burned_external_holdout_used": False,
    })
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "dep001b_semantic_model_manifest_v1",
        "generated_at": generated_at,
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "base_encoder": config["base_encoder"],
        "embedding_dimension": model["embedding_dimension"],
        "heads": ["unsafe_binary", "urgent_binary", "intent_family_multiclass"],
        "artifacts": {
            "model": _artifact_record(MODEL_PATH),
            "calibration": _artifact_record(CALIBRATION_PATH),
            "thresholds": _artifact_record(THRESHOLDS_PATH),
        },
        "training_inputs": {
            "train_sha256": _sha256(TRAIN_PATH),
            "validation_sha256": _sha256(VALIDATION_PATH),
            "internal_test_sha256": _sha256(INTERNAL_TEST_PATH),
            "internal_blind_loaded": False,
        },
        "clinical_validation": False,
        "healthcare_production_ready": False,
    }
    MODEL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_signals = _signals(model, calibration, train_embeddings, [str(row["text"]) for row in train_rows])
    test_signals = _signals(model, calibration, test_embeddings, [str(row["text"]) for row in test_rows])
    evaluation = {
        "schema_version": "dep001b_training_evaluation_v1",
        "generated_at": generated_at,
        "status": "trained_pending_runtime_integration",
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "architecture": (
            "local multilingual MiniLM embeddings with separately calibrated unsafe "
            "and urgent logistic heads, multiclass intent family, and deterministic policy action"
        ),
        "train": _evaluate_rows(train_rows, train_signals, thresholds),
        "validation": _evaluate_rows(validation_rows, validation_signals, thresholds),
        "internal_test": _evaluate_rows(test_rows, test_signals, thresholds),
        "calibration": {
            "unsafe_reliability_diagram": _reliability_diagram(
                [int(row["unsafe_expected"]) for row in validation_rows],
                validation_signals["unsafe"],
            ),
            "urgent_reliability_diagram": _reliability_diagram(
                [int(row["urgent_expected"]) for row in validation_rows],
                validation_signals["urgent"],
            ),
        },
        "thresholds": thresholds,
        "internal_blind_evaluated": False,
        "burned_external_holdout_used": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
    }
    EVAL_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    failures = _failure_artifact(validation_rows, validation_signals, thresholds)
    FAILURES_PATH.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return evaluation


def _binary_head(embeddings: np.ndarray, labels: list[int]) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.5,
        random_state=28117,
    ).fit(embeddings, labels)


def _fit_calibrator(head: Any, embeddings: np.ndarray, labels: list[int]) -> LogisticRegression:
    scores = np.asarray(head.decision_function(embeddings), dtype=float).reshape(-1, 1)
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=28117).fit(scores, labels)


def _signals(model: dict[str, Any], calibration: dict[str, Any], embeddings: np.ndarray, texts: list[str]) -> dict[str, Any]:
    features = _hybrid_features(embeddings, texts, model["vectorizers"], model["feature_spec"])
    unsafe = _calibrated(model["unsafe_head"], calibration["unsafe_calibrator"], features)
    urgent = _calibrated(model["urgent_head"], calibration["urgent_calibrator"], features)
    family_probabilities = model["intent_family_head"].predict_proba(features)
    family_classes = np.asarray(model["intent_family_head"].classes_)
    family_indices = np.argmax(family_probabilities, axis=1)
    family = family_classes[family_indices]
    family_confidence = np.max(family_probabilities, axis=1)
    family_uncertainty = np.asarray([_normalized_entropy(row) for row in family_probabilities])
    unsafe_uncertainty = np.asarray([_binary_entropy(value) for value in unsafe])
    uncertainty = np.maximum(unsafe_uncertainty, family_uncertainty)
    return {
        "unsafe": unsafe,
        "urgent": urgent,
        "family": family,
        "family_confidence": family_confidence,
        "family_uncertainty": family_uncertainty,
        "uncertainty": uncertainty,
    }


def _select_thresholds(rows: list[dict[str, Any]], signals: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    targets = config["internal_targets"]
    quantiles = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    urgent_candidates = np.unique(np.round(np.concatenate((
        np.linspace(0.02, 0.70, 12), np.quantile(signals["urgent"], quantiles),
    )), 6))
    unsafe_candidates = np.unique(np.round(np.concatenate((
        np.linspace(0.25, 0.90, 12), np.quantile(signals["unsafe"], quantiles),
    )), 6))
    urgent_labels = np.asarray([bool(row["urgent_expected"]) for row in rows])
    family_is_urgent = signals["family"] == "URGENT_PRESENTATION"
    best_urgent: tuple[tuple[float, ...], tuple[float, float]] | None = None
    for supported_threshold in urgent_candidates:
        for independent_threshold in (0.50, 0.65, 0.80, 0.90, 0.97):
            if independent_threshold < supported_threshold:
                continue
            prediction = (signals["urgent"] >= independent_threshold) | (
                family_is_urgent
                & (signals["family_confidence"] >= 0.30)
                & (signals["urgent"] >= supported_threshold)
            )
            recall = _binary_rate(prediction[urgent_labels])
            false_rate = _binary_rate(prediction[~urgent_labels])
            score = (
                1.0 if recall >= float(targets["urgent_escalation_recall"]) else 0.0,
                recall,
                -false_rate,
                float(supported_threshold),
                independent_threshold,
            )
            if best_urgent is None or score > best_urgent[0]:
                best_urgent = (score, (float(supported_threshold), independent_threshold))
    if best_urgent is None or best_urgent[0][0] == 0.0:
        raise RuntimeError("no_threshold_candidate_meets_urgent_floor")
    urgent_threshold, urgent_independent_threshold = best_urgent[1]

    best: tuple[tuple[float, ...], dict[str, float]] | None = None
    for unsafe_threshold in unsafe_candidates:
            for family_threshold in (0.30, 0.40, 0.50):
                for uncertainty_threshold in (0.75, 0.85, 0.95):
                    candidate = {
                        "unsafe_route_threshold": float(unsafe_threshold),
                        "urgent_route_threshold": float(urgent_threshold),
                        "urgent_independent_threshold": float(urgent_independent_threshold),
                        "intent_family_confidence_threshold": family_threshold,
                        "uncertainty_route_threshold": uncertainty_threshold,
                        "urgent_family_support_floor": float(urgent_threshold),
                    }
                    metrics = _policy_metrics(rows, signals, candidate)
                    feasible = (
                        metrics["unsafe_intent_recall"] >= float(targets["unsafe_intent_recall"])
                        and metrics["urgent_escalation_recall"] >= float(targets["urgent_escalation_recall"])
                    )
                    score = (
                        1.0 if feasible else 0.0,
                        metrics["safe_educational_acceptance_rate"],
                        metrics["unsafe_intent_recall"],
                        metrics["urgent_escalation_recall"],
                        best_urgent[0][2],
                        -metrics["over_refusal_rate"],
                    )
                    if best is None or score > best[0]:
                        best = (score, candidate)
    if best is None:
        raise RuntimeError("no_threshold_candidate_meets_urgent_floor")
    return best[1]


def _evaluate_rows(rows: list[dict[str, Any]], signals: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    unsafe_labels = np.asarray([int(row["unsafe_expected"]) for row in rows])
    urgent_labels = np.asarray([int(row["urgent_expected"]) for row in rows])
    unsafe_pred = signals["unsafe"] >= thresholds["unsafe_route_threshold"]
    urgent_pred = signals["urgent"] >= thresholds["urgent_independent_threshold"]
    unsafe_precision, unsafe_recall, unsafe_f1, _ = precision_recall_fscore_support(
        unsafe_labels, unsafe_pred.astype(int), average="binary", zero_division=0
    )
    urgent_precision, urgent_recall, urgent_f1, _ = precision_recall_fscore_support(
        urgent_labels, urgent_pred.astype(int), average="binary", zero_division=0
    )
    policy = _policy_metrics(rows, signals, thresholds)
    return {
        "n": len(rows),
        "unsafe_head": {
            "auroc": round(float(roc_auc_score(unsafe_labels, signals["unsafe"])), 6),
            "auprc": round(float(average_precision_score(unsafe_labels, signals["unsafe"])), 6),
            "recall": round(float(unsafe_recall), 6),
            "precision": round(float(unsafe_precision), 6),
            "f1": round(float(unsafe_f1), 6),
            "brier": round(float(brier_score_loss(unsafe_labels, signals["unsafe"])), 6),
            "ece": round(_ece(unsafe_labels, signals["unsafe"]), 6),
        },
        "urgent_head": {
            "auroc": round(float(roc_auc_score(urgent_labels, signals["urgent"])), 6),
            "auprc": round(float(average_precision_score(urgent_labels, signals["urgent"])), 6),
            "recall": round(float(urgent_recall), 6),
            "precision": round(float(urgent_precision), 6),
            "f1": round(float(urgent_f1), 6),
            "brier": round(float(brier_score_loss(urgent_labels, signals["urgent"])), 6),
            "ece": round(_ece(urgent_labels, signals["urgent"]), 6),
        },
        "intent_family_accuracy": round(float(np.mean(signals["family"] == np.asarray([row["intent_family"] for row in rows]))), 6),
        "policy": policy,
    }


def _policy_metrics(rows: list[dict[str, Any]], signals: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    actions = []
    for index, _row in enumerate(rows):
        actions.append(select_policy_action(
            unsafe_probability=float(signals["unsafe"][index]),
            urgent_probability=float(signals["urgent"][index]),
            intent_family=str(signals["family"][index]),
            intent_family_confidence=float(signals["family_confidence"][index]),
            uncertainty=float(signals["uncertainty"][index]),
            thresholds=thresholds,
        ).action.value)
    unsafe_indices = [i for i, row in enumerate(rows) if row["unsafe_expected"]]
    urgent_indices = [i for i, row in enumerate(rows) if row["urgent_expected"]]
    safe_indices = [i for i, row in enumerate(rows) if not row["unsafe_expected"]]
    routed = {PolicyAction.SAFE_REDIRECT.value, PolicyAction.REFUSE_ACTIONABLE.value, PolicyAction.URGENT_ESCALATION.value, PolicyAction.FAIL_CLOSED.value}
    allowed = {PolicyAction.ALLOW_EDUCATIONAL.value, PolicyAction.ALLOW_WITH_BOUNDARY.value}
    unsafe_recall = _binary_rate([actions[i] in routed for i in unsafe_indices])
    urgent_recall = _binary_rate([actions[i] == PolicyAction.URGENT_ESCALATION.value for i in urgent_indices])
    safe_acceptance = _binary_rate([actions[i] in allowed for i in safe_indices])
    language = {}
    for name in ("english", "taglish", "filipino_heavy"):
        indices = [i for i in unsafe_indices if rows[i]["language"] == name]
        language[name] = _binary_rate([actions[i] in routed for i in indices])
    multi = [i for i in unsafe_indices if rows[i]["multi_turn"]]
    rag = [i for i in unsafe_indices if rows[i]["rag_conditioned"]]
    return {
        "unsafe_intent_recall": round(unsafe_recall, 6),
        "urgent_escalation_recall": round(urgent_recall, 6),
        "safe_educational_acceptance_rate": round(safe_acceptance, 6),
        "over_refusal_rate": round(1.0 - safe_acceptance, 6),
        "english_unsafe_recall": round(language["english"], 6),
        "taglish_unsafe_recall": round(language["taglish"], 6),
        "filipino_heavy_unsafe_recall": round(language["filipino_heavy"], 6),
        "en_taglish_gap": round(abs(language["english"] - language["taglish"]), 6),
        "multi_turn_unsafe_recall": round(_binary_rate([actions[i] in routed for i in multi]), 6),
        "rag_conditioned_unsafe_recall": round(_binary_rate([actions[i] in routed for i in rag]), 6),
        "policy_action_accuracy": round(float(np.mean([actions[i] == rows[i]["expected_policy_action"] for i in range(len(rows))])), 6),
        "action_counts": dict(sorted(Counter(actions).items())),
    }


def _failure_artifact(rows: list[dict[str, Any]], signals: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    failures = []
    for index, row in enumerate(rows):
        decision = select_policy_action(
            unsafe_probability=float(signals["unsafe"][index]), urgent_probability=float(signals["urgent"][index]),
            intent_family=str(signals["family"][index]), intent_family_confidence=float(signals["family_confidence"][index]),
            uncertainty=float(signals["uncertainty"][index]), thresholds=thresholds,
        )
        if decision.action.value != row["expected_policy_action"]:
            failures.append({
                "case_id": row["case_id"],
                "expected_policy_action": row["expected_policy_action"],
                "predicted_policy_action": decision.action.value,
                "expected_intent_family": row["intent_family"],
                "predicted_intent_family": decision.intent_family,
                "language": row["language"],
                "style": row["style"],
                "unsafe_probability": decision.unsafe_probability,
                "urgent_probability": decision.urgent_probability,
                "uncertainty": decision.uncertainty,
            })
    return {
        "schema_version": "dep001b_validation_failures_v1",
        "failure_n": len(failures),
        "failures": failures,
        "burned_external_examples_included": False,
        "clinical_validation": False,
    }


def _assert_training_isolation(train: list[dict[str, Any]], validation: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    for split, rows in (("train", train), ("validation", validation), ("internal_test", test)):
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"split_label_mismatch:{split}")
    if any(row.get("used_for_tuning") for row in test):
        raise ValueError("internal_test_used_for_tuning")


def _calibrated(head: Any, calibrator: Any, embeddings: np.ndarray) -> np.ndarray:
    scores = np.asarray(head.decision_function(embeddings), dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(scores)[:, 1]


def _hybrid_features(embeddings: np.ndarray, texts: list[str], vectorizers: dict[str, Any], feature_spec: dict[str, float]):
    semantic = csr_matrix(embeddings * float(feature_spec["semantic_weight"]))
    word = vectorizers["word"].transform(texts) * float(feature_spec["word_weight"])
    character = vectorizers["character"].transform(texts) * float(feature_spec["character_weight"])
    return hstack((semantic, word, character), format="csr")


def _binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return float(-(value * math.log2(value) + (1.0 - value) * math.log2(1.0 - value)))


def _normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.sum(values * np.log2(values)) / math.log2(len(values)))


def _ece(labels: Iterable[int], probabilities: Iterable[float], bins: int = 10) -> float:
    y = np.asarray(list(labels), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    total = max(len(y), 1)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (p >= low) & (p < high if index < bins - 1 else p <= high)
        if np.any(mask):
            result += float(np.sum(mask)) / total * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(result)


def _reliability_diagram(labels: list[int], probabilities: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=float)
    result = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (probabilities >= low) & (probabilities < high if index < 9 else probabilities <= high)
        if np.any(mask):
            result.append({"bin_low": low, "bin_high": high, "n": int(np.sum(mask)), "mean_confidence": round(float(np.mean(probabilities[mask])), 6), "observed_rate": round(float(np.mean(y[mask])), 6)})
    return result


def _binary_rate(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(bool(value) for value in rows) / len(rows) if rows else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["train_and_evaluate"]
