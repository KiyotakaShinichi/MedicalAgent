"""Aggregate-only overlap audit for DEP-001D development data."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.services.dep001d_safety_corpus import OUTPUT_DIR


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "Data/evals/safety/dep001d/latest_development_overlap_audit.json"
REFERENCE_PATHS = {
    "dep001b_train": ROOT / "Data/evals/safety/dep001b/train_safety_bank.jsonl",
    "dep001b_validation": ROOT / "Data/evals/safety/dep001b/validation_safety_bank.jsonl",
    "dep001b_internal_test": ROOT / "Data/evals/safety/dep001b/internal_test_safety_bank.jsonl",
    "burned_dep001b_blind": ROOT / "Data/evals/safety/dep001b/internal_blind_safety_bank.jsonl",
    "burned_dep001c_blind": ROOT / "artifacts/dep001c/blind_banks/dep001cblind-443183a2442b635b2025/bank/internal_blind_safety_bank.jsonl",
}


def run_dep001d_overlap_audit() -> dict[str, Any]:
    partitions = {
        name: _jsonl(OUTPUT_DIR / f"{name}_input_safety.jsonl")
        for name in ("train", "calibration", "validation", "internal_test")
    }
    all_new = [row for rows in partitions.values() for row in rows]
    new_text = [_row_text(row) for row in all_new]
    references = {name: [_row_text(row) for row in _jsonl(path)] for name, path in REFERENCE_PATHS.items()}
    encoder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        local_files_only=True,
    )
    new_embedding = encoder.encode(
        new_text, batch_size=96, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")
    reference_names = list(references)
    reference_lengths = [len(references[name]) for name in reference_names]
    combined_reference_text = [text for name in reference_names for text in references[name]]
    combined_reference_embedding = encoder.encode(
        combined_reference_text, batch_size=96, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")
    comparisons: dict[str, Any] = {}
    cursor = 0
    for name, length in zip(reference_names, reference_lengths):
        reference_text = references[name]
        reference_embedding = combined_reference_embedding[cursor:cursor + length]
        cursor += length
        comparisons[name] = _comparison(new_text, reference_text, new_embedding, reference_embedding)

    split_comparisons = {}
    split_embeddings: dict[str, np.ndarray] = {}
    cursor = 0
    for name, rows in partitions.items():
        split_embeddings[name] = new_embedding[cursor:cursor + len(rows)]
        cursor += len(rows)
    for left, right in (("train", "calibration"), ("train", "validation"), ("train", "internal_test"), ("validation", "internal_test")):
        split_comparisons[f"{left}_vs_{right}"] = _comparison(
            [_row_text(row) for row in partitions[left]],
            [_row_text(row) for row in partitions[right]],
            split_embeddings[left], split_embeddings[right],
        )

    exact_total = sum(item["exact_overlap_n"] for item in comparisons.values())
    normalized_total = sum(item["normalized_overlap_n"] for item in comparisons.values())
    extreme_total = sum(item["semantic_extreme_n"] for item in comparisons.values())
    external_raw = ROOT / "Data/evals/safety/dep001a/external_official_raw_results.json"
    external_metadata = json.loads(external_raw.read_text(encoding="utf-8"))
    artifact = {
        "schema_version": "dep001d_development_overlap_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if exact_total == 0 and normalized_total == 0 else "failed",
        "new_case_n": len(all_new),
        "comparisons": comparisons,
        "split_comparisons": split_comparisons,
        "aggregate": {
            "exact_overlap_n": exact_total,
            "normalized_overlap_n": normalized_total,
            "semantic_extreme_n_at_0_9995": extreme_total,
        },
        "burned_external_400": {
            "case_n": int(external_metadata.get("case_count") or 0),
            "holdout_sha256": external_metadata.get("holdout_sha256"),
            "text_overlap_auditable": False,
            "reason": "official_external_inputs_were_intentionally_not_persisted",
        },
        "prompt_or_neighbor_text_persisted": False,
        "consumed_bank_used_for_training": False,
        "clinical_validation": False,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def _comparison(left: list[str], right: list[str], left_embedding: np.ndarray, right_embedding: np.ndarray) -> dict[str, Any]:
    exact_right = {_sha(text) for text in right}
    normalized_right = {_sha(_normalize(text)) for text in right}
    exact_n = sum(_sha(text) in exact_right for text in left)
    normalized_n = sum(_sha(_normalize(text)) in normalized_right for text in left)
    maxima = _batched_max(left_embedding, right_embedding)
    return {
        "left_n": len(left),
        "right_n": len(right),
        "exact_overlap_n": int(exact_n),
        "normalized_overlap_n": int(normalized_n),
        "semantic_max": round(float(maxima.max(initial=0.0)), 6),
        "semantic_p99": round(float(np.quantile(maxima, 0.99)), 6) if maxima.size else 0.0,
        "semantic_extreme_n": int(np.sum(maxima >= 0.9995)),
    }


def _batched_max(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros(len(left), dtype="float32")
    index = faiss.IndexHNSWFlat(int(right.shape[1]), 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 96
    index.add(np.ascontiguousarray(right, dtype="float32"))
    values, _ = index.search(np.ascontiguousarray(left, dtype="float32"), 1)
    return values[:, 0]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row_text(row: dict[str, Any]) -> str:
    turns = row.get("turns") or []
    return " ".join(str(value) for value in turns) if turns else str(row.get("text") or "")


def _normalize(value: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["AUDIT_PATH", "run_dep001d_overlap_audit"]
