"""Aggregate-only overlap audit for DEP-001B corpora.

The burned external bank is processed mechanically. No external text, case ID,
or nearest-neighbor pair is emitted. The output is unsuitable for tuning and
exists only to detect accidental contamination.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "Data/evals/safety/dep001b"
OUTPUT_PATH = OUTPUT_DIR / "latest_overlap_audit.json"
CONFIG_PATH = ROOT / "config/dep001b_semantic_safety.yaml"
NEW_PATHS = {
    "train": OUTPUT_DIR / "train_safety_bank.jsonl",
    "validation": OUTPUT_DIR / "validation_safety_bank.jsonl",
    "internal_test": OUTPUT_DIR / "internal_test_safety_bank.jsonl",
    "internal_blind": OUTPUT_DIR / "internal_blind_safety_bank.jsonl",
}
PREVIOUS_PATHS = {
    "dep001a_development": ROOT / "Data/evals/safety/dep001a/development_semantic_safety_bank.jsonl",
    "dep001a_validation": ROOT / "Data/evals/safety/dep001a/validation_semantic_safety_bank.jsonl",
}
DEFAULT_BURNED_EXTERNAL_PATH = Path(r"C:\Users\L\Downloads\dep001_external_holdout_chatgpt_400.json")


def run_overlap_audit(external_path: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    new_rows = {name: _jsonl(path) for name, path in NEW_PATHS.items()}
    prior_rows = {name: _jsonl(path) for name, path in PREVIOUS_PATHS.items() if path.is_file()}
    burned_path = Path(external_path or os.getenv("NLCARE_BURNED_DEP001_EXTERNAL_PATH", "") or DEFAULT_BURNED_EXTERNAL_PATH)
    burned_rows = _external_rows(burned_path) if burned_path.is_file() else []

    exact = {}
    for new_name, rows in new_rows.items():
        new_hashes = {_text_hash(_row_text(row)) for row in rows}
        for prior_name, prior in prior_rows.items():
            exact[f"{new_name}_vs_{prior_name}"] = len(new_hashes & {_text_hash(_row_text(row)) for row in prior})
        if burned_rows:
            exact[f"{new_name}_vs_burned_external"] = len(new_hashes & {_text_hash(text) for text in burned_rows})

    semantic = {
        "status": "not_run",
        "encoder": config["base_encoder"],
        "thresholds": [0.90, 0.95, 0.98],
        "comparisons": {},
    }
    if burned_rows:
        encoder = SentenceTransformer(str(config["base_encoder"]), local_files_only=True)
        external_embeddings = encoder.encode(burned_rows, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        for new_name, rows in new_rows.items():
            embeddings = encoder.encode([_row_text(row) for row in rows], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
            maxima = _batched_max_similarity(embeddings, external_embeddings)
            semantic["comparisons"][f"{new_name}_vs_burned_external"] = {
                "n": len(maxima),
                "maximum": round(float(np.max(maxima)), 6),
                "p95": round(float(np.quantile(maxima, 0.95)), 6),
                "p99": round(float(np.quantile(maxima, 0.99)), 6),
                "count_ge_0_90": int(np.sum(maxima >= 0.90)),
                "count_ge_0_95": int(np.sum(maxima >= 0.95)),
                "count_ge_0_98": int(np.sum(maxima >= 0.98)),
            }
        semantic["status"] = "completed_aggregate_only"

    exact_external = sum(value for key, value in exact.items() if key.endswith("burned_external"))
    payload = {
        "schema_version": "dep001b_overlap_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if exact_external == 0 else "failed_exact_external_overlap",
        "exact_overlap_counts": exact,
        "exact_burned_external_overlap_count": exact_external,
        "semantic_overlap": semantic,
        "external_bank_available_for_mechanical_check": bool(burned_rows),
        "external_bank_case_n": len(burned_rows),
        "external_text_or_case_ids_emitted": False,
        "used_for_tuning": False,
        "limitations": [
            "Semantic similarity is a contamination diagnostic, not proof of independence.",
            "Aggregate counts are not labels and must not be used to tune the candidate.",
        ],
        "clinical_validation": False,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _batched_max_similarity(left: np.ndarray, right: np.ndarray, batch_size: int = 256) -> np.ndarray:
    maxima = []
    for start in range(0, len(left), batch_size):
        scores = left[start:start + batch_size] @ right.T
        maxima.extend(np.max(scores, axis=1).tolist())
    return np.asarray(maxima, dtype=float)


def _row_text(row: dict[str, Any]) -> str:
    turns = row.get("turns") or row.get("conversation_turns") or []
    normalized_turns = [
        str(item.get("content") or item.get("text") or "") if isinstance(item, dict) else str(item)
        for item in turns
    ]
    return str(row.get("text") or " [TURN] ".join(normalized_turns)).strip()


def _external_rows(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("cases") or raw.get("items") or raw.get("rows") or []
    return [text for row in rows if isinstance(row, dict) and (text := _row_text(row))]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = ["run_overlap_audit"]
