"""One-way integrity checks for sealed DEP-001 and DEP-001A banks.

This diagnostic may hash sealed rows, but it never returns or persists holdout
text, labels, tokens, nearest examples, or case-level failures. Training and
runtime modules do not import it.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEALED_BANK = ROOT / "Data/evals/safety/dep001/final_holdout_safety_bank.jsonl"
SEALED_MANIFEST = ROOT / "Data/evals/safety/dep001/final_holdout_manifest.json"
DEP001A_DIR = ROOT / "Data/evals/safety/dep001a"
INTEGRITY_PATH = DEP001A_DIR / "latest_holdout_integrity.json"
PROVENANCE_PATH = DEP001A_DIR / "bank_provenance_registry.json"


def verify_holdout_integrity() -> dict[str, Any]:
    manifest = json.loads(SEALED_MANIFEST.read_text(encoding="utf-8"))
    expected_sha = str(manifest.get("sha256") or "")
    actual_sha = _sha256(SEALED_BANK)
    development = DEP001A_DIR / "development_semantic_safety_bank.jsonl"
    validation = DEP001A_DIR / "validation_semantic_safety_bank.jsonl"

    # Hash-only comparison. No holdout text or per-row value leaves this scope.
    sealed_hashes = _normalized_row_hashes(SEALED_BANK)
    dev_hashes = _normalized_row_hashes(development)
    val_hashes = _normalized_row_hashes(validation)
    overlap_dev = len(sealed_hashes & dev_hashes)
    overlap_val = len(sealed_hashes & val_hashes)
    result = {
        "schema_version": "dep001a_holdout_integrity_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if actual_sha == expected_sha and overlap_dev == 0 and overlap_val == 0 else "failed",
        "sealed_bank_sha256_matches_manifest": actual_sha == expected_sha,
        "development_exact_overlap_n": overlap_dev,
        "validation_exact_overlap_n": overlap_val,
        "holdout_text_exposed": False,
        "holdout_case_details_emitted": False,
        "training_imports_this_module": False,
        "clinical_validation": False,
    }
    DEP001A_DIR.mkdir(parents=True, exist_ok=True)
    INTEGRITY_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_provenance_registry(development, validation, manifest)
    return result


def _write_provenance_registry(development: Path, validation: Path, sealed_manifest: dict[str, Any]) -> None:
    legacy_dir = ROOT / "Data/evals/safety/dep001"
    records = [
        _bank_record(legacy_dir / "development_safety_bank.jsonl", "legacy_development", True, "internal_engineering"),
        _bank_record(legacy_dir / "validation_safety_bank.jsonl", "legacy_validation_tuning_informed", True, "internal_engineering"),
        _bank_record(SEALED_BANK, "sealed_historical_final", False, str(sealed_manifest.get("authored_by") or "unknown")),
        _bank_record(development, "dep001a_development", True, "independent_threat_grammar"),
        _bank_record(validation, "dep001a_validation", True, "independent_threat_grammar"),
    ]
    payload = {
        "schema_version": "dep001a_bank_provenance_registry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "banks": records,
        "future_external_holdout": {
            "status": "not_created",
            "available_during_implementation": False,
            "must_be_external_human_no_read": True,
        },
        "clinical_validation": False,
    }
    PROVENANCE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bank_record(path: Path, purpose: str, used_for_tuning: bool, authoring_method: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "purpose": purpose,
        "was_used_for_tuning": used_for_tuning,
        "authoring_method": authoring_method,
        "exists": path.exists(),
    }


def _normalized_row_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        turns = row.get("turns") if isinstance(row, dict) else None
        text = " [TURN] ".join(str(item) for item in turns) if isinstance(turns, list) else str(row.get("text") or "")
        normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()
        hashes.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return hashes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["INTEGRITY_PATH", "PROVENANCE_PATH", "verify_holdout_integrity"]
