"""Frozen evaluation registry, integrity checks, and contamination diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "config/evaluation_dataset_registry.json"
DEFAULT_OUTPUT = ROOT / "Data/evals/governance/latest_next_generation_eval_integrity.json"
PROMPT_FIELDS = ("query", "user_query", "prompt", "message", "text", "input")
CLAIM_BOUNDARY = (
    "Engineering evaluation provenance only. Integrity and contamination checks do not establish "
    "independent external evaluation, clinical validation, real-world safety, patient benefit, or "
    "production healthcare readiness."
)


class FrozenDatasetIntegrityError(RuntimeError):
    pass


def normalized_content_sha256(path: Path | str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_registry(registry_path: Path | str = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_file = Path(registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    datasets: list[dict[str, Any]] = []
    integrity_failures: list[dict[str, Any]] = []
    prompt_sets: dict[str, set[str]] = {}

    for configured in registry.get("datasets") or []:
        row = dict(configured)
        relative = row.get("path")
        expected_hash = row.get("normalized_content_sha256")
        if not relative:
            row.update({"exists": False, "integrity_status": "generated_matrix_no_static_file", "case_count": None})
            datasets.append(row)
            continue
        path = ROOT / str(relative)
        row["exists"] = path.exists()
        if not path.exists():
            external = row.get("external_status") == "BLOCKED_EXTERNAL"
            row["integrity_status"] = "BLOCKED_EXTERNAL" if external else "missing"
            row["case_count"] = 0
            if row.get("frozen") and not external:
                integrity_failures.append({
                    "dataset_name": row.get("dataset_name"),
                    "reason": "frozen_dataset_missing",
                    "path": relative,
                })
            datasets.append(row)
            continue

        observed_hash = normalized_content_sha256(path)
        rows = _load_rows(path)
        row["case_count"] = len(rows)
        row["observed_normalized_content_sha256"] = observed_hash
        row["integrity_status"] = "verified" if not expected_hash or observed_hash == expected_hash else "hash_mismatch"
        if row.get("frozen") and expected_hash and observed_hash != expected_hash:
            integrity_failures.append({
                "dataset_name": row.get("dataset_name"),
                "reason": "frozen_hash_mismatch",
                "path": relative,
                "expected_hash": expected_hash,
                "observed_hash": observed_hash,
            })
        prompt_sets[str(row.get("dataset_name"))] = set(_prompts(rows))
        datasets.append(row)

    overlaps = _contamination_overlaps(datasets, prompt_sets)
    independently_authored_external = sum(
        1 for row in datasets
        if row.get("origin") == "independently_authored_external" and row.get("integrity_status") == "verified"
    )
    blocked_external = [
        str(row.get("dataset_name")) for row in datasets
        if row.get("integrity_status") == "BLOCKED_EXTERNAL"
    ]
    return {
        "schema_version": "next_generation_eval_integrity_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if integrity_failures else "verified_with_external_gaps",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "registry_path": _relative(registry_file),
        "registry_version": registry.get("registry_version"),
        "dataset_count": len(datasets),
        "required_evaluation_class_count": 10,
        "evaluation_classes": sorted(str(row.get("evaluation_class")) for row in datasets),
        "datasets": datasets,
        "integrity_failure_count": len(integrity_failures),
        "integrity_failures": integrity_failures,
        "exact_prompt_overlap_diagnostics": overlaps,
        "independently_authored_external_completed_count": independently_authored_external,
        "blocked_external": blocked_external,
        "external_review_status": "BLOCKED_EXTERNAL" if blocked_external else "available",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_integrity_report(
    output_path: Path | str = DEFAULT_OUTPUT,
    registry_path: Path | str = DEFAULT_REGISTRY,
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    payload = verify_registry(registry_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if raise_on_failure and payload["integrity_failure_count"]:
        raise FrozenDatasetIntegrityError(
            f"{payload['integrity_failure_count']} frozen evaluation dataset integrity failure(s)"
        )
    return payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    for key in ("cases", "rows", "prompts"):
        if isinstance(payload.get(key), list):
            return [row for row in payload[key] if isinstance(row, dict)]
    return []


def _prompts(rows: Iterable[Mapping[str, Any]]) -> Iterable[str]:
    for row in rows:
        for field in PROMPT_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                normalized = re.sub(r"\s+", " ", value.strip().lower())
                yield normalized
                break


def _contamination_overlaps(
    datasets: list[Mapping[str, Any]],
    prompt_sets: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    by_name = {str(row.get("dataset_name")): row for row in datasets}
    names = sorted(prompt_sets)
    results: list[dict[str, Any]] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = prompt_sets[left_name] & prompt_sets[right_name]
            if not overlap:
                continue
            left = by_name[left_name]
            right = by_name[right_name]
            higher_risk = bool(
                (left.get("was_used_for_tuning") and right.get("frozen"))
                or (right.get("was_used_for_tuning") and left.get("frozen"))
            )
            results.append({
                "left_dataset": left_name,
                "right_dataset": right_name,
                "exact_normalized_overlap_count": len(overlap),
                "overlap_risk": "needs_attention" if higher_risk else "informational",
                "examples_sha256": sorted(hashlib.sha256(item.encode()).hexdigest() for item in overlap)[:5],
            })
    return results


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "FrozenDatasetIntegrityError",
    "normalized_content_sha256",
    "verify_registry",
    "write_integrity_report",
]
