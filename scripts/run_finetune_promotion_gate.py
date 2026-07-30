"""Evaluate baseline/candidate generations and emit the fine-tune gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.finetune_promotion import build_promotion_decision
from scripts.evaluate_finetuned_behavior import evaluate_dataset

DEFAULT_HOLDOUT = ROOT / "Data" / "finetune" / "prepared" / "dataset_internal_frozen_holdout.jsonl"
DEFAULT_BASELINE = ROOT / "Data" / "finetune" / "evaluations" / "baseline_generations.jsonl"
DEFAULT_CANDIDATE = ROOT / "Data" / "finetune" / "evaluations" / "candidate_generations.jsonl"
DEFAULT_BASELINE_MANIFEST = ROOT / "Data" / "finetune" / "evaluations" / "baseline_generation_manifest.json"
DEFAULT_CANDIDATE_MANIFEST = ROOT / "Data" / "finetune" / "evaluations" / "candidate_generation_manifest.json"
DEFAULT_TRAIN = ROOT / "Data" / "finetune" / "prepared" / "dataset_train.jsonl"
DEFAULT_OUTPUT = ROOT / "Data" / "evals" / "models" / "latest_finetune_promotion_gate.json"


def run_gate(
    holdout: Path = DEFAULT_HOLDOUT,
    baseline_path: Path = DEFAULT_BASELINE,
    candidate_path: Path = DEFAULT_CANDIDATE,
    output_path: Path = DEFAULT_OUTPUT,
    baseline_manifest_path: Path = DEFAULT_BASELINE_MANIFEST,
    candidate_manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    train_path: Path = DEFAULT_TRAIN,
) -> dict:
    _validate_internal_holdout(holdout)
    baseline = None
    candidate = None
    evaluation_dir = output_path.parent
    if baseline_path.exists():
        baseline = evaluate_dataset(
            holdout,
            evaluation_dir / "latest_finetune_baseline_behavior_eval.json",
            baseline_path,
            "base_model_generations",
        )
        baseline["generation_manifest"] = _generation_manifest_check(
            baseline_manifest_path,
            baseline_path,
            holdout,
            expected_subject="baseline",
        )
        baseline["generation_manifest_verified"] = baseline["generation_manifest"]["verified"]
    if candidate_path.exists():
        candidate = evaluate_dataset(
            holdout,
            evaluation_dir / "latest_finetune_candidate_behavior_eval.json",
            candidate_path,
            "adapter_candidate_generations",
        )
        candidate["generation_manifest"] = _generation_manifest_check(
            candidate_manifest_path,
            candidate_path,
            holdout,
            expected_subject="candidate",
        )
        candidate["generation_manifest_verified"] = candidate["generation_manifest"]["verified"]
        candidate["memorization_audit"] = _train_output_memorization_audit(
            train_path,
            candidate_path,
        )
    decision = build_promotion_decision(baseline, candidate)
    decision["evidence"] = {
        "holdout_path": _rel(holdout),
        "baseline_generations_path": _rel(baseline_path),
        "candidate_generations_path": _rel(candidate_path),
        "baseline_present": baseline_path.exists(),
        "candidate_present": candidate_path.exists(),
        "baseline_manifest_path": _rel(baseline_manifest_path),
        "candidate_manifest_path": _rel(candidate_manifest_path),
        "candidate_generation_lineage_verified": bool(
            candidate and candidate.get("generation_manifest_verified")
        ),
        "candidate_memorization_audit_completed": bool(
            candidate and (candidate.get("memorization_audit") or {}).get("completed")
        ),
        "internal_holdout_is_external_evidence": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def _validate_internal_holdout(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Fine-tune holdout is empty.")
    for row in rows:
        if row.get("split") != "internal_frozen_holdout":
            raise ValueError(f"Case {row.get('id')} is not in the internal frozen holdout.")
        if (row.get("provenance") or {}).get("was_used_for_tuning") is not False:
            raise ValueError(f"Case {row.get('id')} does not prove tuning exclusion.")


def _generation_manifest_check(
    manifest_path: Path,
    generations_path: Path,
    holdout_path: Path,
    *,
    expected_subject: str,
) -> dict:
    required = {
        "schema_version",
        "subject",
        "base_model",
        "base_revision",
        "tokenizer_revision",
        "generations_sha256",
        "evaluation_holdout_sha256",
        "clinical_validation",
        "patient_facing_use_allowed",
    }
    if not manifest_path.exists():
        return {
            "verified": False,
            "reason": "manifest_missing",
            "path": _rel(manifest_path),
            "missing_fields": sorted(required),
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "verified": False,
            "reason": f"manifest_unreadable:{exc.__class__.__name__}",
            "path": _rel(manifest_path),
            "missing_fields": [],
        }
    missing = sorted(field for field in required if field not in payload)
    checks = {
        "subject_matches": payload.get("subject") == expected_subject,
        "generation_hash_matches": payload.get("generations_sha256") == _sha256(generations_path),
        "holdout_hash_matches": payload.get("evaluation_holdout_sha256") == _sha256(holdout_path),
        "base_revision_pinned": bool(payload.get("base_revision")),
        "tokenizer_revision_pinned": bool(payload.get("tokenizer_revision")),
        "clinical_validation_false": payload.get("clinical_validation") is False,
        "patient_facing_use_blocked": payload.get("patient_facing_use_allowed") is False,
    }
    return {
        "verified": not missing and all(checks.values()),
        "reason": "verified" if not missing and all(checks.values()) else "manifest_check_failed",
        "path": _rel(manifest_path),
        "missing_fields": missing,
        "checks": checks,
        "content": {
            "schema_version": payload.get("schema_version"),
            "subject": payload.get("subject"),
            "base_model": payload.get("base_model"),
            "base_revision": payload.get("base_revision"),
            "tokenizer_revision": payload.get("tokenizer_revision"),
            "adapter_sha256": payload.get("adapter_sha256"),
        },
    }


def _train_output_memorization_audit(train_path: Path, candidate_path: Path) -> dict:
    if not train_path.exists() or not candidate_path.exists():
        return {
            "completed": False,
            "reason": "training_or_candidate_file_missing",
            "exact_train_output_match_count": None,
        }
    train_outputs = set()
    for row in _jsonl(train_path):
        assistant = next(
            (
                str(message.get("content") or "")
                for message in row.get("messages") or []
                if message.get("role") == "assistant"
            ),
            "",
        )
        if assistant:
            train_outputs.add(_normalize(assistant))
    matches = []
    candidate_rows = _jsonl(candidate_path)
    for row in candidate_rows:
        output = next(
            (
                str(row.get(field))
                for field in ("assistant", "output", "text", "generation")
                if isinstance(row.get(field), str)
            ),
            "",
        )
        if output and _normalize(output) in train_outputs:
            matches.append(str(row.get("id") or row.get("case_id") or "unknown"))
    return {
        "completed": True,
        "method": "exact_normalized_candidate_output_vs_training_assistant_text",
        "training_output_count": len(train_outputs),
        "candidate_output_count": len(candidate_rows),
        "exact_train_output_match_count": len(matches),
        "matching_case_ids": matches,
        "semantic_memorization_measured": False,
        "limitation": "Exact normalized matching only; semantic memorization requires a separate audit.",
    }


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", value.lower())).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NLCare fine-tune promotion gate.")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_gate(
        args.holdout,
        args.baseline,
        args.candidate,
        args.output,
        args.baseline_manifest,
        args.candidate_manifest,
        args.train,
    )
    print(json.dumps(report, indent=2))
    return 1 if report["decision"] == "REJECT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
