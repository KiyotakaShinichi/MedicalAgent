"""Emit a reproducible, non-training LoRA/QLoRA execution plan.

No model weights are loaded and no adapter is trained. The dry run verifies
that only the training split is selected, checks its hash against the split
manifest when available, and records every prerequisite that a future GPU run
must satisfy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "Data" / "finetune" / "prepared" / "dataset_train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "finetune" / "runs"
CANDIDATE_CONFIG = ROOT / "config" / "finetune_candidate.json"


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_CANDIDATE = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))

PLAN = {
    "execution_mode": "dry_run_only",
    "base_model": {
        "model_id": _CANDIDATE["model_id"],
        "revision": _CANDIDATE["revision"],
        "tokenizer_revision": _CANDIDATE["tokenizer_revision"],
        "license": _CANDIDATE["license"],
        "license_review": _CANDIDATE["license_review"],
        "official_model_card": _CANDIDATE["official_model_card"],
        "selection_policy": "local instruction model with documented license and pinned revision",
    },
    "adapter_name": "nlcare-behavior-v0-dryrun",
    "method": "QLoRA_candidate_plan",
    "quantization": "4bit_nf4_double_quant_candidate",
    "compute_dtype": "bfloat16_when_supported",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "epochs": 1,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.0001,
    "warmup_ratio": 0.05,
    "max_seq_len": 1536,
    "seed": 20260721,
    "early_stopping": "development_behavior_score_with_safety_tripwire",
    "checkpoint_policy": "adapter_only; keep at most two; never select on frozen holdout",
    "intended_behavior_targets": [
        "clinician_summary",
        "missing_data_disclosure",
        "questions_to_ask_care_team",
        "supplement_boundary",
        "taglish_safety",
        "emotional_support",
        "privacy_boundary",
        "tool_confirmation",
        "out_of_scope_redirect",
        "uncertainty_disclosure",
    ],
    "not_intended_for": [
        "medical_knowledge_injection",
        "diagnosis",
        "treatment_recommendation",
        "dosage_change",
        "prognosis_estimate",
        "genetic_risk_prediction",
        "tumor_marker_conclusion",
        "supplement_safety_authority",
        "clinical_decision_support",
    ],
    "mandatory_runtime_layers": [
        "agent_safety.safety_scope_check",
        "agent_input_gate.input_guardrail_check",
        "agent_output_gate.output_guardrail_check",
        "agent_post_gen.apply_post_gen_validator",
        "medical_claim_boundary.classify_medical_claim",
        "rag_claim_validator.validate_claims",
    ],
}


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _verify_lineage(dataset: Path) -> dict[str, Any]:
    if "holdout" in dataset.name.lower():
        raise ValueError("Fine-tuning dry run refuses to use a holdout file as training data.")
    manifest_path = dataset.parent / "split_manifest.json"
    actual_hash = _sha256_file(dataset)
    result: dict[str, Any] = {
        "dataset_sha256": actual_hash,
        "split_manifest_path": _rel(manifest_path),
        "split_manifest_present": manifest_path.exists(),
        "manifest_hash_match": None,
        "declared_split": None,
    }
    if not manifest_path.exists():
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split_name, details in (manifest.get("splits") or {}).items():
        if Path(str(details.get("path", ""))).name == dataset.name:
            result["declared_split"] = split_name
            result["manifest_hash_match"] = details.get("sha256") == actual_hash
            if split_name != "train":
                raise ValueError(f"Fine-tuning requires train split, received {split_name}.")
            if not result["manifest_hash_match"]:
                raise ValueError("Training split hash does not match split_manifest.json.")
            break
    if result["declared_split"] is None:
        raise ValueError("Dataset is not declared in split_manifest.json.")
    return result


def run_dryrun(dataset: Path, output_dir: Path) -> dict[str, Any]:
    if not dataset.exists():
        raise FileNotFoundError(
            f"Prepared training split not found at {dataset}. Run "
            "`python scripts/prepare_finetune_dataset.py` first."
        )
    lineage = _verify_lineage(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    example_count = _line_count(dataset)
    prerequisites = {
        "training_split_hash_verified": lineage["manifest_hash_match"] is True,
        "base_model_revision_pinned": bool(PLAN["base_model"]["revision"]),
        "tokenizer_revision_pinned": bool(PLAN["base_model"]["tokenizer_revision"]),
        "license_review_complete": PLAN["base_model"]["license_review"].startswith("recorded_"),
        "gpu_runtime_verified": False,
        "baseline_generations_complete": False,
        "candidate_generations_complete": False,
        "external_review_complete": False,
    }
    run_manifest = {
        "schema_version": "finetune_dryrun_manifest_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run_completed_not_ready_for_training",
        "clinical_validation": False,
        "model_trained": False,
        "dataset_path": _rel(dataset),
        "dataset_example_count": example_count,
        "lineage": lineage,
        "plan": PLAN,
        "prerequisites": prerequisites,
        "ready_for_real_training": all(prerequisites.values()),
        "executed_steps": [
            "verify_training_split",
            "verify_lineage_when_manifest_present",
            "emit_reproducible_training_plan",
            "emit_model_card_stub",
        ],
        "skipped_steps": [
            "load_base_model_weights",
            "tokenize_dataset",
            "instantiate_adapter",
            "trainer_loop",
            "save_adapter_weights",
            "generate_baseline_or_candidate_outputs",
        ],
        "claim_boundary": (
            "No model was trained. Any future adapter is behavior-only, remains "
            "behind every safety layer, and cannot establish clinical validity."
        ),
    }
    manifest_path = output_dir / "latest_dryrun_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    model_card = {
        "schema_version": "finetune_model_card_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "template_only_no_adapter",
        "clinical_validation": False,
        "model_trained": False,
        "base_model": PLAN["base_model"],
        "adapter_name": PLAN["adapter_name"],
        "intended_behavior": PLAN["intended_behavior_targets"],
        "not_intended_for": PLAN["not_intended_for"],
        "training_data": {
            "path": _rel(dataset),
            "sha256": lineage["dataset_sha256"],
            "example_count": example_count,
            "all_synthetic": True,
        },
        "evaluation_results": {
            "status": "candidate_generations_not_available",
            "promotion_decision": "HOLD",
        },
        "mandatory_runtime_layers": PLAN["mandatory_runtime_layers"],
        "known_limitations": [
            "No adapter exists.",
            "The training data is small, synthetic, and internally authored.",
            "The internal frozen split is not independent external evidence.",
            "A behavior adapter cannot replace source-governed RAG or safety gates.",
        ],
        "claim_boundary": run_manifest["claim_boundary"],
    }
    model_card_path = output_dir / "latest_model_card.json"
    model_card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    return {
        "manifest_path": _rel(manifest_path),
        "model_card_path": _rel(model_card_path),
        "dataset_example_count": example_count,
        "lineage": lineage,
        "plan": PLAN,
        "prerequisites": prerequisites,
        "ready_for_real_training": run_manifest["ready_for_real_training"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NLCare LoRA/QLoRA dry run.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(run_dryrun(args.dataset, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
