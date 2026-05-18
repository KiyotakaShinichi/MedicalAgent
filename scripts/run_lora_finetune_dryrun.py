"""LoRA fine-tuning DRY-RUN — no real training, no GPU required.

This script mimics the structural shape of a LoRA training run for the
behavior dataset.  It does NOT load weights, does NOT call a trainer,
does NOT consume any compute beyond writing two artifacts.

What it actually does:

  1. Verifies the prepared dataset exists.
  2. Echoes a deterministic "training plan" (base model placeholder,
     epoch count, LoRA rank, target modules) into a run manifest.
  3. Emits a model-card stub for the would-be adapter.

The point of the dry-run is to keep the scaffold testable on CI / on
the same machines that ship pytest, without requiring HF, transformers,
or peft.  Replacing the deterministic body with a real
``transformers.Trainer.train()`` is left for a future contributor who
has GPU access; the dataset card already enumerates what must remain
true after training (claim boundary + safety filters preserved).

Usage
~~~~~
    python scripts/run_lora_finetune_dryrun.py
    python scripts/run_lora_finetune_dryrun.py --dataset data/finetune/prepared/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "data" / "finetune" / "prepared" / "dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "finetune" / "runs"


def _rel(path: Path) -> str:
    """Stringify a path relative to ROOT when possible; fall back to the
    absolute form for paths outside the repo (tests use temp dirs)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


PLAN = {
    "base_model_placeholder":   "local Ollama / quantized 7B-class model TBD",
    "adapter_name_placeholder": "oncotrack-behavior-v0-dryrun",
    "lora_rank":                16,
    "lora_alpha":                32,
    "target_modules":           ["q_proj", "v_proj"],
    "epochs":                   3,
    "batch_size":               4,
    "learning_rate":            2e-4,
    "warmup_steps":             10,
    "max_seq_len":              2048,
    "intended_behavior_targets": [
        "clinician_summary",
        "missing_data_disclosure",
        "questions_to_ask_care_team",
        "supplement_boundary",
        "taglish_safety",
    ],
    "not_intended_for": [
        "diagnosis",
        "treatment_recommendation",
        "dosage_change",
        "prognosis_estimate",
        "genetic_risk_prediction",
        "tumor_marker_conclusion",
        "supplement_safe_with_chemo_claim",
        "any clinical decision support",
    ],
    "safety_validator_compatibility": [
        "agent_safety.safety_scope_check",
        "agent_input_gate.input_guardrail_check",
        "agent_output_gate.output_guardrail_check",
        "agent_post_gen.apply_post_gen_validator",
        "medical_claim_boundary.classify_medical_claim",
        "rag_claim_validator.validate_claims",
    ],
}


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def run_dryrun(dataset: Path, output_dir: Path) -> dict[str, Any]:
    if not dataset.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {dataset}. Run "
            f"`python scripts/prepare_finetune_dataset.py` first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    example_count = _line_count(dataset)

    run_manifest = {
        "schema_version":     "finetune_dryrun_manifest_v1",
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "status":             "dry_run_completed",
        "dataset_path":       _rel(dataset),
        "dataset_example_count": example_count,
        "plan":               PLAN,
        "executed_steps": [
            "verify_dataset_present",
            "echo_training_plan",
            "emit_model_card_stub",
        ],
        "skipped_steps": [
            "load_base_model_weights",
            "tokenize_dataset",
            "instantiate_lora_adapter",
            "trainer_loop",
            "save_adapter_weights",
            "evaluate_on_held_out",
        ],
        "skipped_reason":     "Default scaffold runs without GPU / transformers / peft.",
        "claim_boundary":     (
            "Dry-run: no model was trained. The manifest reflects the plan "
            "that a future training run would execute. Any real adapter "
            "must still pass the OncoTrack safety stack."
        ),
    }
    manifest_path = output_dir / "latest_dryrun_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    model_card_stub = {
        "schema_version":    "finetune_model_card_v1",
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "base_model":        PLAN["base_model_placeholder"],
        "adapter_name":      PLAN["adapter_name_placeholder"],
        "intended_behavior": PLAN["intended_behavior_targets"],
        "not_intended_for":  PLAN["not_intended_for"],
        "evaluation_results": {
            "status": "not_evaluated_dry_run",
            "note":   "Run scripts/evaluate_finetuned_behavior.py to populate.",
        },
        "safety_validator_compatibility": PLAN["safety_validator_compatibility"],
        "claim_boundary":    run_manifest["claim_boundary"],
    }
    card_path = output_dir / "latest_model_card.json"
    card_path.write_text(json.dumps(model_card_stub, indent=2), encoding="utf-8")

    return {
        "manifest_path":   _rel(manifest_path),
        "model_card_path": _rel(card_path),
        "dataset_example_count": example_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OncoTrack LoRA fine-tune DRY-RUN (no training).")
    parser.add_argument("--dataset",     type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    summary = run_dryrun(args.dataset, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
