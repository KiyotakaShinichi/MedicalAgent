"""Compatibility exporter for the governed NLCare fine-tuning dataset.

This script no longer owns a second embedded example bank. It runs the
canonical preparer and exports the verified training split for the optional
QLoRA experiment, preventing two tuning paths from drifting apart.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_finetune_dataset import prepare_dataset


def build(output_dir: Path) -> dict:
    card = prepare_dataset()
    source = ROOT / card["files"]["dataset_train"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "behavior_tuning_examples.jsonl"
    shutil.copyfile(source, destination)
    plan = {
        "schema_version": "qlora_behavior_experiment_v2",
        "status": "dataset_exported_no_model_trained",
        "clinical_validation": False,
        "model_trained": False,
        "patient_facing_promotion_allowed": False,
        "purpose": "Behavior and format experiment only; not medical-knowledge tuning.",
        "canonical_dataset_card": card["files"]["dataset_card"],
        "canonical_split_manifest": card["files"]["split_manifest"],
        "training_split": destination.relative_to(ROOT).as_posix(),
        "example_count": card["example_counts"]["by_split"].get("train", 0),
        "promotion_decision": "HOLD",
        "next_step": "Pin base model/tokenizer revisions, then run the experimental preflight.",
        "claim_boundary": (
            "Synthetic behavior-only export. No adapter was trained and no "
            "patient-facing or clinical promotion is allowed."
        ),
    }
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Export governed QLoRA behavior data.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "qlora_behavior")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
