"""Run the complete non-training fine-tuning governance scaffold."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_finetuned_behavior import evaluate_dataset
from scripts.prepare_finetune_dataset import prepare_dataset
from scripts.run_finetune_promotion_gate import run_gate
from scripts.run_lora_finetune_dryrun import run_dryrun

OUTPUT = ROOT / "Data" / "evals" / "models" / "latest_finetune_governance.json"


def run() -> dict:
    card = prepare_dataset()
    train = ROOT / card["files"]["dataset_train"]
    holdout = ROOT / card["files"]["dataset_internal_frozen_holdout"]
    dryrun = run_dryrun(train, ROOT / "Data" / "finetune" / "runs")
    reference_eval = evaluate_dataset(
        holdout,
        ROOT / "Data" / "evals" / "models" / "latest_finetune_scaffold_eval.json",
        subject_label="internal_frozen_reference_audit",
    )
    promotion = run_gate(holdout=holdout)
    report = {
        "schema_version": "finetune_governance_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention",
        "clinical_validation": False,
        "model_trained": False,
        "dataset": {
            "accepted": card["example_counts"]["accepted_total"],
            "rejected": card["example_counts"]["rejected_total"],
            "by_split": card["example_counts"]["by_split"],
            "contamination_status": card["contamination_audit"]["status"],
            "exact_overlap_count": card["contamination_audit"]["exact_overlap_count"],
        },
        "dry_run": dryrun,
        "reference_audit": {
            "status": reference_eval["status"],
            "is_model_evaluation": reference_eval["is_model_evaluation"],
            "unsafe_leakage_rate": reference_eval["unsafe_leakage_rate"],
            "claim_boundary_compliance": reference_eval["claim_boundary_compliance"],
            "behavior_contract_pass_rate": reference_eval["behavior_contract_pass_rate"],
        },
        "promotion": {
            "status": promotion["status"],
            "decision": promotion["decision"],
            "reason": promotion["reason"],
            "promotion_scope": promotion["promotion_scope"],
        },
        "remaining_blockers": [
            "No base model or tokenizer revision is pinned.",
            "No adapter has been trained.",
            "Baseline and candidate generations are absent.",
            "The dataset is small, synthetic, and internally authored.",
            "No external author or clinician has reviewed the outputs.",
        ],
        "claim_boundary": (
            "Behavior-only synthetic engineering scaffold. It does not train "
            "medical knowledge, prove safety, or provide clinical validation."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
