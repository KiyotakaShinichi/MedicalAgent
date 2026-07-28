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
MINIMUM_TRAIN_EXAMPLES = 100
MINIMUM_DEVELOPMENT_EXAMPLES = 25
MINIMUM_INTERNAL_HOLDOUT_EXAMPLES = 50


def build_execution_readiness(card: dict, dryrun: dict, promotion: dict, runtime_preflight: dict | None = None) -> dict:
    counts = card["example_counts"]["by_split"]
    prerequisites = dryrun.get("prerequisites") or {}
    checks = [
        {
            "id": "contamination_audit_clear",
            "passed": card["contamination_audit"]["status"] == "acceptable"
            and card["contamination_audit"]["exact_overlap_count"] == 0,
            "required_for": "experimental_training",
        },
        {
            "id": "training_case_floor",
            "passed": counts.get("train", 0) >= MINIMUM_TRAIN_EXAMPLES,
            "observed": counts.get("train", 0),
            "required": MINIMUM_TRAIN_EXAMPLES,
            "required_for": "experimental_training",
        },
        {
            "id": "development_case_floor",
            "passed": counts.get("development", 0) >= MINIMUM_DEVELOPMENT_EXAMPLES,
            "observed": counts.get("development", 0),
            "required": MINIMUM_DEVELOPMENT_EXAMPLES,
            "required_for": "experimental_training",
        },
        {
            "id": "internal_holdout_case_floor",
            "passed": counts.get("internal_frozen_holdout", 0) >= MINIMUM_INTERNAL_HOLDOUT_EXAMPLES,
            "observed": counts.get("internal_frozen_holdout", 0),
            "required": MINIMUM_INTERNAL_HOLDOUT_EXAMPLES,
            "required_for": "candidate_comparison",
        },
        {
            "id": "base_model_revision_pinned",
            "passed": prerequisites.get("base_model_revision_pinned") is True,
            "required_for": "experimental_training",
        },
        {
            "id": "tokenizer_revision_pinned",
            "passed": prerequisites.get("tokenizer_revision_pinned") is True,
            "required_for": "experimental_training",
        },
        {
            "id": "license_review_complete",
            "passed": prerequisites.get("license_review_complete") is True,
            "required_for": "experimental_training",
        },
        {
            "id": "baseline_generations_complete",
            "passed": prerequisites.get("baseline_generations_complete") is True,
            "required_for": "candidate_comparison",
        },
        {
            "id": "candidate_generations_complete",
            "passed": prerequisites.get("candidate_generations_complete") is True,
            "required_for": "candidate_comparison",
        },
        {
            "id": "promotion_gate_not_hold",
            "passed": promotion.get("decision") in {"PROMOTE", "REJECT"},
            "required_for": "candidate_decision",
        },
    ]
    if runtime_preflight is not None:
        checks.append({
            "id": "training_runtime_healthy_and_explicitly_enabled",
            "passed": runtime_preflight.get("ready_for_offline_experiment") is True,
            "required_for": "experimental_training",
        })
    failed = [check["id"] for check in checks if not check["passed"]]
    return {
        "state": "ready_for_offline_shadow_candidate" if not failed else "not_ready_for_training_or_promotion",
        "training_ready": not any(
            not check["passed"] and check["required_for"] == "experimental_training" for check in checks
        ),
        "promotion_ready": not any(
            not check["passed"] and check["required_for"] in {"candidate_comparison", "candidate_decision"}
            for check in checks
        ),
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "failed_checks": failed,
        "checks": checks,
        "independent_external_evidence": False,
    }


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
    runtime_path = ROOT / "Data" / "evals" / "models" / "latest_finetune_runtime_preflight.json"
    runtime_preflight = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {
        "status": "missing",
        "ready_for_offline_experiment": False,
    }
    readiness = build_execution_readiness(card, dryrun, promotion, runtime_preflight)
    report = {
        "schema_version": "finetune_governance_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention",
        "clinical_validation": False,
        "model_trained": False,
        "readiness_state": readiness["state"],
        "training_ready": readiness["training_ready"],
        "promotion_ready": readiness["promotion_ready"],
        "dataset": {
            "accepted": card["example_counts"]["accepted_total"],
            "rejected": card["example_counts"]["rejected_total"],
            "by_split": card["example_counts"]["by_split"],
            "contamination_status": card["contamination_audit"]["status"],
            "exact_overlap_count": card["contamination_audit"]["exact_overlap_count"],
        },
        "dry_run": dryrun,
        "runtime_preflight": runtime_preflight,
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
        "execution_readiness": readiness,
        "remaining_blockers": [
            "The local PEFT training runtime is not healthy and explicitly enabled.",
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
