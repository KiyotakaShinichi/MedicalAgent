"""Build statistical confidence reporting over existing eval artifacts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.statistical_eval import (  # noqa: E402
    CLAIM_BOUNDARY,
    STATISTICAL_EVAL_VERSION,
    binomial_metric,
    fold_mean_metric,
    two_proportion_delta,
)


OUTPUT_PATH = ROOT / "Data/evals/governance/latest_statistical_eval_summary.json"


def _load(rel_path: str) -> dict[str, Any] | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_statistical_summary(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    missing: list[str] = []

    def require(path: str) -> dict[str, Any] | None:
        payload = _load(path)
        if payload is None:
            missing.append(path)
        return payload

    adversarial = require("Data/evals/safety/latest_adversarial_safety_regression.json")
    if adversarial:
        metrics.append(binomial_metric(
            name="adversarial_original_bank_pass_rate",
            artifact_path="Data/evals/safety/latest_adversarial_safety_regression.json",
            successes=int(adversarial.get("pass_count", adversarial.get("total_passed", 0))),
            total=int(adversarial.get("total_n", adversarial.get("total_cases", 0))),
        ))

    holdout = require("Data/evals/safety/latest_adversarial_safety_holdout.json")
    if holdout:
        metrics.append(binomial_metric(
            name="adversarial_heldout_v1_pass_rate",
            artifact_path="Data/evals/safety/latest_adversarial_safety_holdout.json",
            successes=int(holdout.get("pass_count", 0)),
            total=int(holdout.get("total_n", 0)),
        ))

    generalization = require("Data/evals/safety/latest_adversarial_generalization_v2_eval.json")
    if generalization:
        for name, key in [
            ("adversarial_generalization_original_bank", "original_bank"),
            ("adversarial_generalization_heldout_v1", "heldout_v1"),
            ("adversarial_generalization_heldout_v2", "heldout_v2"),
            ("adversarial_generalization_paraphrase", "paraphrase_robustness"),
            ("adversarial_generalization_safe_negative_controls", "safe_negative_controls"),
        ]:
            item = generalization.get(key)
            if isinstance(item, dict) and item.get("total_n"):
                metrics.append(binomial_metric(
                    name=f"{name}_pass_rate",
                    artifact_path="Data/evals/safety/latest_adversarial_generalization_v2_eval.json",
                    successes=int(item.get("pass_count", 0)),
                    total=int(item.get("total_n", 0)),
                ))

    live_rag = require("Data/evals/rag/latest_live_rag_eval.json")
    if live_rag:
        summary = live_rag.get("summary") or {}
        n = int(summary.get("case_count", 0))
        metrics.append(binomial_metric(
            name="live_rag_pass_rate",
            artifact_path="Data/evals/rag/latest_live_rag_eval.json",
            successes=int(summary.get("passed", 0)),
            total=n,
        ))
        unsafe_successes = n - int(round(float(summary.get("unsafe_answer_rate", 0.0)) * n))
        metrics.append(binomial_metric(
            name="live_rag_safe_answer_rate",
            artifact_path="Data/evals/rag/latest_live_rag_eval.json",
            successes=unsafe_successes,
            total=n,
            estimate_name="safe_answer_rate",
        ))

    retrieval = require("Data/evals/rag/latest_retrieval_goldset_eval.json")
    if retrieval:
        strategies = retrieval.get("strategies") or {}
        best_name = retrieval.get("best_strategy") or "hybrid_rrf"
        best = strategies.get(best_name) or next(iter(strategies.values()), {})
        best_summary = best.get("summary") or {}
        n = int(best_summary.get("case_count") or retrieval.get("total_n") or 0)
        recall10 = float(best_summary.get("recall_at_10", 0.0))
        unsupported = float(best_summary.get("unsupported_answer_rate", 0.0))
        metrics.append(binomial_metric(
            name=f"retrieval_{best_name}_recall_at_10",
            artifact_path="Data/evals/rag/latest_retrieval_goldset_eval.json",
            successes=int(round(recall10 * n)),
            total=n,
            estimate_name="recall_at_10",
        ))
        metrics.append(binomial_metric(
            name=f"retrieval_{best_name}_supported_context_rate",
            artifact_path="Data/evals/rag/latest_retrieval_goldset_eval.json",
            successes=n - int(round(unsupported * n)),
            total=n,
            estimate_name="supported_context_rate",
        ))

    temporal_cv = require("Data/evals/models/latest_patient_temporal_cv.json")
    if temporal_cv:
        folds = ((temporal_cv.get("patient_level_temporal_cv") or {}).get("folds") or [])
        aucs = [float(row["roc_auc"]) for row in folds if row.get("roc_auc") is not None]
        briers = [float(row["brier"]) for row in folds if row.get("brier") is not None]
        metrics.append(fold_mean_metric(
            name="patient_temporal_cv_roc_auc_mean",
            artifact_path="Data/evals/models/latest_patient_temporal_cv.json",
            values=aucs,
            estimate_name="roc_auc",
        ))
        metrics.append(fold_mean_metric(
            name="patient_temporal_cv_brier_mean",
            artifact_path="Data/evals/models/latest_patient_temporal_cv.json",
            values=briers,
            estimate_name="brier",
        ))

    if adversarial and holdout:
        deltas.append({
            "name": "heldout_minus_original_adversarial_pass_rate",
            "claim_boundary": CLAIM_BOUNDARY,
            **two_proportion_delta(
                baseline_successes=int(adversarial.get("pass_count", adversarial.get("total_passed", 0))),
                baseline_total=int(adversarial.get("total_n", adversarial.get("total_cases", 0))),
                candidate_successes=int(holdout.get("pass_count", 0)),
                candidate_total=int(holdout.get("total_n", 0)),
            ),
        })

    status = "acceptable" if metrics and not missing else "needs_attention"
    report = {
        "schema_version": STATISTICAL_EVAL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "metric_count": len(metrics),
        "missing_artifacts": missing,
        "metrics": metrics,
        "deltas": deltas,
        "claim_boundary": CLAIM_BOUNDARY,
        "reviewer_note": (
            "Report n, pass/fail counts, and intervals beside headline metrics. "
            "Small n, internal authorship, and benchmark contamination remain credibility risks."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = build_statistical_summary()
    print(json.dumps({
        "status": report["status"],
        "metric_count": report["metric_count"],
        "missing_artifacts": report["missing_artifacts"],
        "output": str(OUTPUT_PATH.relative_to(ROOT)),
    }, indent=2))
    return 0 if report["status"] in {"acceptable", "strong"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
