"""Statistical evidence dossier for synthetic-only ML artifacts.

This module adds statistical framing to existing model artifacts.  It prefers
raw paired predictions when available, but most current artifacts are summary
level, so the report explicitly labels those sections as descriptive or
approximate rather than pretending to run definitive tests.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.statistical_eval import (
    CLAIM_BOUNDARY as STAT_CLAIM_BOUNDARY,
    mean_interval,
    two_proportion_delta,
    wilson_interval,
)


ML_STATISTICAL_TESTS_VERSION = "ml_statistical_evidence_v1_2026_05"
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_ml_statistical_evidence.json")
DEFAULT_ARTIFACTS = {
    "per_head_calibration": Path("Data/evals/models/latest_per_head_calibration.json"),
    "modality_robustness": Path("Data/evals/models/latest_modality_robustness_comparison.json"),
    "deep_learning_candidates": Path("Data/evals/models/latest_deep_learning_candidate_benchmark.json"),
    "hybrid_subgroups": Path("Data/evals/models/latest_hybrid_subgroup_metrics.json"),
    "patient_temporal_cv": Path("Data/evals/models/latest_patient_temporal_cv.json"),
}

CLAIM_BOUNDARY = (
    "This is a synthetic-only statistical evidence dossier. Intervals and "
    "comparison tests describe internal simulator-built artifacts and do not "
    "establish clinical validity, real patient calibration, treatment utility, "
    "or real-world safety."
)


def build_ml_statistical_evidence(
    *,
    artifacts: dict[str, Path] | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    paths = artifacts or DEFAULT_ARTIFACTS
    loaded = {name: _load(path) for name, path in paths.items()}
    missing = [str(paths[name]) for name, payload in loaded.items() if payload is None]

    calibration = _calibration_section(loaded.get("per_head_calibration") or {})
    robustness = _modality_comparison_section(loaded.get("modality_robustness") or {})
    dl = _deep_learning_section(loaded.get("deep_learning_candidates") or {})
    subgroup = _subgroup_section(loaded.get("hybrid_subgroups") or {})
    temporal = _temporal_cv_section(loaded.get("patient_temporal_cv") or {})

    warning_count = sum(
        1
        for section in [calibration, robustness, dl, subgroup, temporal]
        if section.get("status") in {"needs_attention", "insufficient_raw_data"}
    )
    report = {
        "schema_version": ML_STATISTICAL_TESTS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if not missing else "needs_attention",
        "missing_artifacts": missing,
        "warning_count": warning_count,
        "sections": {
            "per_head_calibration": calibration,
            "modality_robustness_comparison": robustness,
            "deep_learning_candidate_comparison": dl,
            "subgroup_statistical_screen": subgroup,
            "patient_temporal_cv": temporal,
        },
        "recommended_next_raw_prediction_exports": [
            "per-row classification labels/probabilities for champion and robust heads",
            "per-row regression predictions/residuals by model variant",
            "per-row toxicity labels/probabilities for legacy and soft target heads",
            "subgroup membership per prediction row",
            "abstention decision per row and model head",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _calibration_section(payload: dict[str, Any]) -> dict[str, Any]:
    heads = payload.get("heads") or {}
    out: dict[str, Any] = {
        "status": "acceptable" if heads else "needs_attention",
        "method": "reliability_bin_wilson_intervals",
        "heads": {},
        "claim_boundary": STAT_CLAIM_BOUNDARY,
    }
    for head_name, head in heads.items():
        bins = []
        for item in head.get("reliability_bins") or []:
            n = int(item.get("count") or 0)
            empirical = item.get("empirical_rate")
            successes = int(round(float(empirical) * n)) if empirical is not None else 0
            interval = wilson_interval(successes, n)
            bins.append({
                **item,
                "empirical_rate_ci": {
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "method": interval["method"],
                    "total_n": interval["total_n"],
                },
                "bin_absolute_gap": (
                    round(abs(float(item["mean_probability"]) - float(empirical)), 6)
                    if item.get("mean_probability") is not None and empirical is not None
                    else None
                ),
                "small_n_flag": n < 30,
            })
        out["heads"][head_name] = {
            "headline_metrics": {k: v for k, v in head.items() if k != "reliability_bins"},
            "reliability_bins_with_ci": bins,
            "small_bin_count": sum(1 for row in bins if row["small_n_flag"]),
        }
    return out


def _modality_comparison_section(payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = payload.get("scenarios") or []
    rows = []
    wins = losses = ties = 0
    for scenario in scenarios:
        n = int(scenario.get("rows_evaluated") or 0)
        champion_acc = float(((scenario.get("force_score") or {}).get("champion") or {}).get("accuracy", 0.0))
        robust_acc = float(((scenario.get("force_score") or {}).get("robust") or {}).get("accuracy", 0.0))
        champion_success = int(round(champion_acc * n))
        robust_success = int(round(robust_acc * n))
        delta = robust_acc - champion_acc
        if abs(delta) < 1e-9:
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
        rows.append({
            "scenario": scenario.get("scenario"),
            "rows_evaluated": n,
            "champion_accuracy": champion_acc,
            "robust_accuracy": robust_acc,
            "accuracy_delta_robust_minus_champion": round(delta, 6),
            "approx_two_proportion_delta_ci": two_proportion_delta(
                baseline_successes=champion_success,
                baseline_total=n,
                candidate_successes=robust_success,
                candidate_total=n,
            ),
            "raw_paired_predictions_available": False,
            "paired_test_limitation": "McNemar/paired bootstrap requires row-level paired predictions; current artifact is summary-level.",
        })
    sign = _exact_sign_test_p_value(wins, losses)
    return {
        "status": "acceptable" if scenarios else "needs_attention",
        "method": "scenario_level_sign_test_plus_approx_two_proportion_ci",
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "scenario_sign_test_p_value": sign,
        "raw_prediction_availability": False,
        "rows": rows,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _deep_learning_section(payload: dict[str, Any]) -> dict[str, Any]:
    models = payload.get("models") or {}
    candidates = []
    for feature_set, variants in models.items():
        for model_name, metrics in (variants or {}).items():
            cls = metrics.get("classification") or {}
            reg = metrics.get("regression") or {}
            candidates.append({
                "feature_set": feature_set,
                "model": model_name,
                "classification_auroc": cls.get("auroc"),
                "classification_brier": cls.get("brier"),
                "classification_accuracy": cls.get("accuracy"),
                "regression_mae_percent": reg.get("mae_percent"),
                "regression_r2": reg.get("r2"),
            })
    best_classifier = max(candidates, key=lambda row: row.get("classification_auroc") or -1, default=None)
    best_regressor = min(candidates, key=lambda row: row.get("regression_mae_percent") or float("inf"), default=None)
    return {
        "status": "acceptable" if candidates else "needs_attention",
        "candidate_count": len(candidates),
        "best_classifier_by_auroc": best_classifier,
        "best_regressor_by_mae": best_regressor,
        "comparison_limitation": (
            "This ranks summary metrics. Paired bootstrap over per-patient predictions "
            "is still needed before making stronger model-comparison claims."
        ),
        "promotion_boundary": "No DL candidate may be promoted beyond synthetic monitor-only evidence.",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _subgroup_section(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload.get("overall") or {}
    groups = []
    for key in ["by_molecular_subtype", "by_stage"]:
        for row in payload.get(key) or []:
            n = int(row.get("n") or 0)
            acc = row.get("classification_accuracy")
            successes = int(round(float(acc) * n)) if acc is not None else 0
            interval = wilson_interval(successes, n)
            overall_acc = overall.get("classification_accuracy")
            groups.append({
                "axis": key.replace("by_", ""),
                "group": row.get("group"),
                "n": n,
                "classification_accuracy": acc,
                "classification_accuracy_ci": {
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "method": interval["method"],
                },
                "gap_vs_overall": (
                    round(float(acc) - float(overall_acc), 6)
                    if acc is not None and overall_acc is not None
                    else None
                ),
                "small_n_flag": n < 100,
            })
    max_abs_gap = max((abs(row["gap_vs_overall"]) for row in groups if row["gap_vs_overall"] is not None), default=0.0)
    return {
        "status": "acceptable" if groups else "needs_attention",
        "group_count": len(groups),
        "small_n_group_count": sum(1 for row in groups if row["small_n_flag"]),
        "max_abs_classification_accuracy_gap_vs_overall": round(max_abs_gap, 6),
        "groups": groups,
        "claim_boundary": payload.get("claim_boundary") or CLAIM_BOUNDARY,
    }


def _temporal_cv_section(payload: dict[str, Any]) -> dict[str, Any]:
    folds = ((payload.get("patient_level_temporal_cv") or {}).get("folds") or [])
    aucs = [float(row["roc_auc"]) for row in folds if row.get("roc_auc") is not None]
    briers = [float(row["brier"]) for row in folds if row.get("brier") is not None]
    return {
        "status": "acceptable" if aucs else "needs_attention",
        "roc_auc_interval": mean_interval(aucs),
        "brier_interval": mean_interval(briers),
        "patient_overlap_pairs": (payload.get("patient_level_temporal_cv") or {}).get("patient_overlap_pairs"),
        "temporal_violations": (payload.get("patient_level_temporal_cv") or {}).get("temporal_violations"),
        "limitation": "Fold-level interval only; raw patient-level prediction bootstrap is a future upgrade.",
        "claim_boundary": payload.get("claim_boundary") or CLAIM_BOUNDARY,
    }


def _exact_sign_test_p_value(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n <= 0:
        return None
    k = min(wins, losses)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return round(min(1.0, 2 * prob), 6)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_ARTIFACTS",
    "DEFAULT_OUTPUT_PATH",
    "ML_STATISTICAL_TESTS_VERSION",
    "build_ml_statistical_evidence",
]
