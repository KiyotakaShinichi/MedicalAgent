"""Statistical audit over exported synthetic row-level predictions.

This uses model outputs that already exist. It does not retrain models and does
not simulate feature-level robustness. The purpose is to make uncertainty,
paired comparisons, selective risk, and synthetic distribution sensitivity
visible without presenting any result as clinical evidence.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


INPUT_PATH = Path("Data/evals/models/latest_row_level_prediction_export.csv")
OUTPUT_PATH = Path("Data/evals/models/latest_synthetic_prediction_statistical_audit.json")
BOOTSTRAP_REPLICATES = 1000
PAIRED_BOOTSTRAP_REPLICATES = 4000
PERTURBATION_SEEDS = 30


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    positive = probability[y == 1]
    negative = probability[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return error


def _classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int | None]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "prevalence": round(float(y.mean()), 4),
        "auroc": round(_auc(y, probability), 4) if _auc(y, probability) is not None else None,
        "brier": round(float(np.mean((probability - y) ** 2)), 4),
        "ece_10_bin": round(_ece(y, probability), 4),
        "accuracy": round(float(np.mean(predicted == y)), 4),
    }


def _interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": round(float(np.median(array)), 4),
        "lower_95": round(float(np.quantile(array, 0.025)), 4),
        "upper_95": round(float(np.quantile(array, 0.975)), 4),
    }


def _bootstrap(y: np.ndarray, probability: np.ndarray, actual_reg: np.ndarray, predicted_reg: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(20260714)
    aucs: list[float] = []
    briers: list[float] = []
    eces: list[float] = []
    maes: list[float] = []
    n = len(y)
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, n, size=n)
        auc = _auc(y[sample], probability[sample])
        if auc is not None:
            aucs.append(auc)
        briers.append(float(np.mean((probability[sample] - y[sample]) ** 2)))
        eces.append(_ece(y[sample], probability[sample]))
        maes.append(float(np.mean(np.abs(predicted_reg[sample] - actual_reg[sample]))))
    return {
        "resampling_unit": "patient_row (one exported row per synthetic patient)",
        "replicates": BOOTSTRAP_REPLICATES,
        "classification_auroc_95_ci": _interval(aucs),
        "classification_brier_95_ci": _interval(briers),
        "classification_ece_95_ci": _interval(eces),
        "regression_mae_95_ci": _interval(maes),
    }


def _mcnemar(y: np.ndarray, champion_probability: np.ndarray, baseline_probability: np.ndarray) -> dict[str, Any]:
    champion_correct = (champion_probability >= 0.5).astype(int) == y
    baseline_correct = (baseline_probability >= 0.5).astype(int) == y
    baseline_only = int((baseline_correct & ~champion_correct).sum())
    champion_only = int((champion_correct & ~baseline_correct).sum())
    discordant = baseline_only + champion_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, champion_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    paired_delta = _paired_accuracy_delta_interval(
        champion_correct.astype(float),
        baseline_correct.astype(float),
    )
    return {
        "test": "exact_two_sided_mcnemar",
        "baseline_only_correct_n": baseline_only,
        "champion_only_correct_n": champion_only,
        "discordant_n": discordant,
        "p_value": round(float(p_value), 6),
        "significant_at_0_05": bool(p_value < 0.05),
        "champion_win_fraction_among_discordant": (
            round(champion_only / discordant, 4) if discordant else None
        ),
        "champion_win_fraction_wilson_95": _wilson_interval(
            champion_only,
            discordant,
        ),
        "paired_accuracy_delta": paired_delta,
        "practical_delta_required": 0.02,
        "superiority_proven": bool(
            p_value < 0.05
            and paired_delta["observed_delta"] >= 0.02
            and paired_delta["lower_95"] > 0
        ),
        "interpretation": "Synthetic paired prediction comparison only; significance does not imply clinical superiority.",
    }


def _paired_accuracy_delta_interval(
    champion_correct: np.ndarray,
    baseline_correct: np.ndarray,
) -> dict[str, Any]:
    differences = champion_correct - baseline_correct
    rng = np.random.default_rng(20260730)
    n = len(differences)
    draws = np.empty(PAIRED_BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(PAIRED_BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, n, size=n)
        draws[index] = float(np.mean(differences[sample]))
    return {
        "resampling_unit": "paired synthetic patient row",
        "replicates": PAIRED_BOOTSTRAP_REPLICATES,
        "observed_delta": round(float(np.mean(differences)), 6),
        "lower_95": round(float(np.quantile(draws, 0.025)), 6),
        "upper_95": round(float(np.quantile(draws, 0.975)), 6),
    }


def _selective_risk(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    confidence_margin = np.abs(probability - 0.5)
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.4):
        covered = confidence_margin >= threshold
        accuracy = float(np.mean((probability[covered] >= 0.5).astype(int) == y[covered])) if covered.any() else None
        rows.append({
            "minimum_probability_margin": threshold,
            "coverage": round(float(covered.mean()), 4),
            "abstention_rate": round(float(1.0 - covered.mean()), 4),
            "covered_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "covered_n": int(covered.sum()),
        })
    return rows


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + (z**2 / total)
    center = (proportion + (z**2 / (2 * total))) / denominator
    half_width = (
        z
        * math.sqrt((proportion * (1 - proportion) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    return {
        "lower_95": round(max(0.0, center - half_width), 4),
        "upper_95": round(min(1.0, center + half_width), 4),
        "width": round(2 * half_width, 4),
    }


def _threshold_sensitivity(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    baseline = probability >= 0.5
    rows: list[dict[str, Any]] = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        predicted = probability >= threshold
        correct = predicted == y
        rows.append(
            {
                "decision_threshold": threshold,
                "predicted_positive_rate": round(float(predicted.mean()), 4),
                "accuracy": round(float(correct.mean()), 4),
                "decision_flip_rate_vs_0_50": round(float(np.mean(predicted != baseline)), 4),
                "accuracy_wilson_95": _wilson_interval(int(correct.sum()), len(correct)),
            }
        )
    return rows


def _probability_stability(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_decision = probability >= 0.5
    for sigma in (0.01, 0.03, 0.05):
        flip_rates: list[float] = []
        briers: list[float] = []
        for seed in range(PERTURBATION_SEEDS):
            rng = np.random.default_rng(4000 + seed)
            perturbed = np.clip(
                probability + rng.normal(0.0, sigma, size=len(probability)),
                0.0,
                1.0,
            )
            flip_rates.append(float(np.mean((perturbed >= 0.5) != baseline_decision)))
            briers.append(float(np.mean((perturbed - y) ** 2)))
        rows.append(
            {
                "gaussian_probability_jitter_sigma": sigma,
                "decision_flip_rate_distribution": _interval(flip_rates),
                "brier_distribution": _interval(briers),
            }
        )
    return {
        "seed_count": PERTURBATION_SEEDS,
        "rows": rows,
        "note": (
            "This perturbs exported probabilities, not raw features or retrained models. "
            "It measures decision-boundary brittleness only."
        ),
    }


def _prevalence_reweighting(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260729)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    total = len(y)
    for target in (0.25, 0.50, 0.75):
        positive_n = max(1, round(total * target))
        negative_n = max(1, total - positive_n)
        sampled = np.concatenate(
            [
                rng.choice(positives, size=positive_n, replace=True),
                rng.choice(negatives, size=negative_n, replace=True),
            ]
        )
        rows.append(
            {
                "target_positive_prevalence": target,
                "resampling": "class-conditional bootstrap with replacement",
                **_classification_metrics(y[sampled], probability[sampled]),
            }
        )
    return rows


def _perturbations(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    label_noise_metrics: list[dict[str, float | int | None]] = []
    missing_outcome_metrics: list[dict[str, float | int | None]] = []
    for seed in range(PERTURBATION_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        noisy_y = y.copy()
        flips = rng.random(len(y)) < 0.10
        noisy_y[flips] = 1 - noisy_y[flips]
        label_noise_metrics.append(_classification_metrics(noisy_y, probability))

        kept = rng.random(len(y)) >= 0.20
        missing_outcome_metrics.append(_classification_metrics(y[kept], probability[kept]))

    return {
        "seed_count": PERTURBATION_SEEDS,
        "label_noise_10_percent": {
            "auroc_distribution": _interval([float(row["auroc"]) for row in label_noise_metrics if row["auroc"] is not None]),
            "brier_distribution": _interval([float(row["brier"]) for row in label_noise_metrics]),
            "note": "Outcome-label sensitivity only; features and model outputs were not recomputed.",
        },
        "outcome_missingness_20_percent": {
            "auroc_distribution": _interval([float(row["auroc"]) for row in missing_outcome_metrics if row["auroc"] is not None]),
            "brier_distribution": _interval([float(row["brier"]) for row in missing_outcome_metrics]),
            "note": "Random missing-outcome sensitivity only; this is not a missing-feature inference test.",
        },
    }


def build_report(input_path: Path = INPUT_PATH) -> dict[str, Any]:
    if not input_path.exists():
        return {
            "schema_version": "synthetic_prediction_statistical_audit_v1",
            "status": "needs_attention",
            "clinical_validation": False,
            "reason": f"missing row-level export: {input_path.as_posix()}",
        }
    rows = _read_rows(input_path)
    y = np.asarray([int(row["actual_label"]) for row in rows], dtype=int)
    probability = np.asarray([float(row["gradient_boosting_calibrated_probability"]) for row in rows])
    baseline_probability = np.asarray([float(row["logistic_regression_probability"]) for row in rows])
    actual_reg = np.asarray([float(row["actual_response_score_percent"]) for row in rows])
    predicted_reg = np.asarray([float(row["robust_response_ensemble_response_score_percent"]) for row in rows])

    subgroup_rows: list[dict[str, Any]] = []
    subtypes = sorted({row["molecular_subtype"] for row in rows})
    for subtype in subtypes:
        mask = np.asarray([row["molecular_subtype"] == subtype for row in rows])
        subgroup_metrics = _classification_metrics(y[mask], probability[mask])
        subgroup_correct = int(
            np.sum((probability[mask] >= 0.5).astype(int) == y[mask])
        )
        subgroup_rows.append(
            {
                "subgroup": subtype,
                **subgroup_metrics,
                "accuracy_wilson_95": _wilson_interval(
                    subgroup_correct,
                    int(mask.sum()),
                ),
                "minimum_n_warning": bool(mask.sum() < 30),
            }
        )

    regression_mae = float(np.mean(np.abs(predicted_reg - actual_reg)))
    classification = _classification_metrics(y, probability)
    paired = _mcnemar(y, probability, baseline_probability)
    small_subgroup_count = sum(row["minimum_n_warning"] for row in subgroup_rows)
    credibility_flags = [
        {
            "id": "synthetic_metric_saturation",
            "severity": "high",
            "triggered": bool(float(classification.get("auroc") or 0.0) >= 0.98),
            "detail": (
                "Near-perfect performance on simulator-built outcomes is a shortcut/"
                "homogeneity warning, not evidence of clinical strength."
            ),
        },
        {
            "id": "paired_champion_superiority_not_proven",
            "severity": "high",
            "triggered": not bool(paired.get("superiority_proven")),
            "detail": (
                "The nonlinear champion has not shown a statistically and "
                "practically credible paired accuracy lift over logistic regression."
            ),
        },
        {
            "id": "small_subgroup_intervals",
            "severity": "medium",
            "triggered": small_subgroup_count > 0,
            "detail": f"{small_subgroup_count} subgroup slices have fewer than 30 rows.",
        },
        {
            "id": "outcome_only_perturbations",
            "severity": "medium",
            "triggered": True,
            "detail": (
                "Current label/missing-outcome perturbations do not retrain the "
                "model or propagate feature-level measurement error."
            ),
        },
    ]
    active_flags = [item for item in credibility_flags if item["triggered"]]
    return {
        "schema_version": "synthetic_prediction_statistical_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if active_flags else "acceptable",
        "clinical_validation": False,
        "synthetic_only": True,
        "healthcare_production_ready": False,
        "input_artifact": input_path.as_posix(),
        "total_n": len(rows),
        "patient_count": len({row["patient_id"] for row in rows}),
        "classification_model": "gradient_boosting_calibrated",
        "classification_metrics": classification,
        "regression_model": "robust_response_ensemble",
        "regression_metrics": {"mae": round(regression_mae, 4)},
        "patient_level_bootstrap": _bootstrap(y, probability, actual_reg, predicted_reg),
        "paired_baseline_comparison": paired,
        "selective_risk_curve": _selective_risk(y, probability),
        "decision_threshold_sensitivity": _threshold_sensitivity(y, probability),
        "exported_probability_stability": _probability_stability(y, probability),
        "prevalence_reweighting_sensitivity": _prevalence_reweighting(y, probability),
        "subgroup_slices": subgroup_rows,
        "controlled_outcome_perturbations": _perturbations(y, probability),
        "credibility_flags": active_flags,
        "headline": {
            "champion_superiority_over_logistic_proven": paired[
                "superiority_proven"
            ],
            "paired_accuracy_delta": paired["paired_accuracy_delta"],
            "synthetic_auroc": classification.get("auroc"),
            "interpretation": (
                "Near-perfect synthetic metrics remain simulator-bounded. "
                "Model complexity is not justified by paired superiority."
            ),
        },
        "promotion_decision": "hold_synthetic_only",
        "limitations": [
            "The export contains synthetic patients and simulator-built outcomes.",
            "Bootstrap intervals quantify sampling variability inside this synthetic export only.",
            "Label and outcome-missingness perturbations do not recompute model predictions from perturbed features.",
            "Probability jitter is a decision-boundary stress test and does not simulate realistic covariate shift.",
            "Prevalence reweighting is synthetic resampling, not external transportability evidence.",
            "Subgroup slices are engineering checks and do not establish fairness in real populations.",
        ],
        "claim_boundary": (
            "Statistical engineering audit over synthetic row-level predictions only. It is not "
            "clinical evidence, external validation, a real-population fairness result, or proof "
            "that model performance generalises to patients."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "PAIRED_BOOTSTRAP_REPLICATES",
    "OUTPUT_PATH",
    "PERTURBATION_SEEDS",
    "build_report",
    "write_report",
]
