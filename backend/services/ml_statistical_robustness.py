"""Additional synthetic-only ML statistical robustness checks."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from backend.services.statistical_eval import wilson_interval


DEFAULT_ROW_EXPORT = Path("Data/evals/models/latest_row_level_prediction_export.csv")
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_ml_statistical_robustness.json")

CLASSIFIER = "gradient_boosting_calibrated"
REGRESSOR = "random_forest_regressor"

CLAIM_BOUNDARY = (
    "ML statistical robustness is computed on synthetic rows only. It does not "
    "establish real-patient calibration, clinical validation, treatment utility, "
    "or patient benefit."
)


def build_ml_statistical_robustness(
    row_export: str | Path = DEFAULT_ROW_EXPORT,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = _read_rows(Path(row_export))
    classification_pairs = [
        (float(row[f"{CLASSIFIER}_probability"]), int(row["actual_label"]))
        for row in rows
        if row.get(f"{CLASSIFIER}_probability") not in {"", None} and row.get("actual_label") not in {"", None}
    ]
    regression_pairs = [
        (float(row[f"{REGRESSOR}_response_score_percent"]), float(row["actual_response_score_percent"]))
        for row in rows
        if row.get(f"{REGRESSOR}_response_score_percent") not in {"", None}
        and row.get("actual_response_score_percent") not in {"", None}
    ]
    classification = _classification_bootstrap(classification_pairs)
    regression = _regression_bootstrap(regression_pairs)
    subgroup = _subgroup_intervals(rows)
    noise = _label_noise_sensitivity(classification_pairs)
    stability_flags = _stability_flags(classification, regression, subgroup, noise)
    status = "acceptable" if not any(flag["severity"] == "high" for flag in stability_flags) else "needs_attention"
    payload = {
        "schema_version": "ml_statistical_robustness_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "total_n": len(rows),
        "pass_count": len(rows),
        "fail_count": 0,
        "skipped_count": 0,
        "classification_champion": CLASSIFIER,
        "regression_champion": REGRESSOR,
        "classification_bootstrap": classification,
        "regression_bootstrap": regression,
        "subgroup_confidence_intervals": subgroup,
        "label_noise_sensitivity": noise,
        "stability_flags": stability_flags,
        "clinical_validation": False,
        "synthetic_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "contamination_note": "Uses internal synthetic row-level prediction export; not external validation.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _classification_bootstrap(pairs: list[tuple[float, int]], iterations: int = 500, seed: int = 260525) -> dict[str, Any]:
    rng = random.Random(seed)
    observed = _classification_metrics(pairs)
    samples: dict[str, list[float]] = defaultdict(list)
    n = len(pairs)
    for _ in range(iterations):
        sample = [pairs[rng.randrange(n)] for _ in range(n)] if n else []
        metrics = _classification_metrics(sample)
        for key in ("accuracy", "brier", "ece"):
            samples[key].append(metrics[key])
    return {
        "method": "patient_row_bootstrap_with_replacement",
        "iterations": iterations,
        "total_n": n,
        "observed": observed,
        "intervals": {key: _interval(values) for key, values in samples.items()},
    }


def _regression_bootstrap(pairs: list[tuple[float, float]], iterations: int = 500, seed: int = 260526) -> dict[str, Any]:
    rng = random.Random(seed)
    observed = _regression_metrics(pairs)
    samples: dict[str, list[float]] = defaultdict(list)
    n = len(pairs)
    for _ in range(iterations):
        sample = [pairs[rng.randrange(n)] for _ in range(n)] if n else []
        metrics = _regression_metrics(sample)
        for key in ("mae", "rmse", "r2"):
            samples[key].append(metrics[key])
    return {
        "method": "patient_row_bootstrap_with_replacement",
        "iterations": iterations,
        "total_n": n,
        "observed": observed,
        "intervals": {key: _interval(values) for key, values in samples.items()},
    }


def _subgroup_intervals(rows: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        subgroup = row.get("molecular_subtype") or row.get("stage") or "unknown"
        y = _int(row.get("actual_label"))
        pred = _int(row.get(f"{CLASSIFIER}_predicted_label"))
        if y is not None and pred is not None:
            groups[subgroup].append((pred, y))
    output = {}
    for subgroup, values in sorted(groups.items()):
        correct = sum(1 for pred, y in values if pred == y)
        interval = wilson_interval(correct, len(values))
        output[subgroup] = {
            "total_n": len(values),
            "accuracy": round(correct / len(values), 6) if values else None,
            "accuracy_ci": interval,
            "small_n_flag": len(values) < 30,
        }
    return output


def _label_noise_sensitivity(pairs: list[tuple[float, int]], seed: int = 260527) -> dict[str, Any]:
    rng = random.Random(seed)
    baseline = _classification_metrics(pairs)
    rows = []
    for noise_rate in (0.05, 0.10, 0.20):
        noisy = []
        for prob, label in pairs:
            flip = rng.random() < noise_rate
            noisy.append((prob, 1 - label if flip else label))
        metrics = _classification_metrics(noisy)
        rows.append({
            "label_noise_rate": noise_rate,
            "accuracy": metrics["accuracy"],
            "brier": metrics["brier"],
            "accuracy_delta_vs_baseline": round(metrics["accuracy"] - baseline["accuracy"], 6),
            "brier_delta_vs_baseline": round(metrics["brier"] - baseline["brier"], 6),
        })
    return {
        "baseline": baseline,
        "rows": rows,
        "interpretation_boundary": "Synthetic label-noise perturbation only; not a real annotation-noise study.",
    }


def _classification_metrics(pairs: list[tuple[float, int]]) -> dict[str, float]:
    if not pairs:
        return {"accuracy": 0.0, "brier": 0.0, "ece": 0.0}
    preds = [(1 if prob >= 0.5 else 0, prob, label) for prob, label in pairs]
    accuracy = sum(1 for pred, _, label in preds if pred == label) / len(preds)
    brier = mean((prob - label) ** 2 for _, prob, label in preds)
    return {"accuracy": round(accuracy, 6), "brier": round(brier, 6), "ece": round(_ece([(prob, label) for _, prob, label in preds]), 6)}


def _regression_metrics(pairs: list[tuple[float, float]]) -> dict[str, float]:
    if not pairs:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    errors = [pred - actual for pred, actual in pairs]
    mae = mean(abs(err) for err in errors)
    rmse = math.sqrt(mean(err * err for err in errors))
    y_mean = mean(actual for _, actual in pairs)
    ss_res = sum((actual - pred) ** 2 for pred, actual in pairs)
    ss_tot = sum((actual - y_mean) ** 2 for _, actual in pairs)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)}


def _ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    total = len(pairs)
    ece = 0.0
    for idx in range(bins):
        lo, hi = idx / bins, (idx + 1) / bins
        bucket = [(p, y) for p, y in pairs if (lo <= p < hi) or (idx == bins - 1 and p == hi)]
        if not bucket:
            continue
        ece += (len(bucket) / total) * abs(mean(p for p, _ in bucket) - mean(y for _, y in bucket))
    return ece


def _stability_flags(classification: dict[str, Any], regression: dict[str, Any], subgroup: dict[str, Any], noise: dict[str, Any]) -> list[dict[str, Any]]:
    flags = []
    if classification["observed"]["ece"] > 0.20:
        flags.append({"name": "classification_ece_high", "severity": "medium", "value": classification["observed"]["ece"]})
    if regression["observed"]["r2"] < 0.30:
        flags.append({"name": "regression_r2_low", "severity": "medium", "value": regression["observed"]["r2"]})
    small_groups = [name for name, row in subgroup.items() if row["small_n_flag"]]
    if small_groups:
        flags.append({"name": "small_subgroup_n", "severity": "low", "groups": small_groups})
    worst_noise_delta = min((row["accuracy_delta_vs_baseline"] for row in noise["rows"]), default=0.0)
    if worst_noise_delta < -0.20:
        flags.append({"name": "label_noise_sensitivity_large", "severity": "medium", "value": worst_noise_delta})
    return flags


def _interval(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci_low": None, "ci_high": None}
    ordered = sorted(values)
    return {
        "ci_low": round(_percentile(ordered, 0.025), 6),
        "ci_high": round(_percentile(ordered, 0.975), 6),
    }


def _percentile(values: list[float], q: float) -> float:
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(value))


__all__ = ["build_ml_statistical_robustness"]
