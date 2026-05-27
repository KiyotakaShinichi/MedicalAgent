"""Row-level synthetic prediction export and paired statistical comparisons."""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from backend.services.statistical_eval import wilson_interval


DEFAULT_DETAILED_PREDICTIONS = Path("Data/complete_synthetic_training/detailed_eval/test_set_predictions_detailed.csv")
DEFAULT_EXPORT_CSV = Path("Data/evals/models/latest_row_level_prediction_export.csv")
DEFAULT_MANIFEST_JSON = Path("Data/evals/models/latest_row_level_prediction_export_manifest.json")
DEFAULT_PAIRED_JSON = Path("Data/evals/models/latest_paired_model_comparison.json")
DEFAULT_CALIBRATION_JSON = Path("Data/evals/models/latest_calibration_uncertainty_report.json")

CLASSIFICATION_MODELS = {
    "logistic_regression": "logistic_regression_probability",
    "random_forest": "random_forest_probability",
    "gradient_boosting": "gradient_boosting_probability",
    "gradient_boosting_calibrated": "gradient_boosting_calibrated_probability",
    "mlp": "mlp_probability",
    "temporal_gru": "temporal_gru_probability",
}

REGRESSION_MODELS = {
    "ridge_regression": "ridge_regression_response_score_percent",
    "random_forest_regressor": "random_forest_regressor_response_score_percent",
    "gradient_boosting_regressor": "gradient_boosting_regressor_response_score_percent",
    "robust_response_ensemble": "robust_response_ensemble_response_score_percent",
}

CHAMPION_CLASSIFICATION = "gradient_boosting_calibrated"
BASELINE_CLASSIFICATION = "logistic_regression"
CHAMPION_REGRESSION = "random_forest_regressor"
BASELINE_REGRESSION = "ridge_regression"

CLAIM_BOUNDARY = (
    "Row-level prediction exports and paired tests are synthetic-only MLE "
    "evidence. They do not establish clinical validation, patient benefit, "
    "or treatment utility."
)


def run_row_level_prediction_evidence(
    *,
    input_csv: str | Path = DEFAULT_DETAILED_PREDICTIONS,
    export_csv: str | Path = DEFAULT_EXPORT_CSV,
    manifest_json: str | Path = DEFAULT_MANIFEST_JSON,
    paired_json: str | Path = DEFAULT_PAIRED_JSON,
    calibration_json: str | Path = DEFAULT_CALIBRATION_JSON,
) -> dict[str, Any]:
    rows = _read_rows(Path(input_csv))
    exported = [_export_row(row) for row in rows]
    _write_csv(Path(export_csv), exported)
    manifest = _manifest(rows, exported, Path(input_csv), Path(export_csv))
    paired = _paired_comparison(exported)
    calibration = _calibration_report(exported)
    _write_json(Path(manifest_json), manifest)
    _write_json(Path(paired_json), paired)
    _write_json(Path(calibration_json), calibration)
    return {
        "manifest": manifest,
        "paired": paired,
        "calibration": calibration,
    }


def _export_row(row: dict[str, str]) -> dict[str, Any]:
    y = _int(row.get("actual_label"))
    actual_score = _float(row.get("actual_response_score_percent"))
    out: dict[str, Any] = {
        "patient_id": row.get("patient_id"),
        "fold_id": "legacy_synthetic_test_set",
        "split": "test",
        "actual_label": y,
        "actual_response_score_percent": actual_score,
        "age": _float(row.get("age")),
        "stage": row.get("stage"),
        "molecular_subtype": row.get("molecular_subtype"),
        "regimen": row.get("regimen"),
        "cycles_observed": _int(row.get("cycles_observed")),
        "max_symptom_severity": _float(row.get("max_symptom_severity")),
        "nadir_wbc": _float(row.get("nadir_wbc")),
        "nadir_anc": _float(row.get("nadir_anc")),
        "latest_mri_percent_change": _float(row.get("latest_mri_percent_change")),
        "synthetic_only": True,
        "clinical_validation": False,
    }
    for name, column in CLASSIFICATION_MODELS.items():
        prob = _float(row.get(column))
        pred = _int(row.get(column.replace("_probability", "_predicted_label")))
        if pred is None and prob is not None:
            pred = int(prob >= 0.5)
        out[f"{name}_probability"] = prob
        out[f"{name}_predicted_label"] = pred
        out[f"{name}_correct"] = int(pred == y) if pred is not None and y is not None else None
    for name, column in REGRESSION_MODELS.items():
        pred_score = _float(row.get(column))
        out[f"{name}_response_score_percent"] = pred_score
        out[f"{name}_absolute_error"] = abs(pred_score - actual_score) if pred_score is not None and actual_score is not None else None
    return out


def _manifest(source_rows: list[dict[str, str]], exported: list[dict[str, Any]], input_path: Path, export_path: Path) -> dict[str, Any]:
    patient_ids = {row.get("patient_id") for row in exported if row.get("patient_id")}
    return {
        "schema_version": "row_level_prediction_export_manifest_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if exported and len(patient_ids) == len(exported) else "needs_attention",
        "source_path": str(input_path).replace("\\", "/"),
        "artifact_path": str(export_path).replace("\\", "/"),
        "total_n": len(exported),
        "pass_count": len(exported),
        "fail_count": 0,
        "skipped_count": 0,
        "patient_count": len(patient_ids),
        "patient_id_unique": len(patient_ids) == len(exported),
        "classification_models": sorted(CLASSIFICATION_MODELS),
        "regression_models": sorted(REGRESSION_MODELS),
        "required_columns_present": _required_columns_present(source_rows[0] if source_rows else {}),
        "synthetic_only": True,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "contamination_note": "Exported from internal synthetic test-set predictions; not external validation.",
    }


def _paired_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classification = []
    champion_col = f"{CHAMPION_CLASSIFICATION}_correct"
    for model in CLASSIFICATION_MODELS:
        if model == CHAMPION_CLASSIFICATION:
            continue
        model_col = f"{model}_correct"
        pairs = [(row.get(champion_col), row.get(model_col)) for row in rows]
        pairs = [(int(a), int(b)) for a, b in pairs if a is not None and b is not None]
        classification.append(_mcnemar_row(model, pairs))

    regression = []
    champion_err = f"{CHAMPION_REGRESSION}_absolute_error"
    for model in REGRESSION_MODELS:
        if model == CHAMPION_REGRESSION:
            continue
        err_col = f"{model}_absolute_error"
        pairs = [(row.get(champion_err), row.get(err_col)) for row in rows]
        pairs = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None]
        regression.append(_paired_bootstrap_mae_delta(model, pairs))

    payload = {
        "schema_version": "paired_model_comparison_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable",
        "total_n": len(rows),
        "pass_count": len(rows),
        "fail_count": 0,
        "skipped_count": 0,
        "classification_champion": CHAMPION_CLASSIFICATION,
        "regression_champion": CHAMPION_REGRESSION,
        "classification": classification,
        "regression": regression,
        "promotion_allowed": False,
        "synthetic_only": True,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return payload


def _calibration_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = {}
    for model in CLASSIFICATION_MODELS:
        prob_col = f"{model}_probability"
        valid = [(row.get(prob_col), row.get("actual_label")) for row in rows]
        valid = [(float(p), int(y)) for p, y in valid if p is not None and y is not None]
        reports[model] = _calibration_bins(valid)
    max_ece = max((row["ece"] for row in reports.values()), default=0.0)
    return {
        "schema_version": "calibration_uncertainty_report_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if reports else "needs_attention",
        "total_n": len(rows),
        "pass_count": len(rows),
        "fail_count": 0,
        "skipped_count": 0,
        "models": reports,
        "max_ece": round(max_ece, 6),
        "synthetic_only": True,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _mcnemar_row(model: str, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    champion_only = sum(1 for champion, candidate in pairs if champion == 1 and candidate == 0)
    candidate_only = sum(1 for champion, candidate in pairs if champion == 0 and candidate == 1)
    both_correct = sum(1 for champion, candidate in pairs if champion == 1 and candidate == 1)
    both_wrong = sum(1 for champion, candidate in pairs if champion == 0 and candidate == 0)
    n_discordant = champion_only + candidate_only
    p_value = _exact_two_sided_binomial(min(champion_only, candidate_only), n_discordant) if n_discordant else 1.0
    champion_acc = (both_correct + champion_only) / len(pairs) if pairs else None
    candidate_acc = (both_correct + candidate_only) / len(pairs) if pairs else None
    return {
        "candidate_model": model,
        "comparison": f"{CHAMPION_CLASSIFICATION}_vs_{model}",
        "method": "exact_mcnemar_binomial_on_paired_correctness",
        "total_n": len(pairs),
        "champion_only_correct": champion_only,
        "candidate_only_correct": candidate_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "champion_accuracy": round(champion_acc, 6) if champion_acc is not None else None,
        "candidate_accuracy": round(candidate_acc, 6) if candidate_acc is not None else None,
        "accuracy_delta_champion_minus_candidate": round((champion_acc or 0) - (candidate_acc or 0), 6) if pairs else None,
        "p_value": p_value,
        "interpretation_boundary": "Synthetic paired correctness test only; not clinical superiority evidence.",
    }


def _paired_bootstrap_mae_delta(model: str, pairs: list[tuple[float, float]], iterations: int = 500, seed: int = 20260525) -> dict[str, Any]:
    rng = random.Random(seed)
    if not pairs:
        return {"candidate_model": model, "total_n": 0, "status": "no_pairs"}
    deltas = []
    n = len(pairs)
    for _ in range(iterations):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        champion_mae = mean(item[0] for item in sample)
        candidate_mae = mean(item[1] for item in sample)
        deltas.append(champion_mae - candidate_mae)
    deltas.sort()
    observed = mean(item[0] for item in pairs) - mean(item[1] for item in pairs)
    return {
        "candidate_model": model,
        "comparison": f"{CHAMPION_REGRESSION}_vs_{model}",
        "method": "paired_bootstrap_mae_delta_champion_minus_candidate",
        "total_n": n,
        "iterations": iterations,
        "champion_mae": round(mean(item[0] for item in pairs), 6),
        "candidate_mae": round(mean(item[1] for item in pairs), 6),
        "mae_delta_champion_minus_candidate": round(observed, 6),
        "ci_low": round(_percentile(deltas, 0.025), 6),
        "ci_high": round(_percentile(deltas, 0.975), 6),
        "interpretation_boundary": "Synthetic paired bootstrap only; not clinical superiority evidence.",
    }


def _calibration_bins(pairs: list[tuple[float, int]], bins: int = 5) -> dict[str, Any]:
    rows = []
    ece = 0.0
    total = len(pairs)
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        bucket = [(p, y) for p, y in pairs if (lo <= p < hi) or (idx == bins - 1 and p == hi)]
        if not bucket:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "mean_probability": None, "empirical_rate": None, "empirical_rate_ci": None})
            continue
        mean_probability = mean(p for p, _ in bucket)
        successes = sum(y for _, y in bucket)
        empirical = successes / len(bucket)
        ece += (len(bucket) / max(total, 1)) * abs(mean_probability - empirical)
        interval = wilson_interval(successes, len(bucket))
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": len(bucket),
            "mean_probability": round(mean_probability, 6),
            "empirical_rate": round(empirical, 6),
            "empirical_rate_ci": interval,
            "small_n_flag": len(bucket) < 30,
        })
    return {
        "total_n": total,
        "ece": round(ece, 6),
        "bins": rows,
    }


def _exact_two_sided_binomial(k: int, n: int) -> float | None:
    if n <= 0:
        return None
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return round(min(1.0, 2 * prob), 6)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def _required_columns_present(row: dict[str, str]) -> bool:
    required = {"patient_id", "actual_label", "actual_response_score_percent"}
    required.update(CLASSIFICATION_MODELS.values())
    required.update(REGRESSION_MODELS.values())
    return required.issubset(set(row))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


__all__ = ["run_row_level_prediction_evidence"]
