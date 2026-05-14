import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.calibration_eval import run_calibration_eval  # noqa: E402

DEFAULT_SYNTHETIC_METRICS = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
DEFAULT_BASELINE_METRICS = "Data/breastdcedl_spy1_baseline_metrics.json"
DEFAULT_CNN_METRICS = "Data/breastdcedl_spy1_cnn_metrics.json"
DEFAULT_CALIBRATION_REPORT = "Data/mle_monitoring/calibration_eval_report.json"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_model_benchmark.json"
DEFAULT_CSV_PATH = "Data/evals/models/latest_model_benchmark.csv"


def _load_json(path: str) -> dict | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _safe_round(value, places=3):
    if value is None:
        return None
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None


def _best_model_key(models: dict, preferred: str | None = None) -> str | None:
    if preferred and preferred in models:
        return preferred
    best_key = None
    best_score = -1.0
    for key, metrics in models.items():
        score = metrics.get("patient_level_roc_auc") or metrics.get("roc_auc")
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_key = key
    return best_key


def _calibration_summary(model: dict) -> dict:
    calibration = model.get("patient_level_calibration") or model.get("calibration") or {}
    before = calibration.get("before_temperature_scaling") or {}
    after = calibration.get("after_temperature_scaling") or {}
    return {
        "ece_before": _safe_round(before.get("ece")),
        "ece_after": _safe_round(after.get("ece")),
        "brier_before": _safe_round(before.get("brier_score")),
        "brier_after": _safe_round(after.get("brier_score")),
        "temperature": after.get("temperature"),
    }


def _classification_snapshot(models: dict, model_key: str | None, label: str) -> dict | None:
    if not models or not model_key or model_key not in models:
        return None
    model = models[model_key]
    return {
        "label": label,
        "model": model_key,
        "roc_auc": _safe_round(model.get("patient_level_roc_auc") or model.get("roc_auc")),
        "auprc": _safe_round(model.get("patient_level_average_precision") or model.get("average_precision")),
        "f1": _safe_round(model.get("patient_level_f1") or model.get("f1")),
        "recall": _safe_round(model.get("patient_level_sensitivity") or model.get("sensitivity")),
        "precision": _safe_round(model.get("patient_level_precision") or model.get("precision")),
        "brier": _safe_round(model.get("patient_level_brier_score") or model.get("brier_score")),
        **_calibration_summary(model),
    }


def _regression_snapshot(regression_block: dict) -> dict | None:
    if not regression_block:
        return None
    models = regression_block.get("models") or {}
    best_key = regression_block.get("best_model_by_patient_level_mae") or regression_block.get("best_response_regressor_by_patient_level_mae")
    if best_key and best_key not in models:
        best_key = None
    if not best_key and models:
        best_key = min(models, key=lambda key: models[key].get("patient_level_mae") or 1e9)
    if not best_key:
        return None
    metrics = models[best_key]
    return {
        "model": best_key,
        "mae": _safe_round(metrics.get("patient_level_mae") or metrics.get("mae")),
        "rmse": _safe_round(metrics.get("patient_level_rmse") or metrics.get("rmse")),
        "r2": _safe_round(metrics.get("patient_level_r2") or metrics.get("r2")),
        "target": regression_block.get("target") or metrics.get("target"),
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["benchmark", "model", "metric", "value", "dataset", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize model baseline vs champion metrics for the benchmark ladder.")
    parser.add_argument("--synthetic-metrics", default=DEFAULT_SYNTHETIC_METRICS)
    parser.add_argument("--baseline-metrics", default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--cnn-metrics", default=DEFAULT_CNN_METRICS)
    parser.add_argument("--calibration-report", default=DEFAULT_CALIBRATION_REPORT)
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    synthetic = _load_json(args.synthetic_metrics)
    baseline = _load_json(args.baseline_metrics)
    cnn_metrics = _load_json(args.cnn_metrics)

    if args.run_calibration:
        run_calibration_eval(output_path=args.calibration_report)
    calibration_report = _load_json(args.calibration_report)

    classification = []
    regression = None
    if synthetic:
        models = synthetic.get("models") or {}
        best_key = synthetic.get("best_model_by_patient_level_roc_auc")
        best_key = _best_model_key(models, preferred=best_key)
        champ = _classification_snapshot(models, best_key, label="synthetic_champion")
        if champ:
            classification.append(champ)
        regression = _regression_snapshot(synthetic.get("response_regression") or {})

    external_baselines = []
    if baseline:
        for model_name, metrics in (baseline.get("models") or {}).items():
            external_baselines.append({
                "label": "breastdcedl_baseline",
                "model": model_name,
                "roc_auc": _safe_round(metrics.get("roc_auc")),
                "accuracy": _safe_round(metrics.get("accuracy")),
                "balanced_accuracy": _safe_round(metrics.get("balanced_accuracy")),
            })

    cnn_summary = None
    if cnn_metrics:
        final_val = cnn_metrics.get("final_validation") or {}
        cnn_summary = {
            "label": "breastdcedl_cnn",
            "roc_auc": _safe_round(final_val.get("val_roc_auc")),
            "accuracy": _safe_round(final_val.get("val_accuracy")),
            "balanced_accuracy": _safe_round(final_val.get("val_balanced_accuracy")),
        }

    status = "available" if (classification or external_baselines or regression) else "unavailable"

    payload = {
        "schema_version": "model_benchmark_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "synthetic_classification": classification,
        "synthetic_response_regression": regression,
        "external_baselines": external_baselines,
        "external_cnn": cnn_summary,
        "calibration_report": calibration_report,
        "claim_boundary": "Model benchmarks are synthetic or public-data baselines, not clinical validation.",
        "limitations": [
            "Synthetic metrics are not real-world clinical performance.",
            "BreastDCEDL baselines use MRI-derived tabular features and are not a full clinical pipeline.",
        ],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for entry in classification:
        rows.extend([
            {"benchmark": "model", "model": entry["model"], "metric": "roc_auc", "value": entry.get("roc_auc"), "dataset": entry["label"], "notes": "patient_level"},
            {"benchmark": "model", "model": entry["model"], "metric": "auprc", "value": entry.get("auprc"), "dataset": entry["label"], "notes": "patient_level"},
            {"benchmark": "model", "model": entry["model"], "metric": "brier", "value": entry.get("brier"), "dataset": entry["label"], "notes": "patient_level"},
            {"benchmark": "model", "model": entry["model"], "metric": "ece_after", "value": entry.get("ece_after"), "dataset": entry["label"], "notes": "patient_level"},
        ])

    if regression:
        rows.extend([
            {"benchmark": "response_regression", "model": regression.get("model"), "metric": "mae", "value": regression.get("mae"), "dataset": "synthetic", "notes": "patient_level"},
            {"benchmark": "response_regression", "model": regression.get("model"), "metric": "rmse", "value": regression.get("rmse"), "dataset": "synthetic", "notes": "patient_level"},
            {"benchmark": "response_regression", "model": regression.get("model"), "metric": "r2", "value": regression.get("r2"), "dataset": "synthetic", "notes": "patient_level"},
        ])

    for entry in external_baselines:
        rows.extend([
            {"benchmark": "baseline", "model": entry.get("model"), "metric": "roc_auc", "value": entry.get("roc_auc"), "dataset": entry.get("label"), "notes": "public"},
            {"benchmark": "baseline", "model": entry.get("model"), "metric": "accuracy", "value": entry.get("accuracy"), "dataset": entry.get("label"), "notes": "public"},
            {"benchmark": "baseline", "model": entry.get("model"), "metric": "balanced_accuracy", "value": entry.get("balanced_accuracy"), "dataset": entry.get("label"), "notes": "public"},
        ])

    if cnn_summary:
        rows.extend([
            {"benchmark": "baseline", "model": "small_cnn", "metric": "roc_auc", "value": cnn_summary.get("roc_auc"), "dataset": cnn_summary.get("label"), "notes": "public"},
            {"benchmark": "baseline", "model": "small_cnn", "metric": "accuracy", "value": cnn_summary.get("accuracy"), "dataset": cnn_summary.get("label"), "notes": "public"},
            {"benchmark": "baseline", "model": "small_cnn", "metric": "balanced_accuracy", "value": cnn_summary.get("balanced_accuracy"), "dataset": cnn_summary.get("label"), "notes": "public"},
        ])

    _write_csv(args.csv_path, rows)

    print(json.dumps({
        "output_path": args.output_path,
        "status": status,
        "classification_models": [entry.get("model") for entry in classification],
        "response_regression_model": (regression or {}).get("model"),
        "baseline_models": [entry.get("model") for entry in external_baselines],
        "cnn_metrics": bool(cnn_summary),
    }, indent=2))


if __name__ == "__main__":
    main()
