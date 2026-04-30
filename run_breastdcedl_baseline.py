import argparse
import json

from backend.services.breastdcedl_baseline import run_breastdcedl_baseline


def main():
    parser = argparse.ArgumentParser(description="Run a lightweight BreastDCEDL pCR baseline.")
    parser.add_argument("--manifest", default="Data/breastdcedl_spy1_manifest.csv")
    parser.add_argument("--features", default="Data/breastdcedl_spy1_features.csv")
    parser.add_argument("--metrics", default="Data/breastdcedl_spy1_baseline_metrics.json")
    parser.add_argument("--predictions", default="Data/breastdcedl_spy1_model_predictions.csv")
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()

    result = run_breastdcedl_baseline(
        manifest_csv_path=args.manifest,
        features_csv_path=args.features,
        metrics_json_path=args.metrics,
        predictions_csv_path=args.predictions,
        max_patients=args.max_patients,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
