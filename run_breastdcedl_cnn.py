import argparse
import json

from backend.services.breastdcedl_cnn import run_breastdcedl_small_cnn


def main():
    parser = argparse.ArgumentParser(description="Run a small CPU BreastDCEDL CNN baseline.")
    parser.add_argument("--manifest", default="Data/breastdcedl_spy1_manifest.csv")
    parser.add_argument("--metrics", default="Data/breastdcedl_spy1_cnn_metrics.json")
    parser.add_argument("--max-patients", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = run_breastdcedl_small_cnn(
        manifest_csv_path=args.manifest,
        metrics_json_path=args.metrics,
        max_patients=args.max_patients,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
