import argparse
import json

from backend.services.complete_synthetic_training import train_complete_synthetic_models


def main():
    parser = argparse.ArgumentParser(description="Train models on complete synthetic breast cancer journey data.")
    parser.add_argument("--csv", default="Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
    parser.add_argument("--output-dir", default="Data/complete_synthetic_training")
    parser.add_argument("--target", default="treatment_success_binary")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    args = parser.parse_args()

    result = train_complete_synthetic_models(
        ml_csv_path=args.csv,
        output_dir=args.output_dir,
        target=args.target,
        test_size=args.test_size,
        seed=args.seed,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
