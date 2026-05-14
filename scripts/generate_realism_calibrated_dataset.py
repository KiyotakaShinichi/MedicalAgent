import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.complete_synthetic_dataset import generate_complete_synthetic_breast_dataset  # noqa: E402
from backend.services.synthetic_realism_report import build_synthetic_realism_report  # noqa: E402


DEFAULT_OUTPUT_DIR = "Data/complete_synthetic_breast_journeys_realism_v2"
DEFAULT_REALISM_REPORT = "Data/mle_monitoring/synthetic_realism_candidate_report.json"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a BreastDCEDL-calibrated synthetic candidate dataset and realism report."
    )
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REALISM_REPORT)
    parser.add_argument("--missing-rate", type=float, default=0.08)
    parser.add_argument("--noise-level", type=float, default=0.05)
    args = parser.parse_args()

    summary = generate_complete_synthetic_breast_dataset(
        db=None,
        count=args.count,
        seed=args.seed,
        cycles=args.cycles,
        output_dir=args.output_dir,
        write_db=False,
        patient_prefix="REALISM-BRCA-",
        balanced_outcomes=True,
        balanced_subgroups=True,
        missing_rate=args.missing_rate,
        noise_level=args.noise_level,
        realism_profile="external_calibrated",
        toxicity_profile="realistic",
        missingness_mode="ehr_like",
    )
    training_csv = str(Path(args.output_dir) / "temporal_ml_rows.csv")
    report = build_synthetic_realism_report(
        training_csv=training_csv,
        output_path=args.report_path,
    )
    print(json.dumps({
        "dataset_output_dir": args.output_dir,
        "summary_path": str(Path(args.output_dir) / "summary.json"),
        "realism_report_path": args.report_path,
        "patients": summary.get("patients_created"),
        "rows": report.get("training_rows"),
        "status": report.get("status"),
        "alignment": report.get("realism_alignment_score"),
        "sim_to_real_status": (report.get("sim_to_real_comparison") or {}).get("status"),
        "threshold_status": (report.get("threshold_coverage") or {}).get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
