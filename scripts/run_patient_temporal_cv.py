"""Run patient-level temporal CV vs. naive row-level CV side by side.

Writes ``Data/evals/models/latest_patient_temporal_cv.json`` and prints a
short summary (AUC mean per strategy + the optimism delta).

Usage::

    python scripts/run_patient_temporal_cv.py
    python scripts/run_patient_temporal_cv.py --target treatment_success_binary
    python scripts/run_patient_temporal_cv.py --n-folds 5 --seed 17

The output JSON is consumed by ``docs/patient_temporal_cv.md`` and by
the release gate (PART 8).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.patient_temporal_cv import (  # noqa: E402
    DEFAULT_ML_CSV_PATH,
    DEFAULT_N_FOLDS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SEED,
    DEFAULT_TARGET,
    build_cv_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-csv", default=DEFAULT_ML_CSV_PATH)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_cv_comparison(
        ml_csv_path=args.ml_csv,
        target=args.target,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote: {out}")
    pat = report["patient_level_temporal_cv"]
    naive = report["naive_row_level_kfold"]
    print(f"target: {args.target}  n_folds={args.n_folds}  seed={args.seed}")
    print(
        "  patient-level temporal CV: "
        f"AUC mean={pat['roc_auc_mean']}  std={pat['roc_auc_std']}  "
        f"overlap_pairs={pat['patient_overlap_pairs']}"
    )
    print(
        "  naive row-level KFold:     "
        f"AUC mean={naive['roc_auc_mean']}  std={naive['roc_auc_std']}  "
        f"overlap_pairs={naive['patient_overlap_pairs']}"
    )
    print(f"  optimism delta (naive - patient): {report['headline']['auc_optimism_from_naive_cv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
