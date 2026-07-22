"""Evaluate baseline/candidate generations and emit the fine-tune gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.finetune_promotion import build_promotion_decision
from scripts.evaluate_finetuned_behavior import evaluate_dataset

DEFAULT_HOLDOUT = ROOT / "Data" / "finetune" / "prepared" / "dataset_internal_frozen_holdout.jsonl"
DEFAULT_BASELINE = ROOT / "Data" / "finetune" / "evaluations" / "baseline_generations.jsonl"
DEFAULT_CANDIDATE = ROOT / "Data" / "finetune" / "evaluations" / "candidate_generations.jsonl"
DEFAULT_OUTPUT = ROOT / "Data" / "evals" / "models" / "latest_finetune_promotion_gate.json"


def run_gate(
    holdout: Path = DEFAULT_HOLDOUT,
    baseline_path: Path = DEFAULT_BASELINE,
    candidate_path: Path = DEFAULT_CANDIDATE,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict:
    _validate_internal_holdout(holdout)
    baseline = None
    candidate = None
    evaluation_dir = output_path.parent
    if baseline_path.exists():
        baseline = evaluate_dataset(
            holdout,
            evaluation_dir / "latest_finetune_baseline_behavior_eval.json",
            baseline_path,
            "base_model_generations",
        )
    if candidate_path.exists():
        candidate = evaluate_dataset(
            holdout,
            evaluation_dir / "latest_finetune_candidate_behavior_eval.json",
            candidate_path,
            "adapter_candidate_generations",
        )
    decision = build_promotion_decision(baseline, candidate)
    decision["evidence"] = {
        "holdout_path": _rel(holdout),
        "baseline_generations_path": _rel(baseline_path),
        "candidate_generations_path": _rel(candidate_path),
        "baseline_present": baseline_path.exists(),
        "candidate_present": candidate_path.exists(),
        "internal_holdout_is_external_evidence": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def _validate_internal_holdout(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Fine-tune holdout is empty.")
    for row in rows:
        if row.get("split") != "internal_frozen_holdout":
            raise ValueError(f"Case {row.get('id')} is not in the internal frozen holdout.")
        if (row.get("provenance") or {}).get("was_used_for_tuning") is not False:
            raise ValueError(f"Case {row.get('id')} does not prove tuning exclusion.")


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NLCare fine-tune promotion gate.")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_gate(args.holdout, args.baseline, args.candidate, args.output)
    print(json.dumps(report, indent=2))
    return 1 if report["decision"] == "REJECT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
