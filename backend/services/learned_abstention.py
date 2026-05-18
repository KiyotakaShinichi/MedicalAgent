from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.services.learned_abstention_experiment import run_learned_abstention_experiment


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_learned_abstention.json"


def train_learned_abstention_head(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    payload = run_learned_abstention_experiment(output_path=output_path)
    payload["schema_version"] = "learned_abstention_head_v1"
    payload["production_policy"] = (
        "Rule-first abstention remains production policy. The learned head is a candidate "
        "review signal unless it improves safety under external validation."
    )
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_learned_abstention(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    return train_learned_abstention_head(output_path=output_path)


def load_learned_abstention(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "missing"}


__all__ = ["DEFAULT_OUTPUT_PATH", "train_learned_abstention_head", "evaluate_learned_abstention", "load_learned_abstention"]
