"""One-shot bootstrap that seeds demo data and regenerates Safety & Evaluation Center artifacts.

Use this for a fresh portfolio demo::

    python -m scripts.seed_demo_and_evals

It will:

1. Seed demo patients, labs, symptoms, treatments, imaging reports (via ``seed_db``).
2. Run the safety red-team suite.
3. Run the RAG evaluation suite.
4. Generate the drift report.

All synthetic artifacts are clearly labeled in their JSON payloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _seed_demo_data() -> None:
    import importlib

    print("Seeding demo patient data...")
    importlib.import_module("seed_db")


def main() -> int:
    _seed_demo_data()

    from scripts.run_safety_eval_center import _run_drift, _run_rag, _run_safety

    _run_safety()
    _run_rag()
    _run_drift()

    print("\nDemo data + Safety & Evaluation Center artifacts ready.")
    print("Open the admin dashboard and choose 'Safety & Eval Center' to view them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
