"""Emit the 10/10-under-constraints roadmap artifact.

Writes ``Data/evals/governance/latest_10_out_of_10_constraint_roadmap.json``.

This is an engineering self-rating snapshot.  10/10 here NEVER means
clinically validated, production healthcare ready, or proven patient
benefit.  See ``docs/ten_out_of_ten_under_constraints.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.ten_out_of_ten_roadmap import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_roadmap,
    write_roadmap,
)


def main() -> int:
    out = write_roadmap(DEFAULT_OUTPUT_PATH)
    payload = build_roadmap()
    s = payload["summary"]
    print(f"wrote: {out}")
    print(f"  dimensions={s['n_dimensions']}  roadmap_items={s['n_roadmap_items']}")
    print(
        f"  avg(excl. real_clinical_readiness)={s['average_score_excluding_real_clinical_readiness']}  "
        f"real_clinical_readiness={s['real_clinical_readiness_score']}"
    )
    print(
        f"  A_implement_now={s['n_implement_now']}  "
        f"B_external_reviewer={s['n_needs_external_reviewer']}  "
        f"C_real_data={s['n_needs_real_data']}  "
        f"D_irb={s['n_needs_irb_institution']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
