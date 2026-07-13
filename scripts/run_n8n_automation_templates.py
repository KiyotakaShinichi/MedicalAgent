from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.n8n_automation_templates import (  # noqa: E402
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TEMPLATE_DIR,
    build_n8n_automation_templates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build optional n8n internal automation workflow templates.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    parser.add_argument("--template-dir", default=DEFAULT_TEMPLATE_DIR)
    args = parser.parse_args()

    report = build_n8n_automation_templates(
        output_path=args.output,
        doc_path=args.doc,
        template_dir=args.template_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "template_count": report["template_count"],
                "clinical_validation": report["clinical_validation"],
                "phi_allowed": report["phi_allowed"],
                "output_path": args.output,
                "doc_path": args.doc,
                "template_dir": args.template_dir,
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"ready_for_optional_import", "acceptable", "strong"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
