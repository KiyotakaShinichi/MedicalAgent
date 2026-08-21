from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data/evals/ops/latest_openapi_type_drift_check.json"

# Pinned deliberately. The generator's output is a committed artifact that this
# very check diffs, so an unpinned `npx openapi-typescript` (which resolves to
# whatever is newest) would let a generator release silently rewrite the
# contract and fail the gate for everyone at once.
#
# It is invoked through npx rather than being a devDependency because
# openapi-typescript 7.x declares `peer typescript@^5.x` while this project is
# on typescript ~6.0; adding it to package.json would force --legacy-peer-deps
# on every install. Keep this version in sync with the `typegen` scripts in
# frontend-react/package.json.
OPENAPI_TYPESCRIPT_VERSION = "7.13.0"


def main() -> int:
    npx = "npx.cmd" if shutil.which("npx.cmd") else "npx"
    rows = []
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "generated-openapi.d.ts"
        commands = [
            ["python", "scripts/export_openapi_schema.py"],
            [
                npx,
                "-y",
                f"openapi-typescript@{OPENAPI_TYPESCRIPT_VERSION}",
                "../Data/openapi.json",
                "-o",
                str(generated),
            ],
        ]
        for command in commands:
            cwd = ROOT / "frontend-react" if Path(command[0]).name.startswith("npx") else ROOT
            result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
            rows.append({
                "command": " ".join(command),
                "exit_code": result.returncode,
                "stdout_tail": result.stdout[-500:],
                "stderr_tail": result.stderr[-500:],
            })
            if result.returncode != 0:
                ok = False
                break
        if ok:
            tracked = ROOT / "frontend-react/src/types/generated-openapi.d.ts"
            tracked_text = tracked.read_text(encoding="utf-8") if tracked.exists() else ""
            generated_text = generated.read_text(encoding="utf-8") if generated.exists() else ""
            same = tracked_text.replace("\r\n", "\n") == generated_text.replace("\r\n", "\n")
            rows.append({
                "command": "compare generated OpenAPI frontend types",
                "exit_code": 0 if same else 1,
                "stdout_tail": "types match" if same else "types drift detected",
                "stderr_tail": "",
            })
            ok = same
    payload = {
        "schema_version": "openapi_type_drift_check_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if ok else "needs_attention",
        "summary": {"passed": ok, "command_count": len(rows)},
        "rows": rows,
        "claim_boundary": "OpenAPI type drift check is software contract evidence only.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
