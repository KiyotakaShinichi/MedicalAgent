"""One command that takes a fresh clone to a runnable, testable state.

    python scripts/bootstrap.py

Why this exists
---------------
Setup was previously four commands documented in three places, and the order
mattered in ways only the README's prose conveyed: the semantic-safety
encoders must be cached *before* the offline suite runs, and the derived RAG
and lakehouse artifacts must be rebuilt *before* the tests that read them.
Getting the order wrong does not produce a clear error — it produces a suite
that fails ~90 tests because the safety runtimes fail closed, or ~5 because a
retrieval index is empty.

Everything here is declared, reproducible, and free of developer-local paths.
No step requires a secret, and no step reads a file that merely happens to
exist on the machine that ran it before.

Network use is confined to the dependency sync and the one-time encoder
download. After bootstrap, the test suite runs fully offline — and the default
suite additionally blocks outbound connections and clears third-party
credentials (see `tests/conftest.py`).

    python scripts/bootstrap.py --check       # verify without changing anything
    python scripts/bootstrap.py --with-frontend
    python scripts/bootstrap.py --json-output path.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str
    seconds: float


def _run(name: str, argv: list[str], *, cwd: Path = ROOT, optional: bool = False) -> StepResult:
    started = time.perf_counter()
    executable = argv[0] if Path(argv[0]).is_file() else shutil.which(argv[0])
    if executable is None:
        detail = f"{argv[0]} not found on PATH"
        return StepResult(name, optional, detail, 0.0)
    command = [executable, *argv[1:]]
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    elapsed = round(time.perf_counter() - started, 1)
    if proc.returncode == 0:
        tail = (proc.stdout or "").strip().splitlines()
        return StepResult(name, True, tail[-1][:200] if tail else "ok", elapsed)
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return StepResult(name, False, tail[-1][:300] if tail else f"exit {proc.returncode}", elapsed)


def _uv_argv() -> list[str] | None:
    """uv is the declared installer; fall back to nothing rather than guessing."""
    return ["uv", "sync", "--frozen"] if shutil.which("uv") else None


def bootstrap(*, check_only: bool = False, with_frontend: bool = False) -> list[StepResult]:
    results: list[StepResult] = []
    py = sys.executable

    # 1. Dependencies. Skipped in --check because syncing is not a verification.
    if not check_only:
        argv = _uv_argv()
        if argv is None:
            results.append(
                StepResult(
                    "sync_dependencies",
                    False,
                    "uv not on PATH — install it (`pip install uv==0.8.24`) and re-run",
                    0.0,
                )
            )
        else:
            results.append(_run("sync_dependencies", argv))

    # 2. Semantic safety encoders. The only step that needs the network, and
    #    only on the first run. Without it every DEP-001 runtime fails closed.
    results.append(
        _run(
            "provision_safety_encoders",
            [py, "scripts/provision_semantic_safety_encoders.py"]
            + (["--check-only"] if check_only else []),
        )
    )

    # 3. Derived artifacts (RAG index, KB chunks, lakehouse gold), rebuilt from
    #    tracked inputs. Offline.
    results.append(
        _run(
            "provision_derived_artifacts",
            [py, "scripts/provision_derived_artifacts.py"]
            + (["--check-only"] if check_only else []),
        )
    )

    # 4. Frontend deps are opt-in: the backend suite does not need them, and
    #    `npm ci` is a slow step to impose on someone bootstrapping for tests.
    if with_frontend:
        results.append(_run("frontend_npm_ci", ["npm", "ci"], cwd=ROOT / "frontend-react"))

    # 5. Preflight. Proves the two provisioning steps actually satisfied the
    #    runtimes rather than merely exiting zero.
    results.append(
        _run(
            "verify_safety_runtimes",
            [py, "scripts/provision_semantic_safety_encoders.py", "--check-only", "--verify-runtimes"],
        )
    )
    results.append(
        _run("verify_fresh_clone_contract", [py, "scripts/check_fresh_clone_offline.py"])
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing environment without downloading or generating anything",
    )
    parser.add_argument(
        "--with-frontend",
        action="store_true",
        help="also run `npm ci` in frontend-react",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    results = bootstrap(check_only=args.check, with_frontend=args.with_frontend)
    print()
    for result in results:
        print(f"[{'OK  ' if result.ok else 'FAIL'}] {result.name} ({result.seconds}s) {result.detail}")

    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(
            "Bootstrap incomplete. Fix the failing step above before running the "
            "test suite: a partially provisioned tree does not fail loudly, it "
            "fails as ~90 safety tests or an empty retrieval index.",
            file=sys.stderr,
        )
    else:
        print("Bootstrap complete. Run the default suite with:")
        print("    uv run pytest tests -q --cov=backend --cov-branch --cov-fail-under=60")

    if args.json_output:
        payload: dict[str, Any] = {
            "schema_version": "bootstrap_v1",
            "steps": [asdict(r) for r in results],
            "passed": not failed,
            "claim_boundary": (
                "Confirms a checkout can be brought to a runnable, testable state from "
                "tracked inputs plus a one-time encoder download. Makes no claim about "
                "model quality, safety performance, or clinical validity."
            ),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_output}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
