"""Fail when a backend service file grows past the size limit.

    python scripts/check_file_size.py backend/services --max-loc 500

Large modules are where responsibilities go to hide: a 900-line service has no
seam a reviewer can hold in their head, and every change to it touches code
that has nothing to do with the change.

Why this is not a flat "no file over 500 lines" rule
---------------------------------------------------
This repository already has pre-existing modules above the limit. A flat rule
would fail `main` on day one for debt unrelated to whatever a contributor is
actually changing, and a check that is red for reasons outside your control is
a check people learn to ignore.

So the limit is enforced as a **ratchet** against a checked-in baseline:

* a file **not** in the baseline must be at or under the limit — new code has
  no excuse;
* a file **in** the baseline must not exceed its recorded size — existing debt
  is frozen where it is and cannot grow;
* the baseline is **shrink-only**: `--update` may lower a recorded size or drop
  a file that has come under the limit, and refuses to raise one.

The effect is that the debt can only ever get smaller, without blocking work on
anything else.

Exclusions are deliberately narrow: only machine-generated trees, listed
explicitly in `GENERATED_PATH_PARTS`. Anything else that is too long is meant
to be too long.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "tests" / "contracts" / "backend_loc_baseline.json"
DEFAULT_MAX_LOC = 500
DEFAULT_EXTENSIONS = (".py",)

# Machine-generated or vendored trees. Kept short on purpose: every entry here
# is a place the limit stops applying, so a broad pattern would quietly excuse
# hand-written code.
GENERATED_PATH_PARTS = (
    "migrations/versions",  # alembic revision files are generated
    "__pycache__",
    "frontend-react/src/types/generated-openapi.d.ts",
)


def physical_loc(path: Path) -> int:
    """Physical line count, identical on Windows and Linux.

    Reads bytes and splits on universal newlines so a CRLF checkout and an LF
    checkout of the same file agree — otherwise the baseline would be portable
    in name only.
    """
    text = path.read_bytes().decode("utf-8", errors="replace")
    return len(text.splitlines())


def is_generated(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    return any(part in normalized for part in GENERATED_PATH_PARTS)


def tracked_source_files(
    root: Path,
    target: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Return tracked authored source files under ``target``."""
    result = subprocess.run(
        ["git", "ls-files", "--", str(target.relative_to(root)).replace("\\", "/")],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        root / line
        for line in result.stdout.splitlines()
        if line.strip()
        and not is_generated(line)
        and any(line.endswith(extension) for extension in extensions)
    ]


def tracked_python_files(root: Path, target: Path) -> list[Path]:
    """Backward-compatible Python-only tracked-file helper."""
    return tracked_source_files(root, target)


def load_baseline(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in data.get("files", {}).items()}


def measure(
    root: Path,
    target: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> dict[str, int]:
    sizes = {}
    for path in tracked_source_files(root, target, extensions):
        relative = path.relative_to(root).as_posix()
        sizes[relative] = physical_loc(path)
    return sizes


def evaluate(sizes: dict[str, int], baseline: dict[str, int], max_loc: int) -> list[str]:
    """Return one message per violation, empty when the ratchet holds."""
    problems: list[str] = []
    for relative in sorted(sizes):
        loc = sizes[relative]
        recorded = baseline.get(relative)
        if recorded is None:
            if loc > max_loc:
                problems.append(
                    f"{relative}: {loc} LOC exceeds the {max_loc} limit. "
                    "Split it by responsibility, or if this is pre-existing debt "
                    "record it with --update."
                )
        elif loc > recorded:
            problems.append(
                f"{relative}: {loc} LOC, above its recorded baseline of {recorded}. "
                "This file is known debt and may shrink but must not grow — move "
                "the new code into a focused module instead."
            )
    return problems


def write_baseline(
    path: Path, sizes: dict[str, int], baseline: dict[str, int], max_loc: int
) -> dict:
    """Shrink-only update: sizes may fall, never rise."""
    updated: dict[str, int] = {}
    lowered, dropped = [], []
    for relative, loc in sorted(sizes.items()):
        recorded = baseline.get(relative)
        if loc <= max_loc:
            if recorded is not None:
                dropped.append(relative)
            continue
        if recorded is None:
            updated[relative] = loc
        else:
            updated[relative] = min(recorded, loc)
            if loc < recorded:
                lowered.append(f"{relative} {recorded} -> {loc}")
    # A baseline entry whose file disappeared is dropped too.
    for relative in baseline:
        if relative not in sizes:
            dropped.append(relative)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Pre-existing files above the LOC limit, frozen at their current size. "
            "This list is shrink-only: scripts/check_file_size.py fails if a listed "
            "file grows, and --update may lower an entry or drop one that has come "
            "under the limit, never raise one. Do not add a new file here to get a "
            "large module past review."
        ),
        "max_loc": max_loc,
        "files": updated,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {"recorded": len(updated), "lowered": lowered, "dropped": sorted(set(dropped))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "target",
        nargs="?",
        default="backend/services",
        help="directory to inspect (default: backend/services)",
    )
    parser.add_argument("--max-loc", type=int, default=DEFAULT_MAX_LOC)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--extensions",
        default=".py",
        help="comma-separated authored source suffixes (default: .py)",
    )
    parser.add_argument("--update", action="store_true", help="rewrite the baseline (shrink-only)")
    args = parser.parse_args(argv)

    target = (ROOT / args.target).resolve()
    if not target.is_dir():
        print(f"check_file_size: not a directory: {args.target}", file=sys.stderr)
        return 2

    extensions = tuple(
        suffix if suffix.startswith(".") else f".{suffix}"
        for raw in args.extensions.split(",")
        if (suffix := raw.strip())
    )
    if not extensions:
        print("check_file_size: at least one extension is required", file=sys.stderr)
        return 2
    sizes = measure(ROOT, target, extensions)
    baseline = load_baseline(args.baseline)

    if args.update:
        result = write_baseline(args.baseline, sizes, baseline, args.max_loc)
        print(f"Baseline updated: {result['recorded']} file(s) over {args.max_loc} LOC recorded.")
        for entry in result["lowered"]:
            print(f"  lowered: {entry}")
        for entry in result["dropped"]:
            print(f"  dropped (now within the limit): {entry}")
        return 0

    problems = evaluate(sizes, baseline, args.max_loc)
    if problems:
        print(f"File size check: FAILED ({len(problems)} violation(s))")
        for problem in problems:
            print(f"- {problem}")
        return 1

    over = sum(1 for r, loc in sizes.items() if loc > args.max_loc)
    print(
        f"File size check: PASSED: {len(sizes)} tracked file(s) under {args.target}, "
        f"{over} known over {args.max_loc} LOC and none grew."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
