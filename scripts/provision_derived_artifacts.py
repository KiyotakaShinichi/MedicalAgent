"""Regenerate the derived artifacts the offline test path depends on.

Why this exists
---------------
Several tests consume artifacts that are **gitignored on purpose** - they are
derived data, not source, and some are large. On a developer machine they are
simply present, left over from an earlier run. On a fresh clone they are not,
so the consuming tests fail:

    Data/lakehouse/gold/vector_records.jsonl -> FileNotFoundError
    Data/rag_index/, Data/rag_knowledge_base_chunks.json -> empty retrieval,
        which surfaces as `retrieval_context == []` and `needs_attention`
        eval statuses rather than as a missing-file error.

That is the failure mode where a repository only runs on the machine that
generated its artifacts. The fix is not to commit the artifacts: it is to
declare, for each one, the tracked inputs it derives from and the generator
that rebuilds it, so any clean checkout can reconstruct it deterministically.

Every input listed below is tracked in git. Nothing here reads a file that
merely happens to exist in a developer worktree; if an input were gitignored,
``--verify-inputs`` would fail on a fresh clone, which is exactly the check
that would have caught this regression class before it reached CI.

Provisioning needs no network. It runs before the offline suite, and the suite
itself still runs fully offline afterwards.

Usage
-----
    python scripts/provision_derived_artifacts.py                  # provision what is missing
    python scripts/provision_derived_artifacts.py --check-only     # non-zero if anything is missing
    python scripts/provision_derived_artifacts.py --verify-inputs  # prove inputs are tracked
    python scripts/provision_derived_artifacts.py --force          # rebuild even if present
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class DerivedArtifact:
    """One gitignored artifact plus the tracked inputs that reproduce it."""

    name: str
    path: str
    inputs: tuple[str, ...]
    generator: tuple[str, ...]
    consumers: tuple[str, ...] = field(default=())
    preserved_side_effects: tuple[str, ...] = field(default=())

    def target(self, root: Path) -> Path:
        return root / self.path

    def exists(self, root: Path) -> bool:
        return self.target(root).exists()


# Order matters: the lakehouse pipeline reads the chunk artifact produced by
# the ingestion step, so ingestion is listed (and run) first.
DERIVED_ARTIFACTS: tuple[DerivedArtifact, ...] = (
    DerivedArtifact(
        name="rag_knowledge_base_chunks",
        path="Data/rag_knowledge_base_chunks.json",
        inputs=("KnowledgeBase/raw",),
        generator=("scripts/ingest_knowledge_base.py",),
        consumers=(
            "tests/test_breast_monitoring.py",
            "backend.services.agent_rag",
        ),
    ),
    DerivedArtifact(
        name="rag_vector_index",
        path="Data/rag_index/local_hybrid_rag_index.joblib",
        inputs=("KnowledgeBase/raw",),
        # Built by the same ingestion run that writes the chunk artifact, so it
        # declares no generator of its own and is verified rather than rebuilt.
        generator=(),
        consumers=(
            "tests/test_breast_monitoring.py",
            "backend.services.rag_vector_index",
        ),
    ),
    DerivedArtifact(
        name="lakehouse_gold_vector_records",
        path="Data/lakehouse/gold/vector_records.jsonl",
        inputs=(
            "KnowledgeBase/raw",
            "config/data_contracts.json",
            "Data/evals/rag/latest_kb_source_governance.json",
        ),
        generator=("scripts/run_data_platform_pipeline.py",),
        preserved_side_effects=(
            "Data/lakehouse/lineage/latest_lineage.json",
            "Data/lakehouse/manifests/latest_pipeline_run.json",
            "Data/lakehouse/manifests/latest_source_manifest.json",
        ),
        consumers=(
            "tests/test_managed_vector_shadow_sync.py",
            "tests/test_data_platform_reliability_eval.py",
            "tests/test_cross_domain_assurance_eval.py",
        ),
    ),
)


def missing_inputs(root: Path = ROOT) -> list[str]:
    """Declared inputs absent from this checkout.

    A non-empty result means an artifact is *not* reproducible from tracked
    content - the defect class this module exists to prevent.
    """
    missing: list[str] = []
    for artifact in DERIVED_ARTIFACTS:
        for relative in artifact.inputs:
            if not (root / relative).exists():
                missing.append(f"{artifact.name}:{relative}")
    return sorted(set(missing))


def missing_artifacts(root: Path = ROOT) -> list[str]:
    """Declared artifacts absent from this checkout."""
    return [artifact.name for artifact in DERIVED_ARTIFACTS if not artifact.exists(root)]


def _run_generator(artifact: DerivedArtifact, root: Path) -> dict[str, Any]:
    preserved = {
        root / relative: (root / relative).read_bytes()
        if (root / relative).exists()
        else None
        for relative in artifact.preserved_side_effects
    }
    try:
        proc = subprocess.run(
            [sys.executable, *artifact.generator],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    finally:
        for path, content in preserved.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    return {
        "exit_code": proc.returncode,
        "detail": detail[-1] if detail else "",
    }


def provision(root: Path = ROOT, *, force: bool = False) -> list[dict[str, Any]]:
    """Rebuild missing (or, with ``force``, all) derived artifacts."""
    results: list[dict[str, Any]] = []
    for artifact in DERIVED_ARTIFACTS:
        if not artifact.generator:
            present = artifact.exists(root)
            results.append({
                "artifact": artifact.name,
                "action": "present" if present else "missing",
                "ok": present,
            })
            continue
        if artifact.exists(root) and not force:
            results.append({
                "artifact": artifact.name,
                "action": "already-present",
                "ok": True,
            })
            continue
        outcome = _run_generator(artifact, root)
        results.append({
            "artifact": artifact.name,
            "action": "generated" if outcome["exit_code"] == 0 else "generator-failed",
            "ok": outcome["exit_code"] == 0 and artifact.exists(root),
            "detail": outcome["detail"],
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report missing artifacts without generating; non-zero exit if any are absent",
    )
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="prove every declared input exists in this checkout (catches an untracked input)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even when present")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help=(
            "checkout to operate on (default: this repository). Exists so the preflight "
            "can be exercised against a tree where the artifacts are absent, which is how "
            "the 'missing artifact fails loudly' behaviour is tested."
        ),
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    failed = False
    absent_inputs = missing_inputs(root)
    if args.verify_inputs or args.check_only:
        for entry in absent_inputs:
            print(f"[FAIL] declared input missing: {entry}")
        if absent_inputs:
            failed = True
        else:
            print(f"[OK  ] all declared inputs present ({len(DERIVED_ARTIFACTS)} artifacts)")

    results: list[dict[str, Any]] = []
    if args.check_only:
        for artifact in DERIVED_ARTIFACTS:
            present = artifact.exists(root)
            results.append({
                "artifact": artifact.name,
                "action": "present" if present else "missing",
                "ok": present,
            })
            print(f"[{'OK  ' if present else 'FAIL'}] {artifact.name} ({artifact.path})")
            if not present:
                failed = True
    elif not absent_inputs:
        results = provision(root, force=args.force)
        for entry in results:
            print(f"[{'OK  ' if entry['ok'] else 'FAIL'}] {entry['artifact']} ({entry['action']})")
            if not entry["ok"]:
                failed = True

    if failed:
        print(
            "\nThe offline test path depends on derived artifacts that are absent and "
            "could not be rebuilt. These are gitignored on purpose; run this script "
            "before the suite rather than committing the artifacts.",
            file=sys.stderr,
        )

    if args.json_output:
        payload = {
            "schema_version": "derived_artifact_provisioning_v1",
            "artifacts": results,
            "missing_inputs": absent_inputs,
            "passed": not failed,
            "claim_boundary": (
                "Confirms the gitignored artifacts the offline suite consumes can be rebuilt "
                "from tracked inputs. Makes no claim about model quality, retrieval quality, "
                "safety performance, or clinical validity."
            ),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_output}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
