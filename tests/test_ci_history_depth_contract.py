"""A job that runs every test file needs the history those tests read.

Several suites prove that a refactor preserved behaviour by loading the
pre-refactor module straight out of git, with `git show <commit>:<path>`. That
only works if the commit is in the checkout.

`actions/checkout` clones one commit deep by default, so under it those loaders
raise. Worse, they raise at *import* time, which pytest reports as a collection
error: the run ends with every test collected and none executed, and a job that
looks like a test failure never ran a test at all. That is what happened once
here - 2498 collected, 0 passed, 1 error - and the only visible symptom was an
exit code.

So: any job that runs the whole tests tree must check out full history. Jobs
that run a named subset need not, because pytest imports only the files it is
given, and none of the named subsets include a history-reading suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKFLOW_DIR = ROOT / ".github/workflows"

# `pytest tests` (the whole tree) or the verifier that runs it.
RUNS_WHOLE_TREE = (
    re.compile(r"pytest\s+tests\b(?!/)"),
    re.compile(r"verify_fresh_clone\.sh"),
)

# `git show <rev>:<path>` with a rev that is not HEAD needs real history.
READS_HISTORY = re.compile(r"""["']git["']\s*,\s*["']show["']""")


def _workflows():
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        yield path.name, yaml.safe_load(path.read_text(encoding="utf-8"))


def _jobs_running_the_whole_tree():
    for filename, workflow in _workflows():
        for job_name, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            commands = "\n".join(str(s.get("run") or "") for s in steps)
            if any(pattern.search(commands) for pattern in RUNS_WHOLE_TREE):
                yield filename, job_name, steps


def _checkout_depth(steps) -> object:
    for step in steps:
        if "actions/checkout" in str(step.get("uses") or ""):
            return (step.get("with") or {}).get("fetch-depth", "default")
    return None


def _history_reading_test_files() -> list[Path]:
    """Test modules that load a revision out of git."""
    found = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if READS_HISTORY.search(text):
            found.append(path)
    return found


# --- the premise --------------------------------------------------------------


def test_some_suite_actually_reads_git_history() -> None:
    """Guard against this contract quietly protecting nothing."""
    assert _history_reading_test_files(), (
        "no test loads a module from git any more; if that is intentional, "
        "this contract can go"
    )


def test_the_scan_finds_the_whole_tree_job() -> None:
    jobs = {f"{wf}:{job}" for wf, job, _ in _jobs_running_the_whole_tree()}
    assert jobs, "no job runs the whole tests tree; the scan is broken"


# --- the requirement ----------------------------------------------------------


def test_whole_tree_jobs_check_out_full_history() -> None:
    offenders = []
    for filename, job_name, steps in _jobs_running_the_whole_tree():
        if _checkout_depth(steps) != 0:
            offenders.append(f"{filename}:{job_name}")

    assert not offenders, (
        f"these jobs run every test file on a shallow checkout: {offenders}. "
        "Suites that load a past revision will fail during collection, and the "
        "run will execute no tests at all."
    )


@pytest.mark.parametrize("path", _history_reading_test_files(), ids=lambda p: p.name)
def test_a_pinned_revision_is_reachable(path: Path) -> None:
    """Every fixed commit a suite pins must exist in this checkout.

    A pinned SHA that has been garbage-collected or rebased away fails the same
    way a shallow clone does, and just as opaquely.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    for sha in set(re.findall(r"\b[0-9a-f]{40}\b", text)):
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"{path.name} pins commit {sha[:12]}, which is not in this "
            "repository; the suite cannot load its reference module"
        )


def test_history_readers_fail_loudly_rather_than_silently() -> None:
    """If a loader ever starts skipping instead, this contract loses its point.

    A skip would keep CI green while the equivalence check stopped running,
    which is worse than the collection error it replaced.
    """
    for path in _history_reading_test_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        window = text[: text.find("def test_")] if "def test_" in text else text
        assert "skip" not in window.lower(), (
            f"{path.name} appears to skip at import when history is missing; "
            "prefer giving the job full history so the check actually runs"
        )
