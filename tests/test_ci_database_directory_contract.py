"""A job that points SQLite at a directory must make sure the directory exists.

`Data/test_tmp` is gitignored and holds no tracked files, so it is simply not
there on a fresh runner. SQLite does not create missing parent directories; it
fails the connection with `unable to open database file`. A workflow that sets
`DATABASE_URL` to a path inside that directory and then runs the suite dies on
its first database access, seconds in, for a reason that has nothing to do with
the code under test.

Two of the three jobs that declare a database already handle this and one did
not, which is the asymmetry these tests encode:

* `quality-gates` runs `mkdir -p Data/test_tmp` in the workflow.
* `fresh-clone-smoke` delegates to the fresh-clone verifier, which creates the
  directories it needs as part of proving a clean checkout works.
* the Ship Gate did neither.

Ordering is asserted rather than mere presence, because a `mkdir` that runs
after the suite has already opened the database is decoration.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKFLOW_DIR = ROOT / ".github/workflows"

# The commands that open the application database. Anything earlier in a job
# (installing uv, restoring a cache, provisioning an encoder) does not.
OPENS_THE_DATABASE = re.compile(r"pytest|scripts/ship\.py|verify_fresh_clone\.sh")

# `scripts/verify_fresh_clone.sh` provisions its own working directories, by
# design: its whole purpose is proving a bare checkout can build and test
# itself. A job that delegates to it therefore does not need its own mkdir.
# This is the only entrypoint granted that exemption, and it is named rather
# than pattern-matched so a new self-provisioning script has to be added here
# deliberately.
SELF_PROVISIONING_ENTRYPOINTS = ("verify_fresh_clone.sh",)

SQLITE_FILE_URL = re.compile(r"^sqlite:/+(?P<path>\./.+)$")


def _workflows():
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        yield path.name, yaml.safe_load(path.read_text(encoding="utf-8"))


def _database_jobs():
    """Yield (workflow, job, database directory, run lines in order)."""
    for filename, workflow in _workflows():
        top_level = (workflow.get("env") or {}).get("DATABASE_URL")
        for job_name, job in (workflow.get("jobs") or {}).items():
            url = (job.get("env") or {}).get("DATABASE_URL", top_level)
            if not url:
                continue
            match = SQLITE_FILE_URL.match(str(url))
            if not match:
                continue
            directory = Path(match.group("path")).parent.as_posix().lstrip("./")
            lines = []
            for step in job.get("steps") or []:
                lines.extend(str(step.get("run") or "").splitlines())
            yield filename, job_name, directory, lines


def _is_tracked(directory: str) -> bool:
    """Does git carry any file under this directory?

    An untracked directory does not survive `git clone`, which is the whole
    reason the workflow has to create it.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", directory],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def _creates(directory: str, line: str) -> bool:
    return "mkdir" in line and directory in line


# --- the premise -------------------------------------------------------------


def test_the_scan_finds_the_jobs_that_declare_a_database() -> None:
    """Guard against the contract matching nothing and passing vacuously."""
    jobs = list(_database_jobs())
    assert jobs, "no workflow job declares a file-backed SQLite DATABASE_URL"

    names = {f"{workflow}:{job}" for workflow, job, _, _ in jobs}
    assert "ship.yml:ship" in names, f"the Ship Gate was not detected: {sorted(names)}"


def test_the_ship_database_directory_is_not_tracked() -> None:
    """The premise behind the whole contract, asserted rather than assumed.

    If this directory ever becomes tracked, the mkdir requirement below stops
    being load-bearing and this file should be revisited.
    """
    directories = {d for workflow, job, d, _ in _database_jobs() if workflow == "ship.yml"}
    assert directories == {"Data/test_tmp"}, directories
    assert not _is_tracked("Data/test_tmp"), (
        "Data/test_tmp now has tracked files; it would survive a clone and the "
        "workflow mkdir may no longer be required"
    )


def test_sqlite_will_not_create_a_missing_parent_directory() -> None:
    """Why an absent directory is fatal rather than merely untidy.

    Pinned so that a future reader does not assume the driver back-fills the
    path, which is the assumption that produced the defect.
    """
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "no_such_dir" / "ship.db"
        with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
            sqlite3.connect(target).execute("create table t (a)")


# --- the requirement ---------------------------------------------------------


def test_every_database_job_provides_its_directory() -> None:
    offenders = []
    for workflow, job, directory, lines in _database_jobs():
        if not _is_tracked(directory):
            delegates = any(
                entrypoint in line
                for line in lines
                for entrypoint in SELF_PROVISIONING_ENTRYPOINTS
            )
            if delegates or any(_creates(directory, line) for line in lines):
                continue
            offenders.append(f"{workflow}:{job} needs {directory}")

    assert not offenders, (
        "these jobs point SQLite at an untracked directory without creating it: "
        f"{offenders}. SQLite will fail with 'unable to open database file'."
    )


def test_the_directory_is_created_before_the_database_is_opened() -> None:
    """A mkdir after the first database access has already lost the race."""
    for workflow, job, directory, lines in _database_jobs():
        if _is_tracked(directory):
            continue
        if any(
            entrypoint in line for line in lines for entrypoint in SELF_PROVISIONING_ENTRYPOINTS
        ):
            continue

        created_at = next(
            (i for i, line in enumerate(lines) if _creates(directory, line)),
            None,
        )
        opened_at = next(
            (i for i, line in enumerate(lines) if OPENS_THE_DATABASE.search(line)),
            None,
        )

        assert created_at is not None, f"{workflow}:{job} never creates {directory}"
        assert opened_at is not None, f"{workflow}:{job} never opens the database"
        assert created_at < opened_at, (
            f"{workflow}:{job} creates {directory} at line {created_at} but opens "
            f"the database at line {opened_at}"
        )


def test_the_ship_gate_creates_the_directory_before_running_ship() -> None:
    """The specific regression, stated in its own terms.

    `scripts/ship.py` runs the backend suites; its very first step opens the
    database. On a fresh Ubuntu runner the directory does not exist, so the
    gate failed roughly forty seconds in, long before reaching anything it was
    meant to be testing.
    """
    ship = [entry for entry in _database_jobs() if entry[0] == "ship.yml"]
    assert ship, "the ship job no longer declares a database"

    _, _, directory, lines = ship[0]
    created_at = next((i for i, line in enumerate(lines) if _creates(directory, line)), None)
    ship_at = next((i for i, line in enumerate(lines) if "scripts/ship.py" in line), None)

    assert created_at is not None, f"ship.yml does not create {directory}"
    assert ship_at is not None, "ship.yml no longer runs scripts/ship.py"
    assert created_at < ship_at


# --- the detectors have to be able to fail -----------------------------------


def test_creation_detection_rejects_unrelated_commands() -> None:
    assert not _creates("Data/test_tmp", "npm ci")
    assert not _creates("Data/test_tmp", "uv run python scripts/ship.py")
    assert not _creates("Data/test_tmp", "mkdir -p KnowledgeBase/raw")
    assert _creates("Data/test_tmp", "mkdir -p Data/test_tmp")
    assert _creates("Data/test_tmp", "mkdir -p KnowledgeBase/raw Data/test_tmp")


def test_database_open_detection_rejects_setup_commands() -> None:
    assert not OPENS_THE_DATABASE.search("python -m pip install uv==0.8.24")
    assert not OPENS_THE_DATABASE.search("npx playwright install --with-deps chromium")
    assert OPENS_THE_DATABASE.search("uv run python scripts/ship.py")
    assert OPENS_THE_DATABASE.search("python -m pytest tests -q")
