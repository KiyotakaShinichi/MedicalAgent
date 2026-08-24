"""A failing Ship Gate has to be able to say why.

`scripts/ship.py` already reports well: it prints each step's name, working
directory and command, times it, and on failure writes the failing step, its
exit code and a machine-readable manifest. Its children inherit stdout and
stderr, so their output lands in the job log too.

None of which helped, because reading a GitHub job log needs admin rights on
the repository. From outside, a failing gate was an exit code and nothing else,
and three rounds of diagnosis had to be done by reproduction and inference.

So the workflow now tees the run to a file and uploads it, along with the
manifest ship.py already writes. The tests here are mostly about the ways that
change could go wrong: a tee that swallows the exit code, an upload that only
runs on success, or an artifact that sweeps up the databases sitting in the
same directory.
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

SHIP_WORKFLOW = ROOT / ".github/workflows/ship.yml"
CAPTURE_PATH = "Data/test_tmp/ship_gate_output.log"
ARTIFACT_NAME = "ship-gate-diagnostics"


def _ship_job() -> dict:
    return yaml.safe_load(SHIP_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["ship"]


def _steps() -> list[dict]:
    return _ship_job().get("steps") or []


def _gate_step() -> dict:
    for step in _steps():
        if "scripts/ship.py" in str(step.get("run") or ""):
            return step
    raise AssertionError("no step runs scripts/ship.py")


def _upload_steps() -> list[dict]:
    return [s for s in _steps() if "upload-artifact" in str(s.get("uses") or "")]


def _artifact_paths(step: dict) -> list[str]:
    raw = str((step.get("with") or {}).get("path") or "")
    return [line.strip() for line in raw.splitlines() if line.strip()]


# --- the output is captured --------------------------------------------------


def test_the_gate_tees_its_output_to_a_file() -> None:
    run = str(_gate_step().get("run") or "")
    assert "tee" in run, "the ship gate does not capture its output"
    assert CAPTURE_PATH in run, f"expected the run to tee to {CAPTURE_PATH}"


def test_stderr_is_captured_too() -> None:
    """`[ship] FAILED: <step>` is printed to stderr, so a stdout-only capture
    would lose the one line naming the failing step."""
    run = str(_gate_step().get("run") or "")
    assert "2>&1" in run, "stderr is not redirected into the captured stream"


def test_the_capture_path_is_deterministic() -> None:
    """A reader has to know the filename without guessing."""
    run = str(_gate_step().get("run") or "")
    assert not re.search(r"\$\{\{", run.split("tee")[-1]), (
        "the capture path is templated; it must be a fixed, predictable name"
    )
    assert CAPTURE_PATH in _artifact_paths(_upload_steps()[0])


# --- the exit code stays authoritative ---------------------------------------


def test_the_gate_sets_pipefail() -> None:
    """Without it the pipeline reports tee's status, and tee always succeeds.

    This is the single most dangerous way this change could go wrong: it would
    turn every future Ship Gate failure green while looking like an improvement.
    """
    step = _gate_step()
    run = str(step.get("run") or "")
    assert "pipefail" in run, "a tee'd gate without pipefail cannot fail the job"
    assert step.get("shell") == "bash", (
        "pipefail needs an explicit bash shell to be guaranteed"
    )


def _usable_bash() -> bool:
    """Is `bash` here a working POSIX shell?

    On Windows `bash` often resolves to the WSL launcher, which fails to exec a
    shell at all when no distribution is installed. The runner this contract
    describes is Ubuntu, so the check below runs for real there and steps aside
    on a developer machine rather than failing for an unrelated reason.
    """
    try:
        probe = subprocess.run(["bash", "-c", "exit 0"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and b"WSL" not in probe.stderr


def test_pipefail_actually_preserves_a_failing_exit_code() -> None:
    """Asserted by running it, not by trusting the flag's reputation."""
    if not _usable_bash():
        pytest.skip("no POSIX bash available to exercise pipefail")

    piped = "set -o pipefail; (exit 3) | tee /dev/null"
    assert subprocess.run(["bash", "-c", piped], capture_output=True).returncode == 3

    without = "(exit 3) | tee /dev/null"
    assert subprocess.run(["bash", "-c", without], capture_output=True).returncode == 0, (
        "if this ever returns non-zero, bash changed and the pipefail "
        "requirement above should be re-examined rather than assumed"
    )


def test_no_step_is_marked_continue_on_error() -> None:
    for step in _steps():
        assert step.get("continue-on-error") is not True, (
            f"step {step.get('name')!r} would swallow its own failure"
        )


def test_the_job_is_not_marked_continue_on_error() -> None:
    assert _ship_job().get("continue-on-error") is not True


def test_the_upload_cannot_rescue_a_failing_gate() -> None:
    """The upload runs after the gate, so it cannot change the job's result.

    Ordering matters here: an `always()` step placed *before* the gate would
    still upload, but an upload that ran first would have nothing to collect.
    """
    names = [str(s.get("name") or s.get("uses") or "") for s in _steps()]
    gate_at = next(i for i, s in enumerate(_steps()) if "scripts/ship.py" in str(s.get("run") or ""))
    upload_at = next(i for i, s in enumerate(_steps()) if "upload-artifact" in str(s.get("uses") or ""))

    assert upload_at > gate_at, f"upload must follow the gate; got {names}"


# --- the diagnosis survives failure -----------------------------------------


def test_diagnostics_are_uploaded_even_when_the_gate_fails() -> None:
    uploads = _upload_steps()
    assert uploads, "the ship job uploads no diagnostics"

    diagnostics = [s for s in uploads if (s.get("with") or {}).get("name") == ARTIFACT_NAME]
    assert diagnostics, f"no upload step named {ARTIFACT_NAME!r}"

    condition = str(diagnostics[0].get("if") or "")
    assert "always()" in condition or "failure()" in condition, (
        f"the diagnostics upload is conditioned on {condition!r}; a plain "
        "success-only step would skip exactly when it is needed"
    )


def test_the_artifact_carries_the_log_and_the_manifest() -> None:
    """ship.py already writes a structured manifest; both go up together.

    The log holds stdout and stderr, the manifest holds the failing step's
    name, command, exit code and timing. Either alone leaves the reader
    guessing at half of it.
    """
    paths = _artifact_paths(
        next(s for s in _upload_steps() if (s.get("with") or {}).get("name") == ARTIFACT_NAME)
    )
    assert CAPTURE_PATH in paths
    assert any("latest_ship_run.json" in p for p in paths), (
        f"the ship manifest is not uploaded: {paths}"
    )


# --- nothing sensitive rides along -------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [".env", "credentials", "secrets", ".key", ".pem", "token"],
)
def test_no_sensitive_path_is_uploaded(forbidden: str) -> None:
    for step in _upload_steps():
        for path in _artifact_paths(step):
            assert forbidden not in path.lower(), f"{path!r} looks sensitive"


def test_the_artifact_does_not_sweep_up_the_databases() -> None:
    """Data/test_tmp holds the SQLite databases the run creates.

    Uploading the directory would ship those too. The paths are named files for
    that reason, so this asserts the directory itself is never listed.
    """
    for step in _upload_steps():
        for path in _artifact_paths(step):
            assert not path.rstrip("/").endswith("Data/test_tmp"), (
                f"{path!r} would upload the whole directory, databases included"
            )
            assert not path.endswith(".db"), f"{path!r} is a database"


def test_no_workflow_secret_is_interpolated_into_the_diagnostics() -> None:
    """The captured log is the run's own output; nothing injects a secret."""
    text = SHIP_WORKFLOW.read_text(encoding="utf-8")
    assert "secrets." not in text, (
        "ship.yml now references a secret; the uploaded log would need review "
        "before it can be published as an artifact"
    )
