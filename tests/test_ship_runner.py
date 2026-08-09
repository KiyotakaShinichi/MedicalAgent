from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import ship


def test_ship_step_uses_bounded_timeout(monkeypatch):
    observed = {}

    def fake_run(command, *, cwd, env, check, timeout):
        observed.update(
            {
                "command": command,
                "cwd": cwd,
                "check": check,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    result = ship._run(
        ship.Step(
            name="fixture",
            command=[sys.executable, "-c", "print('ok')"],
            timeout_seconds=45,
        )
    )

    assert observed["check"] is True
    assert observed["timeout"] == 45
    assert result["status"] == "passed"
    assert result["timeout_seconds"] == 45


def test_ship_manifest_keeps_nonclinical_boundary(tmp_path, monkeypatch):
    target = tmp_path / "ship.json"
    monkeypatch.setattr(ship, "SHIP_MANIFEST", target)

    ship._write_manifest(
        status="failed",
        step_results=[{"name": "fixture", "status": "timed_out"}],
        failed_step="fixture",
        failure_kind="timeout",
    )

    text = target.read_text(encoding="utf-8")
    assert '"clinical_validation": false' in text
    assert '"healthcare_production_ready": false' in text
    assert '"failure_kind": "timeout"' in text


def test_ship_tiers_keep_fast_and_evidence_surfaces_distinct():
    steps = ship._build_steps()
    fast = ship._select_steps(steps, "fast")
    evidence = ship._select_steps(steps, "evidence")
    release = ship._select_steps(steps, "release")
    assert fast
    assert evidence
    assert len(release) > len(fast)
    assert all(step.name in ship.FAST_STEP_NAMES for step in fast)
    assert all(ship._is_evidence_step(step) for step in evidence)
    assert "Frontend Playwright smoke" not in {step.name for step in fast}
    fast_names = {step.name for step in fast}
    assert "Cloud, data-platform, and managed-vector contract tests" in fast_names
    assert "Assurance, XAI, automation, and safety contract tests" in fast_names
    assert "Fail-closed RAG release assurance" in fast_names
    assert "Restricted synthetic staging assurance" in fast_names


def test_fail_closed_rag_assurance_precedes_release_decision_surface():
    names = [step.name for step in ship._build_steps()]
    assert names.index("Fail-closed RAG release assurance") < names.index(
        "Canonical release decision surface"
    )


def test_restricted_staging_assurance_precedes_release_decision_surface():
    names = [step.name for step in ship._build_steps()]
    assert names.index("Restricted synthetic staging assurance") < names.index(
        "Canonical release decision surface"
    )


def test_required_freshness_artifacts_are_regenerated_before_release_gate():
    steps = ship._build_steps()
    names = [step.name for step in steps]
    gate_index = names.index("Release artifact gate")
    expected = {
        "Required safety benchmark refresh": "scripts/run_safety_benchmark.py",
        "Required adversarial benchmark refresh": "scripts/run_adversarial_benchmark.py",
        "Required RAG benchmark refresh": "scripts/run_rag_benchmark.py",
        "Required intent-aware RAG benchmark refresh": "scripts/run_rag_intent_aware_eval.py",
    }
    for name, script in expected.items():
        assert names.index(name) < gate_index
        step = next(item for item in steps if item.name == name)
        assert step.command[-1] == script
        assert ship._is_evidence_step(step)


def test_research_paper_kb_eval_is_a_bounded_evidence_step():
    steps = ship._build_steps()
    step = next(
        item
        for item in steps
        if item.name == "Research-paper KB provenance and retrieval evaluation"
    )
    assert step.command[-1] == "scripts/run_research_paper_kb_eval.py"
    assert ship._effective_timeout(step) <= 600
    assert step in ship._select_steps(steps, "evidence")


def test_research_paper_query_telemetry_is_a_bounded_evidence_step():
    steps = ship._build_steps()
    step = next(
        item
        for item in steps
        if item.name == "Research-paper per-query token and latency telemetry"
    )
    assert step.command[-1] == "scripts/run_research_paper_query_telemetry.py"
    assert ship._effective_timeout(step) <= 600
    assert step in ship._select_steps(steps, "evidence")


def test_oversized_backend_contract_suite_is_split_into_bounded_steps():
    steps = ship._build_steps()
    names = {
        "Cloud, data-platform, and managed-vector contract tests",
        "Assurance, XAI, automation, and safety contract tests",
    }
    backend_contract_steps = [step for step in steps if step.name in names]
    assert len(backend_contract_steps) == 2
    assert all(ship._effective_timeout(step) <= 900 for step in backend_contract_steps)
    assert all(len(step.command) < 40 for step in backend_contract_steps)


def test_dependency_fingerprint_changes_with_relevant_source(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    step = ship.Step(name="fixture", command=[sys.executable, "source.py"])
    monkeypatch.setattr(
        ship, "_candidate_dependency_paths", lambda _step: [source]
    )
    first = ship._dependency_fingerprint(step)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = ship._dependency_fingerprint(step)
    assert first != second


def test_resume_reuses_only_matching_passed_fingerprint():
    step = ship.Step(name="fixture", command=[sys.executable, "-V"])
    previous = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "steps": [
            {
                "name": "fixture",
                "status": "passed",
                "dependency_fingerprint": "same",
            }
        ],
    }
    cached = ship._cached_result(previous, step, "same")
    assert cached is not None
    assert cached["status"] == "cached_pass"
    assert ship._cached_result(previous, step, "changed") is None


def test_nonpassing_step_is_never_resumed():
    step = ship.Step(name="fixture", command=[sys.executable, "-V"])
    previous = {
        "steps": [
            {
                "name": "fixture",
                "status": "failed",
                "dependency_fingerprint": "same",
            }
        ]
    }
    assert ship._cached_result(previous, step, "same") is None
