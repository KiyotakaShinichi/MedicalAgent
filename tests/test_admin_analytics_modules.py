"""Each admin analytics responsibility works through its own module.

`admin_analytics` was a single 928-line module that loaded every artifact,
built every dashboard panel, and computed every statistic. It is now a facade
over three modules — readiness (inputs), panels, and summary (statistics) — and
this file covers each directly so a failure names the responsibility rather
than the file that used to hold all of them.

The property that matters most here is that the dashboard payload did not
change. These panels are read by people making engineering judgements about
model behaviour; a split that silently altered a number would be worse than no
split at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models import Base  # noqa: E402
from backend.services import admin_analytics as facade  # noqa: E402
from backend.services import (  # noqa: E402
    admin_analytics_panels,
    admin_analytics_readiness,
    admin_analytics_summary,
)

ORIGINAL_PATH = "backend/services/admin_analytics.py"

RESPONSIBILITY_MODULES = (
    "backend.services.admin_analytics_readiness",
    "backend.services.admin_analytics_panels",
    "backend.services.admin_analytics_summary",
)


@pytest.fixture(scope="module")
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def payload(db):
    return facade.build_admin_analytics(db)


# ─── the facade contract ─────────────────────────────────────────────────────


def test_facade_exports_every_pre_split_symbol(tmp_path) -> None:
    """Compare real module attributes, not just AST definitions.

    An earlier version of this test parsed the original source and checked only
    the names it *defined*. That missed everything the module imported at module
    scope — and the breast-monitoring suite monkeypatches
    `admin_analytics._calibration_metrics`, which was one of them. The split
    passed this test and broke that suite.
    """
    import importlib
    import importlib.util
    import subprocess
    import sys as _sys

    original_source = subprocess.run(
        ["git", "show", f"HEAD:{ORIGINAL_PATH}"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    original_file = tmp_path / "original_module.py"
    original_file.write_text(original_source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("_pre_split_original", original_file)
    original = importlib.util.module_from_spec(spec)
    _sys.modules["_pre_split_original"] = original
    spec.loader.exec_module(original)

    before = {name for name in dir(original) if not name.startswith("__")}
    after = {name for name in dir(facade) if not name.startswith("__")}

    missing = sorted(before - after)
    assert not missing, (
        f"the split dropped module attributes callers could bind to: {missing}"
    )


def test_public_entrypoint_is_unchanged(payload) -> None:
    assert facade.__all__ == ["build_admin_analytics"]
    assert isinstance(payload, dict) and payload


@pytest.mark.parametrize("module", RESPONSIBILITY_MODULES)
def test_each_module_imports_standalone(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{module} failed to import alone:\n{result.stderr}"


@pytest.mark.parametrize("module", RESPONSIBILITY_MODULES)
def test_each_module_is_within_the_service_limit(module: str) -> None:
    path = ROOT / (module.replace(".", "/") + ".py")
    loc = len(path.read_bytes().decode("utf-8").splitlines())
    assert loc <= 500, f"{path.name} is {loc} LOC"


def test_facade_is_within_the_service_limit() -> None:
    loc = len((ROOT / "backend/services/admin_analytics.py").read_bytes().decode("utf-8").splitlines())
    assert loc <= 500, f"the facade is {loc} LOC"


# ─── payload equivalence ─────────────────────────────────────────────────────


EXPECTED_KEY_ORDER_HEAD = [
    "roles",
    "model_performance",
    "evidence_separation",
    "metric_interpretation_guide",
    "advanced_model_evaluation",
    "confusion_matrix",
]


def test_payload_key_order_is_preserved(payload) -> None:
    """The dashboard renders sections in payload order, so order is contractual."""
    assert list(payload)[: len(EXPECTED_KEY_ORDER_HEAD)] == EXPECTED_KEY_ORDER_HEAD


def test_payload_has_every_expected_panel(payload) -> None:
    for key in (
        "model_performance", "evidence_separation", "advanced_model_evaluation",
        "drift_monitoring", "ab_testing", "audit_and_feedback", "data_quality",
        "data_coverage", "mle_readiness", "rag_evaluation", "guardrails",
        "clinician_loop_metrics", "domain_gap_analysis", "safety_positioning",
    ):
        assert key in payload, f"panel {key} disappeared from the payload"


def test_payload_is_json_serialisable(payload) -> None:
    """It is returned over HTTP; an unserialisable value fails at the boundary."""
    assert json.loads(json.dumps(payload, default=str))


def test_safety_positioning_still_denies_clinical_use(payload) -> None:
    """The panel set must not read as a clinical decision surface."""
    positioning = payload["safety_positioning"].lower()
    assert "do not diagnose" in positioning
    assert "monitoring only" in positioning


# ─── readiness / inputs ──────────────────────────────────────────────────────


def test_missing_artifacts_degrade_rather_than_raise(tmp_path: Path) -> None:
    """An absent optional artifact must not 500 the dashboard."""
    assert admin_analytics_readiness._load_json(tmp_path / "absent.json") is None
    assert admin_analytics_readiness._load_csv(tmp_path / "absent.csv") is None


def test_readiness_artifact_is_passed_through_unmodified(payload) -> None:
    """The panel reports the artifact; it does not recompute or soften it.

    Whatever status the readiness evaluation recorded — including an unflattering
    one — is what the dashboard shows.
    """
    readiness = payload["mle_readiness"]
    assert isinstance(readiness, dict) and readiness
    on_disk = admin_analytics_readiness._load_json(
        admin_analytics_readiness.DEFAULT_MLE_READINESS_PATH
    )
    if on_disk is not None:
        assert readiness == on_disk, "the panel altered the readiness artifact"


def test_readiness_falls_back_without_claiming_readiness(monkeypatch, db) -> None:
    """With no artifact, the panel must say so rather than imply readiness."""
    monkeypatch.setattr(facade, "_load_json", lambda path: None)
    readiness = facade.build_admin_analytics(db)["mle_readiness"]

    assert readiness["status"] == "unavailable"
    assert readiness["clinical_validation"] is False
    assert "No precomputed MLE readiness artifact" in readiness["message"]


def test_input_paths_are_declared_once(payload) -> None:
    """Every artifact path lives in the readiness module, not scattered."""
    for name in (
        "DEFAULT_SYNTHETIC_METRICS_PATH", "DEFAULT_SYNTHETIC_PREDICTIONS_PATH",
        "DEFAULT_SYNTHETIC_TRAINING_CSV", "DEFAULT_SYNTHETIC_MRI_REPORTS_CSV",
        "DEFAULT_BREASTDCEDL_METRICS_PATH", "DEFAULT_MLE_READINESS_PATH",
    ):
        assert hasattr(admin_analytics_readiness, name)
        assert getattr(facade, name) == getattr(admin_analytics_readiness, name)


# ─── panels ──────────────────────────────────────────────────────────────────


def test_panels_handle_absent_data_without_raising() -> None:
    """Panels receive `None` when their input artifact is missing."""
    assert isinstance(admin_analytics_panels._model_performance(None, None), dict)
    assert isinstance(admin_analytics_panels._drift_monitoring(None), dict)
    assert isinstance(admin_analytics_panels._data_quality(None), dict)
    assert isinstance(admin_analytics_panels._data_coverage(None), dict)


def test_guardrail_panel_summarises_without_a_regression_report() -> None:
    summary = admin_analytics_panels._frontend_guardrail_summary({}, None)
    assert isinstance(summary, dict)


# ─── summary / statistics ────────────────────────────────────────────────────


def test_advanced_evaluation_is_present_and_structured(payload) -> None:
    assert isinstance(payload["advanced_model_evaluation"], dict)


@pytest.mark.parametrize("metric", ["AUROC", "AUPRC", "Brier"])
def test_metric_estimate_computes_each_supported_metric(metric: str) -> None:
    """One estimator for the three metrics the bootstrap resamples."""
    labels = [0, 0, 1, 1, 0, 1]
    probabilities = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
    value = admin_analytics_summary._metric_estimate(metric, labels, probabilities)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_bootstrap_intervals_accompany_the_point_estimates() -> None:
    """Intervals are what stop a saturated synthetic score being over-read."""
    import numpy as np

    labels = np.array([0, 1] * 25)
    probabilities = np.linspace(0.01, 0.99, 50)
    result = admin_analytics_summary._bootstrap_confidence_intervals(labels, probabilities)
    assert isinstance(result, dict) and result


def test_percentile_helpers_are_reachable_from_the_summary_module() -> None:
    for name in (
        "_bootstrap_confidence_intervals", "_decision_curve",
        "_threshold_operating_points", "_cost_sensitive_thresholds",
        "_decision_impact_simulation", "_subgroup_performance",
        "_false_negative_review",
    ):
        assert hasattr(admin_analytics_summary, name), f"{name} is missing"


def test_modules_do_not_import_the_facade() -> None:
    """A back-import would recreate the cycle the split removed."""
    for module in RESPONSIBILITY_MODULES:
        source = (ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert "from backend.services.admin_analytics import" not in source
