"""Each governance artifact is built by its own module.

`governance_credibility_artifacts` was a single 957-line module holding four
unrelated artifact builders. It is now a facade over one module per artifact
domain, and this file covers each domain directly so a failure names the
artifact rather than the file that used to contain all of them.

The invariant these artifacts exist to protect is anti-overclaim: every payload
must carry a claim boundary containing "not clinical validation" verbatim, and
the portfolio artifact must keep its banned-phrase list. A split that silently
dropped either would leave the artifacts looking fine while removing the only
thing that stops them reading as clinical evidence.

`tests/test_governance_credibility_artifacts.py` keeps the detailed payload
assertions; this file covers per-domain isolation and the shared invariant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import governance_credibility_artifacts as facade  # noqa: E402
from backend.services.governance_artifacts import (  # noqa: E402
    contamination_harmonization,
    negative_results,
    noisier_v2_readiness,
    portfolio_claim_safety,
)
from backend.services.governance_artifacts.common import (  # noqa: E402
    REQUIRED_CLAIM_BOUNDARY_PHRASE,
)

DOMAIN_MODULES = (
    negative_results,
    portfolio_claim_safety,
    contamination_harmonization,
    noisier_v2_readiness,
)

BUILDERS = {
    "negative_results": negative_results.build_negative_results_gallery,
    "portfolio_claim_safety": portfolio_claim_safety.build_portfolio_claim_safety_check,
    "contamination_harmonization": (
        contamination_harmonization.build_eval_contamination_harmonization
    ),
    "noisier_v2_readiness": noisier_v2_readiness.build_noisier_synthetic_v2_readiness,
}


# ─── the shared anti-overclaim invariant ─────────────────────────────────────


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_every_artifact_carries_the_claim_boundary(domain: str) -> None:
    payload = BUILDERS[domain]()
    boundary = json.dumps(payload).lower()
    assert REQUIRED_CLAIM_BOUNDARY_PHRASE in boundary, (
        f"{domain} lost the '{REQUIRED_CLAIM_BOUNDARY_PHRASE}' boundary"
    )


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_no_artifact_claims_clinical_validation(domain: str) -> None:
    payload = BUILDERS[domain]()
    assert payload.get("clinical_validation") is False


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_every_artifact_is_json_serialisable(domain: str) -> None:
    """These are written to disk verbatim; an unserialisable field fails late."""
    assert json.loads(json.dumps(BUILDERS[domain]()))


# ─── negative results ────────────────────────────────────────────────────────


def test_negative_results_gallery_actually_lists_negatives() -> None:
    """An empty gallery would pass every structural check and mean nothing."""
    payload = negative_results.build_negative_results_gallery()
    serialised = json.dumps(payload).lower()
    assert "bm25" in serialised, "the RAG negative result is the headline one"
    assert len(serialised) > 2000, "gallery is suspiciously thin"


# ─── portfolio claim safety ──────────────────────────────────────────────────


def test_portfolio_check_keeps_its_banned_phrases() -> None:
    banned = " ".join(portfolio_claim_safety.BANNED_AFFIRMATIVE_PHRASES).lower()
    assert banned, "the banned-phrase list is what makes this artifact useful"
    assert "clinical" in banned or "fda" in banned or "diagnos" in banned


def test_portfolio_banned_and_allowed_phrases_are_disjoint() -> None:
    """A phrase in both lists would make the check unresolvable."""
    banned = {p.strip().lower() for p in portfolio_claim_safety.BANNED_AFFIRMATIVE_PHRASES}
    allowed = {p.strip().lower() for p in portfolio_claim_safety.ALLOWED_PHRASES}
    assert not (banned & allowed)


# ─── contamination harmonization ─────────────────────────────────────────────


def test_contamination_report_assigns_a_category_to_every_entry() -> None:
    payload = contamination_harmonization.build_eval_contamination_harmonization()
    assert payload.get("categories"), "category list is the artifact's whole point"


# ─── noisier v2 readiness ────────────────────────────────────────────────────


def test_noisier_v2_scaffold_status_stays_within_the_allowed_set() -> None:
    """`scaffold_status` must not drift to something implying a dataset exists.

    Note this is `scaffold_status`, not `status`: `status` is the release-gate
    classification (`informational`), while `scaffold_status` is the claim about
    whether the dataset and model actually exist. Only the latter is
    constrained by `ALLOWED_NOISIER_V2_STATUS`.
    """
    payload = noisier_v2_readiness.build_noisier_synthetic_v2_readiness()
    assert payload["scaffold_status"] in noisier_v2_readiness.ALLOWED_NOISIER_V2_STATUS


def test_noisier_v2_does_not_claim_a_retrained_model() -> None:
    """The artifact plans a dataset; it must not read as having produced one."""
    payload = noisier_v2_readiness.build_noisier_synthetic_v2_readiness()
    assert payload.get("model_retrained", False) is False
    assert "no model has been retrained" in payload["claim_boundary"].lower()


# ─── module structure ────────────────────────────────────────────────────────


@pytest.mark.parametrize("module", DOMAIN_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_each_domain_module_is_under_the_service_limit(module) -> None:
    loc = len(Path(module.__file__).read_text(encoding="utf-8").splitlines())
    assert loc <= 500, f"{Path(module.__file__).name} is {loc} LOC"


@pytest.mark.parametrize("module", DOMAIN_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_each_domain_declares_its_exports(module) -> None:
    assert module.__all__, f"{module.__name__} declares no __all__"
    for name in module.__all__:
        assert hasattr(module, name)


def test_facade_re_exports_every_domain_symbol() -> None:
    """`from backend.services.governance_credibility_artifacts import ...` must keep working."""
    for module in DOMAIN_MODULES:
        for name in module.__all__:
            assert hasattr(facade, name), f"facade dropped {name}"
            assert getattr(facade, name) is getattr(module, name), (
                f"{name} is a copy on the facade, not a re-export"
            )


def test_domains_do_not_import_each_other() -> None:
    """They share only the claim-boundary invariant.

    Cross-imports would re-couple the artifacts the split just separated, and
    an edit to one would again be able to change another's payload.
    """
    for module in DOMAIN_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        for other in DOMAIN_MODULES:
            if other is module:
                continue
            leaf = other.__name__.rsplit(".", 1)[-1]
            assert f"import {leaf}" not in source, (
                f"{module.__name__} imports sibling domain {leaf}"
            )


def test_artifact_output_paths_are_distinct() -> None:
    paths = {
        facade.NEGATIVE_RESULTS_PATH,
        facade.PORTFOLIO_PATH,
        facade.CONTAMINATION_PATH,
        facade.NOISIER_V2_PATH,
    }
    assert len(paths) == 4, "two artifacts would overwrite each other"


def test_write_helpers_persist_to_a_supplied_path(tmp_path: Path) -> None:
    """Each domain writes its own artifact and returns where it landed."""
    writers = {
        "negative": (negative_results.write_negative_results_gallery, "negative.json"),
        "portfolio": (portfolio_claim_safety.write_portfolio_claim_safety_check, "portfolio.json"),
        "contamination": (
            contamination_harmonization.write_eval_contamination_harmonization,
            "contamination.json",
        ),
        "noisier": (
            noisier_v2_readiness.write_noisier_synthetic_v2_readiness,
            "noisier.json",
        ),
    }
    for label, (writer, filename) in writers.items():
        target = tmp_path / filename
        written = writer(target)
        assert written == target, f"{label} writer returned {written}"
        assert json.loads(target.read_text(encoding="utf-8")), f"{label} wrote empty JSON"
