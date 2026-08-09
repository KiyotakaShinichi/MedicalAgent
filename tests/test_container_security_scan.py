from __future__ import annotations

from backend.services.container_security_scan import (
    _dockerfile_supply_chain,
    summarize_trivy_result,
)


def _raw(*findings: dict) -> dict:
    return {
        "SchemaVersion": 2,
        "ArtifactName": "nlcare:test",
        "ArtifactType": "container_image",
        "Metadata": {"ImageID": "sha256:abc"},
        "Results": [{"Target": "image", "Vulnerabilities": list(findings)}],
    }


def _inspect(*, user: str = "nonroot:nonroot", image_id: str = "sha256:abc") -> dict:
    return {"Id": image_id, "Config": {"User": user}}


def test_critical_or_fixable_high_blocks_public_deployment() -> None:
    report = summarize_trivy_result(_raw(
        {"VulnerabilityID": "CVE-C", "Severity": "CRITICAL", "PkgName": "a"},
        {"VulnerabilityID": "CVE-H", "Severity": "HIGH", "PkgName": "b", "FixedVersion": "2"},
    ), image_inspect=_inspect(), scanner_version="Version: 1")

    assert report["status"] == "blocked"
    assert report["summary"]["severity_counts"]["critical"] == 1
    assert report["summary"]["fixable_high_or_critical_count"] == 1
    assert report["deployment_decision"] == "BLOCK_PUBLIC_DEPLOYMENT"
    assert report["clinical_validation"] is False


def test_unfixed_high_remains_needs_attention() -> None:
    report = summarize_trivy_result(_raw(
        {"VulnerabilityID": "CVE-H", "Severity": "HIGH", "PkgName": "a"},
    ), image_inspect=_inspect())

    assert report["status"] == "needs_attention"
    assert report["summary"]["unfixed_high_or_critical_count"] == 1
    assert report["summary"]["public_deployment_blocked"] is True


def test_clean_nonroot_matching_image_is_acceptable() -> None:
    report = summarize_trivy_result(_raw(), image_inspect=_inspect())

    assert report["status"] == "acceptable"
    assert report["image"]["runs_as_nonroot"] is True
    assert report["image"]["identity_matches_current_image"] is True


def test_root_runtime_is_blocked_even_without_findings() -> None:
    report = summarize_trivy_result(_raw(), image_inspect=_inspect(user="root"))
    assert report["status"] == "blocked"


def test_stale_scan_cannot_be_reused_for_new_image() -> None:
    report = summarize_trivy_result(_raw(), image_inspect=_inspect(image_id="sha256:new"))
    assert report["status"] == "stale_image_mismatch"
    assert report["image"]["identity_matches_current_image"] is False


def test_mutable_base_image_blocks_clean_scan() -> None:
    report = summarize_trivy_result(
        _raw(),
        image_inspect=_inspect(),
        supply_chain={"all_base_images_digest_pinned": False},
        sbom={"available": True},
    )
    assert report["status"] == "blocked"
    assert report["summary"]["base_images_digest_pinned"] is False


def test_dockerfile_supply_chain_requires_sha256_digest(tmp_path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.13@sha256:" + "a" * 64 + " AS builder\n"
        "FROM example/runtime@sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )
    pinned = _dockerfile_supply_chain(dockerfile)
    assert pinned["all_base_images_digest_pinned"] is True

    dockerfile.write_text("FROM python:3.13\n", encoding="utf-8")
    mutable = _dockerfile_supply_chain(dockerfile)
    assert mutable["all_base_images_digest_pinned"] is False
