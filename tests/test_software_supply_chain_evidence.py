from __future__ import annotations

import json

from backend.services.software_supply_chain_evidence import (
    build_software_supply_chain_evidence,
)


def test_supply_chain_evidence_emits_sboms_without_secret_values(tmp_path) -> None:
    (tmp_path / "requirements-lock-py314-win.txt").write_text(
        "FastAPI==1.2.3 --hash=sha256:abc\nnumpy==2.0.0\n",
        encoding="utf-8",
    )
    frontend = tmp_path / "frontend-react"
    frontend.mkdir()
    (frontend / "package-lock.json").write_text(json.dumps({
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "nlcare"},
            "node_modules/react": {"name": "react", "version": "19.0.0"},
        },
    }), encoding="utf-8")
    (tmp_path / "app.py").write_text("print('safe')\n", encoding="utf-8")

    result = build_software_supply_chain_evidence(
        root=tmp_path,
        output_path="out.json",
        sbom_dir="sbom",
    )

    assert result["status"] == "acceptable"
    assert result["sbom"]["component_count"] == 3
    assert result["secret_scan"]["finding_count"] == 0
    assert result["secret_scan"]["secret_values_included"] is False
    assert result["container_scan"]["executed"] is False
    assert result["clinical_validation"] is False
    python_sbom = json.loads((tmp_path / "sbom/python.cdx.json").read_text())
    assert python_sbom["bomFormat"] == "CycloneDX"
    assert {row["name"] for row in python_sbom["components"]} == {"FastAPI", "numpy"}


def test_supply_chain_evidence_reports_location_but_not_secret_value(tmp_path) -> None:
    (tmp_path / "requirements-lock-py314-win.txt").write_text(
        "FastAPI==1.2.3\n",
        encoding="utf-8",
    )
    frontend = tmp_path / "frontend-react"
    frontend.mkdir()
    (frontend / "package-lock.json").write_text(
        json.dumps({"lockfileVersion": 3, "packages": {}}),
        encoding="utf-8",
    )
    secret_value = "abcdefghijklmnopqrstuvwxyz123456"
    (tmp_path / "bad.py").write_text(
        f"api_key = '{secret_value}'\n",
        encoding="utf-8",
    )

    result = build_software_supply_chain_evidence(
        root=tmp_path,
        output_path="out.json",
        sbom_dir="sbom",
    )

    assert result["status"] == "needs_attention"
    assert result["secret_scan"]["finding_count"] == 1
    encoded = json.dumps(result)
    assert secret_value not in encoded
    assert result["secret_scan"]["findings"][0]["path"] == "bad.py"
