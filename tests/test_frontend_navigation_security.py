from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_navigation_has_no_external_or_request_derived_targets() -> None:
    source_root = ROOT / "frontend-react/src"
    router_sources = [
        path.read_text(encoding="utf-8")
        for path in source_root.rglob("*")
        if path.suffix in {".ts", ".tsx"}
        and "react-router-dom" in path.read_text(encoding="utf-8")
    ]
    source = "\n".join(router_sources)

    assert router_sources
    assert "window.location" not in source
    assert "document.location" not in source
    assert "URLSearchParams" not in source
    assert "redirectTo" not in source


def test_frontend_is_client_only_and_has_no_ssr_or_rsc_router() -> None:
    source_root = ROOT / "frontend-react/src"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.rglob("*")
        if path.suffix in {".ts", ".tsx"}
    )

    for marker in (
        "createStaticRouter",
        "StaticRouterProvider",
        "HydratedRouter",
        "ServerRouter",
        "react-server-dom",
    ):
        assert marker not in source
