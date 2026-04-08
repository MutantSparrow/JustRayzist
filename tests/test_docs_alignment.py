from __future__ import annotations

from pathlib import Path

import yaml

from app.api import main as api_main
from app.api.api_manifest import API_EXAMPLES
from app.version import APP_VERSION
from scripts.render_api_docs import render_readme, render_usage


ROOT = Path(__file__).resolve().parents[1]
_EXCLUDED_ROUTE_PATHS = {
    "/",
    "/api",
    "/API",
    "/favicon.ico",
    "/api-manifest",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
}


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_modelpack_example_matches_current_schema() -> None:
    raw = _read("models/packs/modelpack.yaml.example")
    payload = yaml.safe_load(raw)

    assert payload["name"] == "my_z_turbo_pack"
    assert payload["user_visible"] is True
    assert payload["enabled"] is True
    assert payload["components"]["transformer"]["format"] == "safetensors"
    assert payload["components"]["vae"]["format"] == "safetensors"
    assert payload["components"]["text_encoder"]["format"] == "gguf"
    assert "storage_mode" in raw
    assert "storage_dtype" in raw
    assert "compute_dtype" in raw


def test_user_docs_describe_auto_resource_tiering() -> None:
    readme = _read("README.md")
    usage = _read("docs/USAGE.md")
    troubleshooting = _read("docs/TROUBLESHOOTING.md")
    packaging = _read("docs/PACKAGING.md")
    release_readme = _read("scripts/release/README.md")

    for text in (readme, usage, troubleshooting):
        assert "--profile balanced" not in text
        assert "--profile constrained" not in text

    assert f"New in v{APP_VERSION}" in readme
    assert "resource_tier" in readme
    assert "resource_tier" in usage
    assert "auto resource-tiering" in troubleshooting.lower()
    assert "RunMeFirst.sh" in readme
    assert "StartWeb.sh" in readme
    assert "Linux and macOS source mode" in readme
    assert "RunMeFirst.sh" in usage
    assert "StartWeb.sh" in usage
    assert "Windows-only" in packaging
    assert "JUSTRAYZIST_PROFILE" in readme
    assert "native FP8 inference is not implemented" in readme
    assert "native FP8 inference is not implemented" in usage
    assert "Rayzist_fp8_mixed" not in readme
    assert "Rayzist_fp8_mixed" not in usage
    assert "engineering-only" in usage
    assert "pack-compare" in usage
    assert "pack-compare-suite" in usage
    assert "prompt-grid-benchmark" in usage
    assert "public enabled pack" in packaging.lower()
    assert "public enabled pack selection" in release_readme.lower()


def test_pack_docs_explain_hidden_and_derived_pack_behavior() -> None:
    pack_docs = _read("models/packs/README.md")

    assert "user_visible" in pack_docs
    assert "enabled" in pack_docs
    assert "auto_fp8_storage" in pack_docs
    assert "Rayzist_bf16" in pack_docs
    assert "does not provide native FP8 inference" in pack_docs
    assert "should not create duplicate packs" in pack_docs
    assert "public enabled packs" in pack_docs


def test_api_manifest_covers_all_user_facing_routes() -> None:
    manifest_routes = {
        (entry.method, entry.path.split("?", 1)[0])
        for entry in API_EXAMPLES
    }
    app_routes = set()
    for route in api_main.app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or path in _EXCLUDED_ROUTE_PATHS or not methods:
            continue
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            app_routes.add((method, path))

    assert manifest_routes == app_routes


def test_generated_api_docs_are_in_sync() -> None:
    readme = _read("README.md")
    usage = _read("docs/USAGE.md")

    assert render_readme(readme) == readme
    assert render_usage(usage) == usage


def test_api_tester_uses_manifest_feed() -> None:
    api_docs = _read("app/ui/api.js")

    assert "API_MANIFEST_PATH" in api_docs
    assert 'fetch(API_MANIFEST_PATH' in api_docs
    assert "requires_client" in api_docs





