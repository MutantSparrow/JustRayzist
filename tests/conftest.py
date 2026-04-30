from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from app.config.profiles import RUNTIME_PROFILES
from app.config.settings import AppPaths, AppSettings, ResourceTierController

# All tests run with full metadata so assertions on PNG chunks are not broken by filtering.
os.environ.setdefault("JUSTRAYZIST_METADEBUG", "1")


@pytest.fixture
def workspace_tmp_path() -> Path:
    base_dir = Path.cwd() / ".build" / "pytest-runtime"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"case_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def temp_app_root(workspace_tmp_path: Path) -> Path:
    root = workspace_tmp_path / "justrayzist"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def temp_app_paths(temp_app_root: Path) -> AppPaths:
    paths = AppPaths(
        root_dir=temp_app_root,
        models_dir=temp_app_root / "models",
        model_packs_dir=temp_app_root / "models" / "packs",
        outputs_dir=temp_app_root / "outputs",
        data_dir=temp_app_root / "data",
        ui_dir=temp_app_root / "app" / "ui",
    )
    for directory in (
        paths.models_dir,
        paths.model_packs_dir,
        paths.outputs_dir,
        paths.data_dir,
        paths.ui_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture
def make_app_settings():
    def _make(
        *,
        paths: AppPaths,
        runtime_profile_name: str = "balanced",
        resource_tier_name: str | None = None,
        override_profile_name: str | None = None,
        auto_resource_tier: bool | None = None,
    ) -> AppSettings:
        runtime_profile = RUNTIME_PROFILES[runtime_profile_name]
        resource_profile = RUNTIME_PROFILES[resource_tier_name or runtime_profile_name]
        override_profile = (
            RUNTIME_PROFILES[override_profile_name]
            if override_profile_name is not None
            else None
        )
        return AppSettings(
            app_name="JustRayzist",
            app_version="test",
            environment="test",
            offline_mode=True,
            meta_debug=True,
            runtime_profile=runtime_profile,
            resource_tier=resource_profile,
            resource_tier_override=override_profile.name if override_profile is not None else None,
            auto_resource_tier=(
                auto_resource_tier if auto_resource_tier is not None else override_profile is None
            ),
            resource_tier_controller=ResourceTierController(
                current_profile=resource_profile,
                override_profile=override_profile,
            ),
            paths=paths,
        )

    return _make
