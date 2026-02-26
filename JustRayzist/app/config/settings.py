from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config.profiles import RUNTIME_PROFILES, RuntimeProfile
from app.version import APP_NAME, APP_VERSION


@dataclass(frozen=True)
class AppPaths:
    root_dir: Path
    models_dir: Path
    model_packs_dir: Path
    outputs_dir: Path
    data_dir: Path
    ui_dir: Path


@dataclass(frozen=True)
class AppSettings:
    app_name: str
    app_version: str
    environment: str
    offline_mode: bool
    runtime_profile: RuntimeProfile
    paths: AppPaths

    def to_dict(self) -> dict:
        data = asdict(self)
        data["runtime_profile"] = asdict(self.runtime_profile)
        data["paths"] = {key: str(value) for key, value in data["paths"].items()}
        return data


def _resolve_root() -> Path:
    env_root = os.getenv("JUSTRAYZIST_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _get_profile(profile_name: str | None) -> RuntimeProfile:
    resolved = (profile_name or os.getenv("JUSTRAYZIST_PROFILE") or "balanced").lower()
    if resolved not in RUNTIME_PROFILES:
        allowed = ", ".join(sorted(RUNTIME_PROFILES.keys()))
        raise ValueError(f"Invalid profile '{resolved}'. Allowed values: {allowed}.")
    return RUNTIME_PROFILES[resolved]


def _ensure_directories(paths: AppPaths) -> None:
    for directory in (paths.models_dir, paths.model_packs_dir, paths.outputs_dir, paths.data_dir):
        directory.mkdir(parents=True, exist_ok=True)


def enforce_offline_runtime() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_settings(profile_name: str | None = None) -> AppSettings:
    root_dir = _resolve_root()
    paths = AppPaths(
        root_dir=root_dir,
        models_dir=root_dir / "models",
        model_packs_dir=root_dir / "models" / "packs",
        outputs_dir=root_dir / "outputs",
        data_dir=root_dir / "data",
        ui_dir=root_dir / "app" / "ui",
    )
    _ensure_directories(paths)

    profile = _get_profile(profile_name)
    offline_mode = os.getenv("JUSTRAYZIST_OFFLINE", "1") == "1"
    if offline_mode:
        enforce_offline_runtime()

    return AppSettings(
        app_name=APP_NAME,
        app_version=APP_VERSION,
        environment=os.getenv("JUSTRAYZIST_ENV", "dev"),
        offline_mode=offline_mode,
        runtime_profile=profile,
        paths=paths,
    )

