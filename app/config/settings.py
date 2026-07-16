from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config.profiles import RUNTIME_PROFILES, RuntimeProfile
from app.version import APP_NAME, APP_VERSION

_BALANCED_PROFILE_NAME = "balanced"
_RESOURCE_TIER_ORDER = ("constrained", "balanced", "high")


def _bytes_from_gb(value_gb: int) -> int:
    return int(value_gb) * 1024 * 1024 * 1024


def current_free_vram_bytes() -> int | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        return int(free_bytes)
    except Exception:
        return None


def _threshold_gb(tier_name: str, overrides: dict[str, int] | None) -> int:
    """Return the min-free-VRAM (GB) threshold for a tier, honoring optional pack overrides."""
    if overrides is not None and tier_name in overrides:
        return int(overrides[tier_name])
    return RUNTIME_PROFILES[tier_name].min_free_vram_gb


def detect_resource_tier_profile(
    *,
    free_vram_bytes: int | None = None,
    pack_thresholds: dict[str, int] | None = None,
) -> RuntimeProfile:
    """Pick a runtime profile from free-VRAM, optionally with per-pack threshold overrides.

    ``pack_thresholds`` maps tier names (``high``/``balanced``/``constrained``) to GB integers
    that override the corresponding ``RuntimeProfile.min_free_vram_gb``. Absent keys fall back
    to the global defaults. This lets a large model (e.g. Krea2's 12B DiT) require more free VRAM
    to select ``high`` than the default 12 GB the smaller Z-Image family uses.
    """
    free_bytes = current_free_vram_bytes() if free_vram_bytes is None else free_vram_bytes
    if free_bytes is None:
        return RUNTIME_PROFILES[_BALANCED_PROFILE_NAME]
    if free_bytes >= _bytes_from_gb(_threshold_gb("high", pack_thresholds)):
        return RUNTIME_PROFILES["high"]
    if free_bytes >= _bytes_from_gb(_threshold_gb("balanced", pack_thresholds)):
        return RUNTIME_PROFILES["balanced"]
    return RUNTIME_PROFILES["constrained"]


def _next_tier_name(current_name: str) -> str | None:
    if current_name not in _RESOURCE_TIER_ORDER:
        return None
    idx = _RESOURCE_TIER_ORDER.index(current_name)
    if idx >= len(_RESOURCE_TIER_ORDER) - 1:
        return None
    return _RESOURCE_TIER_ORDER[idx + 1]


def _upgrade_margin_gb(target_name: str) -> int:
    if target_name == "high":
        return 3
    if target_name == "balanced":
        return 2
    return 0


@dataclass
class ResourceTierController:
    current_profile: RuntimeProfile
    override_profile: RuntimeProfile | None = None
    consecutive_upgrade_hits: int = 0

    def current(self) -> RuntimeProfile:
        return self.override_profile or self.current_profile

    def current_for(self, pack: Any) -> RuntimeProfile:
        """Pick the runtime profile for a given pack, honoring per-pack threshold overrides.

        Respects any active user/env override (same as ``current()``); otherwise re-selects a
        tier against the pack's ``resource_tier_thresholds`` (see ``ModelPack``). Falls back to
        ``current()`` when the pack does not provide overrides or free-VRAM cannot be read.
        """
        if self.override_profile is not None:
            return self.override_profile
        thresholds = getattr(pack, "resource_tier_thresholds", None)
        if not thresholds:
            return self.current_profile
        return detect_resource_tier_profile(pack_thresholds=thresholds)

    def refresh(self) -> RuntimeProfile:
        if self.override_profile is not None:
            self.current_profile = self.override_profile
            self.consecutive_upgrade_hits = 0
            return self.override_profile

        free_bytes = current_free_vram_bytes()
        if free_bytes is None:
            return self.current_profile

        direct_target = detect_resource_tier_profile(free_vram_bytes=free_bytes)
        current_name = self.current_profile.name
        current_rank = _RESOURCE_TIER_ORDER.index(current_name)
        target_rank = _RESOURCE_TIER_ORDER.index(direct_target.name)

        if target_rank < current_rank:
            self.current_profile = direct_target
            self.consecutive_upgrade_hits = 0
            return self.current_profile

        if target_rank == current_rank:
            self.consecutive_upgrade_hits = 0
            return self.current_profile

        next_name = _next_tier_name(current_name)
        if next_name is None:
            self.consecutive_upgrade_hits = 0
            return self.current_profile

        next_profile = RUNTIME_PROFILES[next_name]
        promotion_threshold = _bytes_from_gb(
            next_profile.min_free_vram_gb + _upgrade_margin_gb(next_name)
        )
        if free_bytes >= promotion_threshold:
            self.consecutive_upgrade_hits += 1
            if self.consecutive_upgrade_hits >= 2:
                self.current_profile = next_profile
                self.consecutive_upgrade_hits = 0
        else:
            self.consecutive_upgrade_hits = 0
        return self.current_profile


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
    meta_debug: bool
    runtime_profile: RuntimeProfile
    resource_tier: RuntimeProfile
    resource_tier_override: str | None
    auto_resource_tier: bool
    resource_tier_controller: ResourceTierController
    paths: AppPaths

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "offline_mode": self.offline_mode,
            "meta_debug": self.meta_debug,
            "runtime_profile": {
                "name": self.runtime_profile.name,
                "description": self.runtime_profile.description,
            },
            "resource_tier": {
                "name": self.resource_tier_controller.current().name,
                "description": self.resource_tier_controller.current().description,
            },
            "resource_tier_override": self.resource_tier_override,
            "auto_resource_tier": self.auto_resource_tier,
            "paths": {
                "root_dir": str(self.paths.root_dir),
                "models_dir": str(self.paths.models_dir),
                "model_packs_dir": str(self.paths.model_packs_dir),
                "outputs_dir": str(self.paths.outputs_dir),
                "data_dir": str(self.paths.data_dir),
                "ui_dir": str(self.paths.ui_dir),
            },
        }


def _resolve_root() -> Path:
    env_root = os.getenv("JUSTRAYZIST_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _get_profile(profile_name: str | None) -> RuntimeProfile:
    resolved = (profile_name or os.getenv("JUSTRAYZIST_PROFILE") or _BALANCED_PROFILE_NAME).lower()
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

    override_profile: RuntimeProfile | None = None
    if profile_name or os.getenv("JUSTRAYZIST_PROFILE"):
        override_profile = _get_profile(profile_name)
    runtime_profile = override_profile or RUNTIME_PROFILES[_BALANCED_PROFILE_NAME]
    resource_tier = override_profile or detect_resource_tier_profile()
    resource_tier_controller = ResourceTierController(
        current_profile=resource_tier,
        override_profile=override_profile,
    )
    offline_mode = os.getenv("JUSTRAYZIST_OFFLINE", "1") == "1"
    if offline_mode:
        enforce_offline_runtime()

    return AppSettings(
        app_name=APP_NAME,
        app_version=APP_VERSION,
        environment=os.getenv("JUSTRAYZIST_ENV", "dev"),
        offline_mode=offline_mode,
        meta_debug=os.getenv("JUSTRAYZIST_METADEBUG", "0") == "1",
        runtime_profile=runtime_profile,
        resource_tier=resource_tier,
        resource_tier_override=override_profile.name if override_profile is not None else None,
        auto_resource_tier=override_profile is None,
        resource_tier_controller=resource_tier_controller,
        paths=paths,
    )
