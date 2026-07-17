"""Per-pack resource-tier thresholds — parser + selector.

Packs can override the default min-free-VRAM (GB) thresholds that decide which runtime profile
(``high``/``balanced``/``constrained``) the auto-tier selector picks. Large models like Krea2
demand more free VRAM to skip sequential CPU offload; small models keep the smaller Z-Image
default.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from app.config.profiles import RUNTIME_PROFILES
from app.config.settings import (
    ResourceTierController,
    detect_resource_tier_profile,
)
from app.core.model_registry import load_model_pack
from app.core.model_registry.model_pack import ModelPackValidationError


def _bytes(gb: int) -> int:
    return gb * 1024 * 1024 * 1024


# --- Selector ---


def test_detect_falls_back_to_globals_without_pack_overrides() -> None:
    # Default global 'high' threshold is 12 GB; 20 GB free should select 'high'.
    profile = detect_resource_tier_profile(free_vram_bytes=_bytes(20))
    assert profile.name == "high"


def test_detect_honors_pack_high_threshold() -> None:
    # Krea2 requires 22 GB free for 'high'. At 20 GB the auto-selector should drop to 'balanced'
    # (14 GB pack threshold), matching the RTX 4090 offload/no-offload finding.
    thresholds = {"high": 22, "balanced": 14, "constrained": 4}
    profile = detect_resource_tier_profile(free_vram_bytes=_bytes(20), pack_thresholds=thresholds)
    assert profile.name == "balanced"


def test_detect_promotes_to_high_when_headroom_meets_pack_threshold() -> None:
    thresholds = {"high": 22, "balanced": 14, "constrained": 4}
    profile = detect_resource_tier_profile(free_vram_bytes=_bytes(24), pack_thresholds=thresholds)
    assert profile.name == "high"


def test_detect_drops_to_constrained_below_all_pack_thresholds() -> None:
    thresholds = {"high": 22, "balanced": 14, "constrained": 4}
    profile = detect_resource_tier_profile(free_vram_bytes=_bytes(8), pack_thresholds=thresholds)
    assert profile.name == "constrained"


# --- Controller wiring ---


def test_controller_current_for_pack_uses_overrides(monkeypatch) -> None:
    """Bench-anchored: 20 GB free → without overrides selects 'high'; Krea's override → 'balanced'."""

    monkeypatch.setattr(
        "app.config.settings.current_free_vram_bytes", lambda: _bytes(20)
    )
    controller = ResourceTierController(current_profile=RUNTIME_PROFILES["balanced"])

    # No pack overrides: sticks with whatever current() reports (i.e. the initial profile).
    plain_pack = SimpleNamespace(resource_tier_thresholds=None)
    assert controller.current_for(plain_pack).name == "balanced"

    # Krea-style thresholds: 20 GB is below the pack's 'high' bar → falls back to 'balanced'.
    krea_pack = SimpleNamespace(
        resource_tier_thresholds={"high": 22, "balanced": 14, "constrained": 4}
    )
    assert controller.current_for(krea_pack).name == "balanced"


def test_controller_current_for_pack_selects_high_when_headroom(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.config.settings.current_free_vram_bytes", lambda: _bytes(23)
    )
    controller = ResourceTierController(current_profile=RUNTIME_PROFILES["balanced"])
    krea_pack = SimpleNamespace(
        resource_tier_thresholds={"high": 22, "balanced": 14, "constrained": 4}
    )
    assert controller.current_for(krea_pack).name == "high"


def test_controller_override_still_wins(monkeypatch) -> None:
    """A user/env override must not be second-guessed by pack thresholds."""

    monkeypatch.setattr(
        "app.config.settings.current_free_vram_bytes", lambda: _bytes(4)
    )
    controller = ResourceTierController(
        current_profile=RUNTIME_PROFILES["constrained"],
        override_profile=RUNTIME_PROFILES["high"],
    )
    krea_pack = SimpleNamespace(
        resource_tier_thresholds={"high": 22, "balanced": 14, "constrained": 4}
    )
    assert controller.current_for(krea_pack).name == "high"


# --- Pack parser ---


def _write_pack(tmp_path: Path, thresholds_yaml: str) -> Path:
    pack_dir = tmp_path / "TestPack"
    pack_dir.mkdir()
    weights_dir = pack_dir / "weights"
    weights_dir.mkdir()
    (weights_dir / "transformer.safetensors").write_bytes(b"ok")
    (weights_dir / "vae.safetensors").write_bytes(b"ok")
    (weights_dir / "text_encoder.safetensors").write_bytes(b"ok")

    config_dir = pack_dir / "config"
    config_dir.mkdir()
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    manifest = f"""\
name: TestPack
architecture: krea2_turbo
backend_preference:
  - fp8_krea
pipeline_config_dir: ./config
components:
  transformer:
    path: ./weights/transformer.safetensors
    format: safetensors
  vae:
    path: ./weights/vae.safetensors
    format: safetensors
  text_encoder:
    path: ./weights/text_encoder.safetensors
    format: safetensors
required_configs:
  - ./config/model_index.json
{thresholds_yaml}
"""
    manifest_path = pack_dir / "modelpack.yaml"
    manifest_path.write_text(manifest, encoding="utf-8")
    return manifest_path


def test_pack_without_thresholds_defaults_to_none(tmp_path: Path) -> None:
    path = _write_pack(tmp_path, thresholds_yaml="")
    pack = load_model_pack(path)
    assert pack.resource_tier_thresholds is None


def test_pack_with_thresholds_parsed(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        thresholds_yaml=(
            "resource_tier_thresholds:\n"
            "  high: 22\n"
            "  balanced: 14\n"
            "  constrained: 4\n"
        ),
    )
    pack = load_model_pack(path)
    assert pack.resource_tier_thresholds == {"high": 22, "balanced": 14, "constrained": 4}


def test_pack_with_lower_thresholds_parsed(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        thresholds_yaml=(
            "resource_tier_thresholds:\n"
            "  high: 20\n"
            "  balanced: 12\n"
            "  constrained: 4\n"
        ),
    )
    pack = load_model_pack(path)
    assert pack.resource_tier_thresholds == {"high": 20, "balanced": 12, "constrained": 4}


def test_pack_rejects_unknown_tier(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        thresholds_yaml=(
            "resource_tier_thresholds:\n"
            "  extreme: 32\n"
        ),
    )
    with pytest.raises(ModelPackValidationError, match="Unknown resource tier"):
        load_model_pack(path)


def test_pack_rejects_non_integer(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        thresholds_yaml=(
            "resource_tier_thresholds:\n"
            "  high: '22gb'\n"
        ),
    )
    with pytest.raises(ModelPackValidationError, match="must be an integer"):
        load_model_pack(path)


def test_krea2_pack_ships_expected_thresholds() -> None:
    """The bundled Krea2 pack's thresholds should match the values documented in modelpack.yaml.

    Lowered from 22/14 to 20/12 on 2026-07-17 — the 22 GiB `high` bar was unreachable on
    RTX 4090-class 24 GB cards at boot (Windows compositor + prior CUDA contexts leave
    ~21-22 GiB free), causing auto-tier to demote to `balanced` and skip SageAttention +
    torch.compile.
    """

    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "models" / "packs" / "Krea2_Turbo" / "modelpack.yaml"
    if not manifest.exists():
        pytest.skip("Krea2 pack manifest not present (opt-in provisioning).")
    pack = load_model_pack(manifest)
    assert pack.resource_tier_thresholds == {"high": 20, "balanced": 12, "constrained": 4}
