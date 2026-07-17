"""Guard: chat/wildcard-suggest capabilities gate on architecture, not just component presence.

Krea2 packs load Qwen3VLModel (VL, mRoPE) — the shared DiffusersQwenInference text-decode path
is built for Qwen3ForCausalLM and crashes / degrades on VL. These tests lock the guard so a
regression that removes the arch check surfaces immediately.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.inference_service import InferenceService
from app.config import ResourceTierController
from app.config.profiles import RUNTIME_PROFILES
from app.core.model_registry import ModelComponent, ModelPack


def _make_settings(root: Path) -> SimpleNamespace:
    model_packs_dir = root / "packs"
    outputs_dir = root / "outputs"
    data_dir = root / "data"
    model_packs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        paths=SimpleNamespace(
            model_packs_dir=model_packs_dir,
            outputs_dir=outputs_dir,
            data_dir=data_dir,
        ),
        runtime_profile=RUNTIME_PROFILES["balanced"],
        resource_tier=RUNTIME_PROFILES["balanced"],
        resource_tier_override=None,
        auto_resource_tier=True,
        resource_tier_controller=ResourceTierController(current_profile=RUNTIME_PROFILES["balanced"]),
    )


def _make_pack(root: Path, *, name: str, architecture: str) -> ModelPack:
    transformer_path = root / f"{name}_transformer.safetensors"
    vae_path = root / f"{name}_vae.safetensors"
    text_encoder_path = root / f"{name}_text_encoder.safetensors"
    config_dir = root / f"{name}_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "model_index.json"
    for path in (transformer_path, vae_path, text_encoder_path):
        path.write_bytes(b"ok")
    config_path.write_text("{}", encoding="utf-8")
    return ModelPack(
        name=name,
        architecture=architecture,
        backend_preference=["diffusers"] if architecture == "z_image_turbo" else ["fp8_krea"],
        components={
            "transformer": ModelComponent(role="transformer", path=transformer_path, file_format="safetensors"),
            "vae": ModelComponent(role="vae", path=vae_path, file_format="safetensors"),
            "text_encoder": ModelComponent(role="text_encoder", path=text_encoder_path, file_format="safetensors"),
        },
        pipeline_config_dir=config_dir,
        required_configs=[config_path],
        source_file=root / f"{name}.yaml",
        user_visible=True,
        enabled=True,
        base_name=name,
        derived_strategy=None,
    )


def test_z_image_pack_supports_text_decode(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "z-image-decode"
    root.mkdir()
    pack = _make_pack(root, name="Rayzist_bf16", architecture="z_image_turbo")
    assert InferenceService._pack_supports_qwen_text_decode(pack) is True


def test_krea2_pack_supports_text_decode_via_qwen3vl_path(workspace_tmp_path: Path) -> None:
    """Krea2 packs are now wired for chat / rewrite / wildcard-suggest via
    DiffusersQwen3VLInference, so ``_pack_supports_qwen_text_decode`` accepts them."""

    root = workspace_tmp_path / "krea2-decode"
    root.mkdir()
    pack = _make_pack(root, name="Krea2_Turbo", architecture="krea2_turbo")
    assert InferenceService._pack_supports_qwen_text_decode(pack) is True


def test_unknown_architecture_blocks_text_decode(workspace_tmp_path: Path) -> None:
    """Any pack outside the two wired architectures is refused so the UI can hide chat."""

    root = workspace_tmp_path / "unknown-decode"
    root.mkdir()
    pack = _make_pack(root, name="MysteryPack", architecture="mystery_arch")
    assert InferenceService._pack_supports_qwen_text_decode(pack) is False


def test_chat_capabilities_reports_supported_for_krea2(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "chat-caps-krea2"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    krea_pack = _make_pack(root, name="Krea2_Turbo", architecture="krea2_turbo")
    monkeypatch.setattr(
        service,
        "_resolve_runtime_pack",
        lambda *args, **kwargs: (krea_pack, krea_pack, RUNTIME_PROFILES["balanced"]),
    )

    caps = service.chat_capabilities("Krea2_Turbo")
    assert caps["supported"] is True
    assert caps["active_pack"] == "Krea2_Turbo"


def test_wildcard_capabilities_reports_supported_for_krea2(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "wildcard-caps-krea2"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    krea_pack = _make_pack(root, name="Krea2_Turbo", architecture="krea2_turbo")
    monkeypatch.setattr(
        service,
        "_resolve_runtime_pack",
        lambda *args, **kwargs: (krea_pack, krea_pack, RUNTIME_PROFILES["balanced"]),
    )

    caps = service.wildcard_capabilities("Krea2_Turbo")
    assert caps["supported"] is True
    assert caps["suggestions_supported"] is True


def test_chat_capabilities_reports_unsupported_for_unknown_arch(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "chat-caps-unknown"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    mystery_pack = _make_pack(root, name="MysteryPack", architecture="mystery_arch")
    monkeypatch.setattr(
        service,
        "_resolve_runtime_pack",
        lambda *args, **kwargs: (mystery_pack, mystery_pack, RUNTIME_PROFILES["balanced"]),
    )

    caps = service.chat_capabilities("MysteryPack")
    assert caps["supported"] is False


def test_chat_call_refuses_unknown_arch(monkeypatch, workspace_tmp_path: Path) -> None:
    """The chat entry point still rejects packs whose architecture isn't wired for text decode."""

    root = workspace_tmp_path / "chat-refuse-unknown"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    mystery_pack = _make_pack(root, name="MysteryPack", architecture="mystery_arch")
    monkeypatch.setattr(
        service,
        "_resolve_runtime_pack",
        lambda *args, **kwargs: (mystery_pack, mystery_pack, RUNTIME_PROFILES["balanced"]),
    )
    session_calls: list[str] = []

    def _fail_session(*args, **kwargs):
        session_calls.append("called")
        raise AssertionError("Session must not be built for unsupported architecture.")

    monkeypatch.setattr(service, "_session_for_pack", _fail_session)

    with pytest.raises(ValueError, match="Chat is not supported"):
        service.chat(owner_id="example-client", message="hello")
    assert session_calls == []


def test_load_base_pack_prefers_active_pack_over_default(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    """After a switch, ``_load_base_pack(None)`` should return the active pack, not the default."""

    root = workspace_tmp_path / "base-active-pref"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    active_pack = _make_pack(root, name="ActivePack", architecture="z_image_turbo")
    default_pack = _make_pack(root, name="DefaultPack", architecture="z_image_turbo")

    def _fake_load_by_name(_dir, name):
        if name == "ActivePack":
            return active_pack
        if name == "DefaultPack":
            return default_pack
        raise AssertionError(f"unexpected pack name {name!r}")

    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack_by_name",
        _fake_load_by_name,
    )
    # Simulate a prior switch that set _active_selected_pack_name.
    service._active_selected_pack_name = "ActivePack"
    service._default_pack_name = "DefaultPack"

    resolved = service._load_base_pack(None)
    assert resolved.name == "ActivePack"


def test_load_base_pack_falls_back_to_default_without_active(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "base-default-fallback"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    default_pack = _make_pack(root, name="DefaultPack", architecture="z_image_turbo")

    def _fake_load_by_name(_dir, name):
        if name == "DefaultPack":
            return default_pack
        raise AssertionError(f"unexpected pack name {name!r}")

    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack_by_name",
        _fake_load_by_name,
    )
    service._active_selected_pack_name = None
    service._default_pack_name = "DefaultPack"

    resolved = service._load_base_pack(None)
    assert resolved.name == "DefaultPack"
