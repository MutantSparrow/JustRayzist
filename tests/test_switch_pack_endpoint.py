"""API-level tests for the ``POST /model-packs/switch`` endpoint + underlying service method.

Weightless: the InferenceService's session-creation path is monkeypatched so no torch / diffusers
imports fire. Tests lock the contract (200 with runtime status on success, 400 on unknown /
hidden / disabled pack, filter honored by the endpoint, `_session_for_pack` called under the
generation lock exactly once per successful switch).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.inference_service import InferenceService
from app.config import ResourceTierController
from app.config.profiles import RUNTIME_PROFILES
from app.core.model_registry import ModelComponent, ModelPack
from app.core.model_registry.model_pack import ModelPackValidationError


def _make_settings(root: Path) -> SimpleNamespace:
    model_packs_dir = root / "packs"
    outputs_dir = root / "outputs"
    model_packs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        paths=SimpleNamespace(model_packs_dir=model_packs_dir, outputs_dir=outputs_dir),
        runtime_profile=RUNTIME_PROFILES["balanced"],
        resource_tier=RUNTIME_PROFILES["balanced"],
        resource_tier_override=None,
        auto_resource_tier=True,
        resource_tier_controller=ResourceTierController(current_profile=RUNTIME_PROFILES["balanced"]),
    )


def _make_pack(
    root: Path,
    *,
    name: str,
    user_visible: bool = True,
    enabled: bool = True,
) -> ModelPack:
    transformer_path = root / f"{name}_transformer.safetensors"
    vae_path = root / f"{name}_vae.safetensors"
    text_encoder_path = root / f"{name}_text_encoder.safetensors"
    config_dir = root / f"{name}_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "model_index.json"
    for path in (transformer_path, vae_path, text_encoder_path):
        path.write_bytes(b"ok")
    config_path.write_text("{}", encoding="utf-8")
    components = {
        "transformer": ModelComponent(role="transformer", path=transformer_path, file_format="safetensors"),
        "vae": ModelComponent(role="vae", path=vae_path, file_format="safetensors"),
        "text_encoder": ModelComponent(role="text_encoder", path=text_encoder_path, file_format="safetensors"),
    }
    return ModelPack(
        name=name,
        architecture="z_image_turbo",
        backend_preference=["diffusers"],
        components=components,
        pipeline_config_dir=config_dir,
        required_configs=[config_path],
        source_file=root / f"{name}.yaml",
        user_visible=user_visible,
        enabled=enabled,
        base_name=name,
        derived_strategy=None,
    )


def _patch_pack_lookup(monkeypatch, service: InferenceService, packs: dict[str, ModelPack]) -> None:
    """Route load_model_pack_by_name (in inference_service module scope) to a fixture map."""

    def _fake_lookup(_dir, name):
        if name in packs:
            return packs[name]
        raise ModelPackValidationError(f"Pack '{name}' not found.")

    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack_by_name",
        _fake_lookup,
    )
    monkeypatch.setattr(service, "_load_base_pack", lambda pack_name: packs[pack_name])


class _SessionSpy:
    """Records calls to _session_for_pack and mirrors the state updates the real method makes.

    The real ``_session_for_pack`` writes ``_active_pack_name`` / ``_active_selected_pack_name``
    as a side effect; ``switch_active_pack`` then re-stamps the selected name using ``base_name``.
    Reproduce those writes so tests that read runtime state after the call still work.
    """

    def __init__(self, service: InferenceService) -> None:
        self._service = service
        self.calls: list[str] = []

    def __call__(self, model_pack, resource_tier):
        self.calls.append(model_pack.name)
        self._service._active_pack_name = model_pack.name
        self._service._active_selected_pack_name = model_pack.base_name or model_pack.name
        return object()


def test_switch_active_pack_returns_runtime_and_records_selected(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "switch-happy"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    packs = {"AlphaPack": _make_pack(root, name="AlphaPack")}
    _patch_pack_lookup(monkeypatch, service, packs)

    spy = _SessionSpy(service)
    monkeypatch.setattr(service, "_session_for_pack", spy)
    monkeypatch.setattr(
        service,
        "runtime_status",
        lambda: {"active_pack": service._active_pack_name, "selected_pack": service._active_selected_pack_name},
    )

    result = service.switch_active_pack("AlphaPack")

    assert spy.calls == ["AlphaPack"]
    assert result["active_pack"] == "AlphaPack"
    assert result["selected_pack"] == "AlphaPack"


def test_switch_active_pack_rejects_unknown_pack(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "switch-unknown"
    root.mkdir()
    service = InferenceService(_make_settings(root))

    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack_by_name",
        lambda _dir, name: (_ for _ in ()).throw(ModelPackValidationError(f"'{name}' not found")),
    )

    with pytest.raises(ModelPackValidationError):
        service.switch_active_pack("Ghost")


def test_switch_active_pack_rejects_hidden_pack(monkeypatch, workspace_tmp_path: Path) -> None:
    """Hidden / disabled packs are not user-selectable even if load_model_pack_by_name resolves."""

    root = workspace_tmp_path / "switch-hidden"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    packs = {"HiddenPack": _make_pack(root, name="HiddenPack", user_visible=False)}
    _patch_pack_lookup(monkeypatch, service, packs)
    spy = _SessionSpy(service)
    monkeypatch.setattr(service, "_session_for_pack", spy)

    with pytest.raises(ModelPackValidationError, match="not user-selectable"):
        service.switch_active_pack("HiddenPack")
    assert spy.calls == []


def test_switch_active_pack_rejects_disabled_pack(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "switch-disabled"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    packs = {"DisabledPack": _make_pack(root, name="DisabledPack", enabled=False)}
    _patch_pack_lookup(monkeypatch, service, packs)
    spy = _SessionSpy(service)
    monkeypatch.setattr(service, "_session_for_pack", spy)

    with pytest.raises(ModelPackValidationError, match="not user-selectable"):
        service.switch_active_pack("DisabledPack")
    assert spy.calls == []


def test_switch_active_pack_empty_name_rejected(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "switch-empty"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    with pytest.raises(ModelPackValidationError, match="required"):
        service.switch_active_pack("   ")


def test_switch_pack_endpoint_returns_200_on_success(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "endpoint-happy"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    packs = {"AlphaPack": _make_pack(root, name="AlphaPack")}
    _patch_pack_lookup(monkeypatch, service, packs)
    monkeypatch.setattr(service, "_session_for_pack", _SessionSpy(service))
    monkeypatch.setattr(
        service,
        "runtime_status",
        lambda: {"active_pack": "AlphaPack", "selected_pack": "AlphaPack", "resource_tier": "balanced"},
    )
    monkeypatch.setattr(api_main, "inference", service)

    client = TestClient(api_main.app)
    response = client.post("/model-packs/switch", json={"name": "AlphaPack"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["runtime"]["active_pack"] == "AlphaPack"


def test_switch_pack_endpoint_returns_400_on_unknown_pack(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "endpoint-unknown"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack_by_name",
        lambda _dir, name: (_ for _ in ()).throw(ModelPackValidationError(f"'{name}' not found")),
    )
    monkeypatch.setattr(api_main, "inference", service)

    client = TestClient(api_main.app)
    response = client.post("/model-packs/switch", json={"name": "Ghost"})
    assert response.status_code == 400
    assert "Ghost" in response.json()["detail"]


def test_switch_pack_endpoint_returns_400_on_hidden_pack(
    monkeypatch, workspace_tmp_path: Path
) -> None:
    root = workspace_tmp_path / "endpoint-hidden"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    packs = {"HiddenPack": _make_pack(root, name="HiddenPack", user_visible=False)}
    _patch_pack_lookup(monkeypatch, service, packs)
    monkeypatch.setattr(api_main, "inference", service)

    client = TestClient(api_main.app)
    response = client.post("/model-packs/switch", json={"name": "HiddenPack"})
    assert response.status_code == 400
    assert "not user-selectable" in response.json()["detail"]


def test_switch_pack_endpoint_rejects_missing_name(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "endpoint-missing"
    root.mkdir()
    service = InferenceService(_make_settings(root))
    monkeypatch.setattr(api_main, "inference", service)

    client = TestClient(api_main.app)
    response = client.post("/model-packs/switch", json={})
    assert response.status_code == 422  # pydantic missing field
