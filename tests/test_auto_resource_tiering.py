from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.inference_service import InferenceService
from app.config import ResourceTierController, detect_resource_tier_profile, load_settings
from app.config.profiles import RUNTIME_PROFILES
from app.core.model_registry import ModelComponent, ModelPack


GB = 1024 ** 3


def _make_temp_root(workspace_tmp_path: Path, name: str) -> Path:
    root = workspace_tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_settings(root: Path, tier_name: str = "constrained") -> SimpleNamespace:
    model_packs_dir = root / "packs"
    outputs_dir = root / "outputs"
    model_packs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        paths=SimpleNamespace(model_packs_dir=model_packs_dir, outputs_dir=outputs_dir),
        runtime_profile=RUNTIME_PROFILES["balanced"],
        resource_tier=RUNTIME_PROFILES[tier_name],
        resource_tier_override=None,
        auto_resource_tier=True,
        resource_tier_controller=ResourceTierController(current_profile=RUNTIME_PROFILES[tier_name]),
    )


def _pack(
    root: Path,
    *,
    name: str,
    transformer_format: str = "safetensors",
    complete: bool = True,
    user_visible: bool = True,
    enabled: bool = True,
) -> ModelPack:
    transformer_path = root / f"{name}_{transformer_format}.{transformer_format}"
    transformer_path.write_bytes(b"ok")
    components = {
        "transformer": ModelComponent(
            role="transformer",
            path=transformer_path,
            file_format=transformer_format,
        )
    }
    pipeline_config_dir = None
    required_configs: list[Path] = []
    if complete:
        vae_path = root / f"{name}_vae.safetensors"
        text_encoder_path = root / f"{name}_text_encoder.safetensors"
        config_dir = root / f"{name}_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "model_index.json"
        vae_path.write_bytes(b"ok")
        text_encoder_path.write_bytes(b"ok")
        config_path.write_text("{}", encoding="utf-8")
        components["vae"] = ModelComponent(
            role="vae",
            path=vae_path,
            file_format="safetensors",
        )
        components["text_encoder"] = ModelComponent(
            role="text_encoder",
            path=text_encoder_path,
            file_format="safetensors",
        )
        pipeline_config_dir = config_dir
        required_configs = [config_path]
    return ModelPack(
        name=name,
        architecture="z_image_turbo",
        backend_preference=["diffusers"],
        components=components,
        pipeline_config_dir=pipeline_config_dir,
        required_configs=required_configs,
        source_file=root / f"{name}.yaml",
        user_visible=user_visible,
        enabled=enabled,
        base_name=name,
        derived_strategy=None,
    )


def test_detect_resource_tier_profile_thresholds() -> None:
    assert detect_resource_tier_profile(free_vram_bytes=13 * GB).name == "high"
    assert detect_resource_tier_profile(free_vram_bytes=8 * GB).name == "balanced"
    assert detect_resource_tier_profile(free_vram_bytes=3 * GB).name == "constrained"


def test_load_settings_defaults_to_balanced_behavior_profile() -> None:
    settings = load_settings()
    assert settings.runtime_profile.name == "balanced"
    assert settings.auto_resource_tier is True


def test_resource_tier_controller_downgrades_immediately(monkeypatch) -> None:
    controller = ResourceTierController(current_profile=RUNTIME_PROFILES["high"])
    monkeypatch.setattr("app.config.settings.current_free_vram_bytes", lambda: 5 * GB)
    assert controller.refresh().name == "constrained"


def test_resource_tier_controller_requires_two_hits_to_promote(monkeypatch) -> None:
    controller = ResourceTierController(current_profile=RUNTIME_PROFILES["constrained"])
    monkeypatch.setattr("app.config.settings.current_free_vram_bytes", lambda: 11 * GB)
    assert controller.refresh().name == "constrained"
    assert controller.refresh().name == "balanced"


def test_inference_service_completes_minimal_pack_from_donor(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "minimal-pack")
    service = InferenceService(_make_settings(root))
    minimal_pack = _pack(root, name="custom_minimal", transformer_format="gguf", complete=False)
    donor_pack = _pack(root, name="Rayzist_bf16", complete=True)
    monkeypatch.setattr(service, "_load_donor_pack", lambda: donor_pack)

    completed = service._complete_pack_with_donor(minimal_pack)

    assert completed.pipeline_config_dir == donor_pack.pipeline_config_dir
    assert completed.required_configs == donor_pack.required_configs
    assert completed.components["transformer"].file_format == "gguf"
    assert completed.components["vae"].path == donor_pack.components["vae"].path
    assert completed.components["text_encoder"].path == donor_pack.components["text_encoder"].path


def test_constrained_tier_derives_fp8_storage_variant(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "constrained-base")
    service = InferenceService(_make_settings(root, tier_name="constrained"))
    base_pack = _pack(root, name="CustomBase", complete=True)
    monkeypatch.setattr(service, "_load_base_pack", lambda pack_name: base_pack)
    monkeypatch.setattr(service, "current_resource_tier", lambda refresh=True: RUNTIME_PROFILES["constrained"])

    selected_pack, effective_pack, resource_tier = service._resolve_runtime_pack("CustomBase")

    assert selected_pack.name == "CustomBase"
    assert resource_tier.name == "constrained"
    assert effective_pack.name == "CustomBase__auto_fp8_storage"
    assert effective_pack.base_name == "CustomBase"
    assert effective_pack.derived_strategy == "fp8_storage"
    assert effective_pack.components["transformer"].storage_mode == "layerwise"
    assert effective_pack.components["transformer"].storage_dtype == "fp8_e4m3fn"
    assert effective_pack.components["transformer"].compute_dtype == "bfloat16"


def test_constrained_tier_does_not_derive_fp8_storage_for_gguf(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "constrained-gguf")
    service = InferenceService(_make_settings(root, tier_name="constrained"))
    base_pack = _pack(root, name="CustomGguf", transformer_format="gguf", complete=True)
    monkeypatch.setattr(service, "_load_base_pack", lambda pack_name: base_pack)
    monkeypatch.setattr(service, "current_resource_tier", lambda refresh=True: RUNTIME_PROFILES["constrained"])

    _, effective_pack, _ = service._resolve_runtime_pack("CustomGguf")

    assert effective_pack.name == "CustomGguf"
    assert effective_pack.derived_strategy is None


def test_balanced_tier_keeps_base_pack_precision(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "balanced-base")
    service = InferenceService(_make_settings(root, tier_name="balanced"))
    base_pack = _pack(root, name="CustomBase", complete=True)
    monkeypatch.setattr(service, "_load_base_pack", lambda pack_name: base_pack)
    monkeypatch.setattr(service, "current_resource_tier", lambda refresh=True: RUNTIME_PROFILES["balanced"])

    _, effective_pack, _ = service._resolve_runtime_pack("CustomBase")

    assert effective_pack.name == "CustomBase"
    assert effective_pack.derived_strategy is None


def test_health_route_reports_resource_tier(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main.inference,
        "runtime_status",
        lambda: {
            "runtime_profile": "balanced",
            "resource_tier": "constrained",
            "resource_tier_description": "desc",
            "resource_tier_override": None,
            "auto_resource_tier": True,
            "active_pack": "CustomBase__auto_fp8_storage",
        },
    )

    client = TestClient(api_main.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_profile"] == "balanced"
    assert payload["resource_tier"] == "constrained"
    assert payload["active_pack"] == "CustomBase__auto_fp8_storage"


def test_config_route_includes_runtime_status(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main.inference,
        "runtime_status",
        lambda: {
            "runtime_profile": "balanced",
            "resource_tier": "high",
            "resource_tier_description": "desc",
            "resource_tier_override": None,
            "auto_resource_tier": True,
            "active_pack": None,
        },
    )

    client = TestClient(api_main.app)
    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime"]["resource_tier"] == "high"
    assert payload["runtime_profile"]["name"] == "balanced"


def test_explicit_fp8_storage_alias_derives_without_on_disk_pack(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "explicit-fp8")
    service = InferenceService(_make_settings(root, tier_name="balanced"))
    base_pack = _pack(root, name="CustomBase", complete=True)
    monkeypatch.setattr(service, "_load_base_pack", lambda pack_name: base_pack)
    monkeypatch.setattr(service, "current_resource_tier", lambda refresh=True: RUNTIME_PROFILES["balanced"])

    selected_pack, effective_pack, resource_tier = service.resolve_runtime_pack(
        "CustomBase__auto_fp8_storage",
        apply_resource_tier_policy=False,
    )

    assert selected_pack.name == "CustomBase"
    assert resource_tier.name == "balanced"
    assert effective_pack.name == "CustomBase__auto_fp8_storage"
    assert effective_pack.user_visible is False
    assert effective_pack.derived_strategy == "fp8_storage"


def test_list_model_packs_hides_non_public_or_disabled_entries(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "hidden-list")
    service = InferenceService(_make_settings(root, tier_name="balanced"))
    public_pack = _pack(root, name="Rayzist_bf16", complete=True, user_visible=True)
    hidden_pack = _pack(root, name="CustomHiddenPack", complete=True, user_visible=False)
    disabled_pack = _pack(root, name="CustomDisabledPack", complete=True, enabled=False)
    pack_files = [public_pack.source_file, hidden_pack.source_file, disabled_pack.source_file]

    monkeypatch.setattr("app.api.inference_service.discover_model_packs", lambda _dir: pack_files)
    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack",
        lambda pack_file: public_pack if pack_file == public_pack.source_file else hidden_pack if pack_file == hidden_pack.source_file else disabled_pack,
    )

    items = service.list_model_packs()

    assert [item["name"] for item in items] == ["Rayzist_bf16"]


def test_default_pack_selection_ignores_hidden_and_disabled_packs(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "default-pack")
    service = InferenceService(_make_settings(root, tier_name="balanced"))
    public_pack = _pack(root, name="Rayzist_bf16", complete=True, user_visible=True)
    hidden_pack = _pack(root, name="CustomHiddenPack", complete=True, user_visible=False)
    disabled_pack = _pack(root, name="CustomDisabledPack", complete=True, enabled=False)
    pack_files = [hidden_pack.source_file, disabled_pack.source_file, public_pack.source_file]

    monkeypatch.setattr("app.api.inference_service.discover_model_packs", lambda _dir: pack_files)
    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack",
        lambda pack_file: public_pack if pack_file == public_pack.source_file else hidden_pack if pack_file == hidden_pack.source_file else disabled_pack,
    )

    selected = service._load_base_pack(None)

    assert selected.name == "Rayzist_bf16"



def test_explicit_disabled_pack_still_loads_by_name(monkeypatch, workspace_tmp_path: Path) -> None:
    root = _make_temp_root(workspace_tmp_path, "explicit-disabled-pack")
    service = InferenceService(_make_settings(root, tier_name="balanced"))
    disabled_pack = _pack(root, name="CustomDisabledPack", complete=True, enabled=False)

    monkeypatch.setattr(
        "app.api.inference_service.load_model_pack_by_name",
        lambda _dir, pack_name: disabled_pack if pack_name == "CustomDisabledPack" else None,
    )

    selected = service._load_base_pack("CustomDisabledPack")

    assert selected.name == "CustomDisabledPack"
    assert selected.enabled is False
