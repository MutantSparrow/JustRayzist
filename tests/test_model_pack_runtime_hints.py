from __future__ import annotations

from pathlib import Path

import pytest

from app.core.model_registry import ModelComponent, ModelPackValidationError, load_model_pack
from app.core.pipeline_factory.zimage import _configure_component_storage


def test_model_pack_loads_runtime_storage_hints(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "runtime-hints"
    pack_dir = root / "pack"
    config_dir = pack_dir / "config"
    pack_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "transformer.safetensors").write_bytes(b"ok")
    (pack_dir / "vae.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                "name: runtime_pack",
                "architecture: z_image_turbo",
                "backend_preference:",
                "  - diffusers",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./transformer.safetensors",
                "    format: safetensors",
                "    storage_mode: layerwise",
                "    storage_dtype: fp8_e4m3fn",
                "    compute_dtype: bfloat16",
                "  vae:",
                "    path: ./vae.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_pack = load_model_pack(pack_dir / "modelpack.yaml")
    transformer = model_pack.components["transformer"]
    assert transformer.storage_mode == "layerwise"
    assert transformer.storage_dtype == "fp8_e4m3fn"
    assert transformer.compute_dtype == "bfloat16"


def test_model_pack_rejects_invalid_runtime_dtype(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "invalid-runtime"
    pack_dir = root / "pack"
    config_dir = pack_dir / "config"
    pack_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "transformer.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                "name: invalid_runtime_pack",
                "architecture: z_image_turbo",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./transformer.safetensors",
                "    format: safetensors",
                "    storage_mode: layerwise",
                "    storage_dtype: fp8_e5m2",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelPackValidationError, match="storage_dtype"):
        load_model_pack(pack_dir / "modelpack.yaml")


class _FakeTorch:
    bfloat16 = "bf16"
    float8_e4m3fn = "fp8_e4m3fn"


class _FakeModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.layerwise_calls: list[tuple[str, str]] = []

    def to(self, *, dtype):
        self.to_calls.append(dtype)
        return self

    def enable_layerwise_casting(self, *, storage_dtype, compute_dtype) -> None:
        self.layerwise_calls.append((storage_dtype, compute_dtype))


def test_configure_component_storage_leaves_default_transformer_unmodified() -> None:
    model = _FakeModel()
    component = ModelComponent(
        role="transformer",
        path=Path(__file__),
        file_format="safetensors",
    )

    configured = _configure_component_storage(
        model=model,
        component=component,
        torch_module=_FakeTorch,
        device="cuda",
        default_compute_dtype=_FakeTorch.bfloat16,
    )

    assert configured is model
    assert model.to_calls == ["bf16"]
    assert model.layerwise_calls == []


def test_configure_component_storage_applies_fp8_layerwise_casting() -> None:
    model = _FakeModel()
    component = ModelComponent(
        role="transformer",
        path=Path(__file__),
        file_format="safetensors",
        storage_dtype="fp8_e4m3fn",
        compute_dtype="bfloat16",
        storage_mode="layerwise",
    )

    configured = _configure_component_storage(
        model=model,
        component=component,
        torch_module=_FakeTorch,
        device="cuda",
        default_compute_dtype=_FakeTorch.bfloat16,
    )

    assert configured is model
    assert model.to_calls == ["bf16"]
    assert model.layerwise_calls == [("fp8_e4m3fn", "bf16")]


def test_configure_component_storage_preserves_existing_fp8_storage() -> None:
    model = _FakeModel()
    component = ModelComponent(
        role="transformer",
        path=Path(__file__),
        file_format="safetensors",
        storage_dtype="fp8_e4m3fn",
        compute_dtype="bfloat16",
        storage_mode="layerwise",
    )

    configured = _configure_component_storage(
        model=model,
        component=component,
        torch_module=_FakeTorch,
        device="cuda",
        default_compute_dtype=_FakeTorch.bfloat16,
        preserve_existing_storage=True,
    )

    assert configured is model
    assert model.to_calls == []
    assert model.layerwise_calls == [("fp8_e4m3fn", "bf16")]


def test_model_pack_defaults_user_visible_to_true(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "visibility-default"
    pack_dir = root / "pack"
    config_dir = pack_dir / "config"
    pack_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "transformer.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                "name: visible_pack",
                "architecture: z_image_turbo",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./transformer.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_pack = load_model_pack(pack_dir / "modelpack.yaml")
    assert model_pack.user_visible is True
    assert model_pack.enabled is True


def test_model_pack_rejects_non_boolean_user_visible(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "visibility-invalid"
    pack_dir = root / "pack"
    config_dir = pack_dir / "config"
    pack_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "transformer.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                "name: hidden_pack",
                'user_visible: "nope"',
                "architecture: z_image_turbo",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./transformer.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelPackValidationError, match="user_visible"):
        load_model_pack(pack_dir / "modelpack.yaml")



def test_model_pack_reads_enabled_flag(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "enabled-flag"
    pack_dir = root / "pack"
    config_dir = pack_dir / "config"
    pack_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "transformer.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                "name: disabled_pack",
                "enabled: false",
                "architecture: z_image_turbo",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./transformer.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_pack = load_model_pack(pack_dir / "modelpack.yaml")
    assert model_pack.enabled is False


def test_model_pack_rejects_non_boolean_enabled(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "enabled-invalid"
    pack_dir = root / "pack"
    config_dir = pack_dir / "config"
    pack_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "transformer.safetensors").write_bytes(b"ok")
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    (pack_dir / "modelpack.yaml").write_text(
        "\n".join(
            [
                "name: broken_pack",
                'enabled: "nope"',
                "architecture: z_image_turbo",
                "pipeline_config_dir: ./config",
                "components:",
                "  transformer:",
                "    path: ./transformer.safetensors",
                "    format: safetensors",
                "required_configs:",
                "  - ./config/model_index.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelPackValidationError, match="enabled"):
        load_model_pack(pack_dir / "modelpack.yaml")
