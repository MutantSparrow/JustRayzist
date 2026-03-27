from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import diffusers
import torch
from transformers import AutoModel, AutoModelForCausalLM

from app.config.profiles import RUNTIME_PROFILES
from app.core.model_registry import ModelComponent, ModelPack
from app.core.pipeline_factory import zimage as zimage_module


class _FakePipeline:
    last_call: dict[str, object] | None = None

    def __init__(self) -> None:
        self.progress_disabled = None
        self.to_calls: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        cls.last_call = {"path": path, **kwargs}
        return cls()

    def set_progress_bar_config(self, disable: bool = True) -> None:
        self.progress_disabled = disable

    def to(self, device: str):
        self.to_calls.append(device)
        return self


class _FakeTransformerLoader:
    last_call: dict[str, object] | None = None

    @classmethod
    def from_single_file(cls, path: str, **kwargs):
        cls.last_call = {"path": path, **kwargs}
        return object()


class _FakeVaeLoader:
    last_call: dict[str, object] | None = None

    @classmethod
    def from_single_file(cls, path: str, **kwargs):
        cls.last_call = {"path": path, **kwargs}
        return object()


class _FakeCausalLmLoader:
    last_call: dict[str, object] | None = None


class _FakeAutoModelLoader:
    last_call: dict[str, object] | None = None


def _build_pack(root: Path, *, components: dict[str, ModelComponent]) -> ModelPack:
    pipeline_dir = root / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    return ModelPack(
        name="TestPack",
        architecture="z_image_turbo",
        backend_preference=["diffusers_zimage"],
        components=components,
        pipeline_config_dir=pipeline_dir,
        required_configs=[],
        source_file=root / "modelpack.yaml",
    )


def test_build_zimage_pipeline_uses_torch_dtype_for_pipeline_and_component_loads(
    monkeypatch,
    workspace_tmp_path: Path,
) -> None:
    root = workspace_tmp_path / "zimage-factory-kwargs"
    root.mkdir(parents=True, exist_ok=True)
    transformer_path = root / "transformer.safetensors"
    vae_path = root / "vae.safetensors"
    text_encoder_path = root / "text_encoder.safetensors"
    transformer_path.write_bytes(b"transformer")
    vae_path.write_bytes(b"vae")
    text_encoder_path.write_bytes(b"text_encoder")

    pack = _build_pack(
        root,
        components={
            "transformer": ModelComponent(role="transformer", path=transformer_path, file_format="safetensors"),
            "vae": ModelComponent(role="vae", path=vae_path, file_format="safetensors"),
            "text_encoder": ModelComponent(role="text_encoder", path=text_encoder_path, file_format="safetensors"),
        },
    )

    def fake_text_encoder_loader(*, component_path: Path, config_dir: Path, dtype, local_files_only: bool, gguf_file=None):
        return {"kind": "text_encoder", "dtype": dtype, "config_dir": str(config_dir), "gguf_file": gguf_file}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diffusers, "ZImagePipeline", _FakePipeline)
    monkeypatch.setattr(diffusers, "ZImageTransformer2DModel", _FakeTransformerLoader)
    monkeypatch.setattr(diffusers, "AutoencoderKL", _FakeVaeLoader)
    monkeypatch.setattr(zimage_module, "_checkpoint_contains_fp8_weights", lambda path: False)
    monkeypatch.setattr(zimage_module, "_is_prefixed_fused_zimage_transformer", lambda path: False)
    monkeypatch.setattr(zimage_module, "_configure_component_storage", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(zimage_module, "_load_text_encoder_from_local_config", fake_text_encoder_loader)

    loaded = zimage_module._build_zimage_pipeline(
        pack,
        RUNTIME_PROFILES["balanced"],
        backend_name="diffusers_zimage",
        enable_real_fp8_checkpoint_support=False,
    )

    assert loaded.pipeline.progress_disabled is True
    assert _FakePipeline.last_call is not None
    assert _FakePipeline.last_call["torch_dtype"] == torch.float32
    assert "dtype" not in _FakePipeline.last_call
    assert _FakePipeline.last_call["text_encoder"]["kind"] == "text_encoder"

    assert _FakeTransformerLoader.last_call is not None
    assert _FakeTransformerLoader.last_call["torch_dtype"] == torch.float32
    assert "dtype" not in _FakeTransformerLoader.last_call

    assert _FakeVaeLoader.last_call is not None
    assert _FakeVaeLoader.last_call["torch_dtype"] == torch.float32
    assert "dtype" not in _FakeVaeLoader.last_call


def test_load_text_encoder_from_local_config_uses_transformers_dtype(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "zimage-text-encoder-kwargs"
    root.mkdir(parents=True, exist_ok=True)
    component_path = root / "encoder.gguf"
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    component_path.write_bytes(b"gguf")

    def fake_causallm_from_pretrained(cls, path: str, **kwargs):
        _FakeCausalLmLoader.last_call = {"path": path, **kwargs}
        model = SimpleNamespace(state_dict=lambda: {})
        return model, {"missing_keys": []}

    def fake_automodel_from_pretrained(cls, path: str, **kwargs):
        _FakeAutoModelLoader.last_call = {"path": path, **kwargs}
        raise AssertionError("Fallback AutoModel loader should not be used in this test.")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(fake_causallm_from_pretrained))
    monkeypatch.setattr(AutoModel, "from_pretrained", classmethod(fake_automodel_from_pretrained))

    model = zimage_module._load_text_encoder_from_local_config(
        component_path=component_path,
        config_dir=config_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        gguf_file=component_path.name,
    )

    assert model is not None
    assert _FakeCausalLmLoader.last_call is not None
    assert _FakeCausalLmLoader.last_call["dtype"] == torch.bfloat16
    assert _FakeCausalLmLoader.last_call["gguf_file"] == "encoder.gguf"
    assert "torch_dtype" not in _FakeCausalLmLoader.last_call
    assert _FakeAutoModelLoader.last_call is None
