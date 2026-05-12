from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from app.config.profiles import RUNTIME_PROFILES
from app.core.model_registry import ModelComponent, ModelPack
from app.core.pipeline_factory import qwen as qwen_module


def _build_pack(root: Path, *, text_encoder: ModelComponent) -> ModelPack:
    pipeline_dir = root / "pipeline"
    (pipeline_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "text_encoder").mkdir(parents=True, exist_ok=True)
    return ModelPack(
        name="TestPack",
        architecture="z_image_turbo",
        backend_preference=["diffusers_zimage"],
        components={
            "transformer": ModelComponent(
                role="transformer",
                path=root / "unused-transformer.safetensors",
                file_format="safetensors",
            ),
            "vae": ModelComponent(
                role="vae",
                path=root / "unused-vae.safetensors",
                file_format="safetensors",
            ),
            "text_encoder": text_encoder,
        },
        pipeline_config_dir=pipeline_dir,
        required_configs=[],
        source_file=root / "modelpack.yaml",
    )


@pytest.mark.parametrize(
    ("file_format", "file_name", "expected_gguf_file"),
    [
        ("gguf", "encoder.gguf", "encoder.gguf"),
        ("safetensors", "encoder.safetensors", None),
    ],
)
def test_build_qwen_pipeline_loads_only_tokenizer_and_text_encoder(
    monkeypatch,
    workspace_tmp_path: Path,
    file_format: str,
    file_name: str,
    expected_gguf_file: str | None,
) -> None:
    root = workspace_tmp_path / f"qwen-factory-{file_format}"
    root.mkdir(parents=True, exist_ok=True)
    encoder_path = root / file_name
    encoder_path.write_bytes(b"encoder")
    pack = _build_pack(
        root,
        text_encoder=ModelComponent(
            role="text_encoder",
            path=encoder_path,
            file_format=file_format,
        ),
    )
    tokenizer_calls: list[dict[str, object]] = []
    encoder_calls: list[dict[str, object]] = []
    text_encoder = SimpleNamespace(label="encoder")

    def fake_tokenizer_from_pretrained(cls, path: str, **kwargs):
        tokenizer_calls.append({"path": path, **kwargs})
        return {"tokenizer": path}

    def fake_text_encoder_loader(**kwargs):
        encoder_calls.append(kwargs)
        return text_encoder

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(fake_tokenizer_from_pretrained))
    monkeypatch.setattr(qwen_module, "_load_text_encoder_from_local_config", fake_text_encoder_loader)

    loaded = qwen_module.build_qwen_pipeline(pack, RUNTIME_PROFILES["balanced"])

    assert loaded.tokenizer == {"tokenizer": str(pack.pipeline_config_dir / "tokenizer")}
    assert loaded.text_encoder is text_encoder
    assert loaded.device == "cpu"
    assert loaded.dtype_name == str(torch.float32)
    assert tokenizer_calls == [
        {"path": str(pack.pipeline_config_dir / "tokenizer"), "local_files_only": True}
    ]
    expected_encoder_call = {
        "component_path": encoder_path,
        "config_dir": pack.pipeline_config_dir / "text_encoder",
        "dtype": torch.float32,
        "local_files_only": True,
    }
    if expected_gguf_file is not None:
        expected_encoder_call["gguf_file"] = expected_gguf_file
    assert encoder_calls == [expected_encoder_call]
