from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from app.core.backends import DiffusersZImageBackend, Fp8ZImageBackend, create_backend
from app.core.pipeline_factory.zimage import (
    _apply_fp8_storage_fallback_materialization,
    _normalize_fp8_state_dict_for_runtime,
)


def _fake_pack(*, backend_preference: list[str]):
    return SimpleNamespace(
        name="Rayzist_fp8_mixed",
        base_name="Rayzist_fp8_mixed",
        backend_preference=backend_preference,
        components={},
        architecture="z_image_turbo",
    )


def test_create_backend_prefers_fp8_backend_when_requested() -> None:
    backend = create_backend(
        settings=SimpleNamespace(),
        model_pack=_fake_pack(backend_preference=["fp8_zimage", "diffusers"]),
        resource_tier=None,
    )

    assert isinstance(backend, Fp8ZImageBackend)


def test_create_backend_uses_diffusers_backend_for_bf16_pack() -> None:
    backend = create_backend(
        settings=SimpleNamespace(),
        model_pack=_fake_pack(backend_preference=["diffusers"]),
        resource_tier=None,
    )

    assert isinstance(backend, DiffusersZImageBackend)


def test_fp8_runtime_fallback_preserves_safe_fp8_matrix_weights() -> None:
    state_dict = {
        "layers.0.attn.weight": torch.randn(4, 4, dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
        "layers.0.attn.bias": torch.randn(4, dtype=torch.bfloat16),
        "layers.0.norm.weight": torch.randn(4, dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
        "layers.0.index": torch.arange(4, dtype=torch.int64),
    }

    preparation = _normalize_fp8_state_dict_for_runtime(
        state_dict,
        compute_dtype=torch.bfloat16,
        torch_module=torch,
    )

    normalized = preparation.state_dict
    assert normalized["layers.0.attn.weight"].dtype == torch.float8_e4m3fn
    assert normalized["layers.0.attn.bias"].dtype == torch.bfloat16
    assert normalized["layers.0.norm.weight"].dtype == torch.bfloat16
    assert normalized["layers.0.index"].dtype == torch.int64
    assert preparation.promoted_names == ("layers.0.norm.weight",)
    assert preparation.storage_preserved_tensor_count == 1
    assert preparation.promoted_tensor_count == 1


def test_fp8_runtime_normalization_promotes_only_sensitive_tensors() -> None:
    state_dict = {
        "layers.0.attn.weight": torch.randn(4, 4, dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
        "layers.0.attn.bias": torch.randn(4, dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
        "layers.0.norm.weight": torch.randn(4, dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
        "layers.0.cls_token": torch.randn(2, 4, dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
    }

    preparation = _normalize_fp8_state_dict_for_runtime(
        state_dict,
        compute_dtype=torch.bfloat16,
        torch_module=torch,
    )

    normalized = preparation.state_dict
    assert normalized["layers.0.attn.weight"].dtype == torch.float8_e4m3fn
    assert normalized["layers.0.attn.bias"].dtype == torch.bfloat16
    assert normalized["layers.0.norm.weight"].dtype == torch.bfloat16
    assert normalized["layers.0.cls_token"].dtype == torch.bfloat16
    assert set(preparation.promoted_names) == {
        "layers.0.attn.bias",
        "layers.0.norm.weight",
        "layers.0.cls_token",
    }
    assert preparation.storage_preserved_tensor_count == 1
    assert preparation.promoted_tensor_count == 3


class _ToyLinearModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
        self.norm_weight = nn.Parameter(torch.randn(4, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


def test_fp8_storage_fallback_materialization_keeps_sensitive_tensors_in_bf16() -> None:
    model = _ToyLinearModule()

    metadata = _apply_fp8_storage_fallback_materialization(
        model=model,
        torch_module=torch,
        compute_dtype=torch.bfloat16,
    )

    assert metadata["fp8_storage_preserved_tensor_count"] == 1
    assert model.weight.dtype == torch.float8_e4m3fn
    assert model.bias.dtype == torch.bfloat16
    assert model.norm_weight.dtype == torch.bfloat16

    output = model(torch.randn(2, 4, dtype=torch.bfloat16))

    assert output.dtype == torch.bfloat16
    assert model.weight.dtype == torch.float8_e4m3fn
    assert model.bias.dtype == torch.bfloat16
    assert model.norm_weight.dtype == torch.bfloat16


