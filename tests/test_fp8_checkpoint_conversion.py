from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from app.core.pipeline_factory.zimage import _checkpoint_contains_fp8_weights


_SCRIPT_PATH = Path.cwd() / "scripts" / "convert_rayzist_transformer_to_fp8.py"
_SPEC = importlib.util.spec_from_file_location("convert_rayzist_transformer_to_fp8", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_convert_transformer_state_dict_respects_mixed_and_full_policies() -> None:
    state_dict = {
        "layers.0.attn.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "layers.0.attn.bias": torch.randn(4, dtype=torch.bfloat16),
        "layers.0.norm.weight": torch.randn(4, dtype=torch.bfloat16),
        "layers.0.cls_token": torch.randn(2, 4, dtype=torch.bfloat16),
        "layers.0.index": torch.arange(4, dtype=torch.int64),
    }

    mixed = _MODULE.convert_transformer_state_dict(state_dict, mode="mixed")
    full = _MODULE.convert_transformer_state_dict(state_dict, mode="full")

    assert mixed["layers.0.attn.weight"].dtype == torch.float8_e4m3fn
    assert mixed["layers.0.attn.bias"].dtype == torch.bfloat16
    assert mixed["layers.0.norm.weight"].dtype == torch.bfloat16
    assert mixed["layers.0.cls_token"].dtype == torch.bfloat16
    assert mixed["layers.0.index"].dtype == torch.int64

    assert full["layers.0.attn.weight"].dtype == torch.float8_e4m3fn
    assert full["layers.0.attn.bias"].dtype == torch.float8_e4m3fn
    assert full["layers.0.norm.weight"].dtype == torch.float8_e4m3fn
    assert full["layers.0.cls_token"].dtype == torch.float8_e4m3fn
    assert full["layers.0.index"].dtype == torch.int64


def test_convert_checkpoint_writes_fp8_safetensors(workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "fp8-checkpoint"
    source_path = root / "source.safetensors"
    mixed_path = root / "mixed.safetensors"
    full_path = root / "full.safetensors"
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "layers.0.attn.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "layers.0.attn.bias": torch.randn(4, dtype=torch.bfloat16),
        },
        str(source_path),
    )

    mixed_summary = _MODULE.convert_checkpoint(input_path=source_path, output_path=mixed_path, mode="mixed")
    full_summary = _MODULE.convert_checkpoint(input_path=source_path, output_path=full_path, mode="full")
    mixed_state = load_file(str(mixed_path))
    full_state = load_file(str(full_path))

    assert mixed_state["layers.0.attn.weight"].dtype == torch.float8_e4m3fn
    assert mixed_state["layers.0.attn.bias"].dtype == torch.bfloat16
    assert full_state["layers.0.attn.weight"].dtype == torch.float8_e4m3fn
    assert full_state["layers.0.attn.bias"].dtype == torch.float8_e4m3fn
    assert _checkpoint_contains_fp8_weights(mixed_path)
    assert _checkpoint_contains_fp8_weights(full_path)
    assert "torch.float8_e4m3fn" in mixed_summary
    assert "torch.float8_e4m3fn" in full_summary

