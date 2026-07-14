"""Weightless tests for the ComfyUI/native -> diffusers Krea2 converters.

These use tiny synthetic tensors (no weights, no GPU) to lock the key-mapping contract:
- native transformer detection + key remap (incl. mod.lin -> scale_shift_table reshape and the
  dropped last.up/last.down),
- ComfyUI scaled-fp8 encoder detection + dequant + model./visual. prefix remap.

The exact-match validation against the real 430-key diffusers models was done during development;
these tests guard the mapping rules against regressions.
"""

from __future__ import annotations

import torch

from app.core.pipeline_factory.krea_comfy_convert import (
    convert_comfy_scaled_fp8_encoder_state_dict,
    convert_native_krea2_transformer_state_dict,
    is_comfy_scaled_fp8_encoder,
    is_native_krea2_transformer,
)


def test_is_native_transformer_detection() -> None:
    assert is_native_krea2_transformer(["blocks.0.attn.wq.weight", "first.weight"])
    assert is_native_krea2_transformer(["txtfusion.projector.weight"])
    # diffusers-native keys must NOT be flagged as native/ComfyUI.
    assert not is_native_krea2_transformer(
        ["transformer_blocks.0.attn.to_q.weight", "img_in.weight"]
    )


def test_transformer_block_key_remap() -> None:
    raw = {
        "blocks.0.attn.wq.weight": torch.zeros(4, 4),
        "blocks.0.attn.wk.weight": torch.zeros(2, 4),
        "blocks.0.attn.wo.weight": torch.zeros(4, 4),
        "blocks.0.attn.gate.weight": torch.zeros(4, 4),
        "blocks.0.attn.qknorm.qnorm.scale": torch.zeros(4),
        "blocks.0.attn.qknorm.knorm.scale": torch.zeros(4),
        "blocks.0.mlp.gate.weight": torch.zeros(8, 4),
        "blocks.0.mlp.up.weight": torch.zeros(8, 4),
        "blocks.0.mlp.down.weight": torch.zeros(4, 8),
        "blocks.0.prenorm.scale": torch.zeros(4),
        "blocks.0.postnorm.scale": torch.zeros(4),
        "blocks.0.mod.lin": torch.zeros(24),  # 6 * 4
    }
    out = convert_native_krea2_transformer_state_dict(raw)
    assert "transformer_blocks.0.attn.to_q.weight" in out
    assert "transformer_blocks.0.attn.to_k.weight" in out
    assert "transformer_blocks.0.attn.to_out.0.weight" in out
    assert "transformer_blocks.0.attn.to_gate.weight" in out
    assert "transformer_blocks.0.attn.norm_q.weight" in out
    assert "transformer_blocks.0.attn.norm_k.weight" in out
    assert "transformer_blocks.0.ff.gate.weight" in out
    assert "transformer_blocks.0.ff.up.weight" in out
    assert "transformer_blocks.0.ff.down.weight" in out
    assert "transformer_blocks.0.norm1.weight" in out
    assert "transformer_blocks.0.norm2.weight" in out
    # mod.lin (24,) -> scale_shift_table (6, 4)
    assert out["transformer_blocks.0.scale_shift_table"].shape == (6, 4)


def test_transformer_global_key_remap_and_drops() -> None:
    raw = {
        "first.weight": torch.zeros(4, 2),
        "first.bias": torch.zeros(4),
        "tmlp.0.weight": torch.zeros(4, 2),
        "tmlp.2.weight": torch.zeros(4, 4),
        "tproj.1.weight": torch.zeros(24, 4),
        "txtmlp.0.scale": torch.zeros(2),
        "txtmlp.1.weight": torch.zeros(4, 2),
        "txtmlp.3.weight": torch.zeros(4, 4),
        "last.linear.weight": torch.zeros(2, 4),
        "last.norm.scale": torch.zeros(4),
        "last.modulation.lin": torch.zeros(2, 4),
        "txtfusion.projector.weight": torch.zeros(1, 12),
        # dropped extras present in some fp8 repacks:
        "last.up.weight": torch.zeros(4, 4),
        "last.down.weight": torch.zeros(4, 4),
    }
    out = convert_native_krea2_transformer_state_dict(raw)
    assert out["img_in.weight"].shape == (4, 2)
    assert "time_embed.linear_1.weight" in out
    assert "time_embed.linear_2.weight" in out
    assert "time_mod_proj.weight" in out
    assert "txt_in.norm.weight" in out
    assert "txt_in.linear_1.weight" in out
    assert "txt_in.linear_2.weight" in out
    assert "final_layer.linear.weight" in out
    assert "final_layer.norm.weight" in out
    assert "final_layer.scale_shift_table" in out
    assert "text_fusion.projector.weight" in out
    # up/down are dropped, not mapped.
    assert not any("up.weight" in k or "down.weight" in k for k in out if k.startswith("final_layer"))
    assert "last.up.weight" not in out and "last.down.weight" not in out


def test_unknown_transformer_key_raises() -> None:
    try:
        convert_native_krea2_transformer_state_dict({"totally.unknown.key": torch.zeros(1)})
    except ValueError as exc:
        assert "Unrecognized native Krea2 transformer key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError on unknown key.")


def test_encoder_detection_and_dequant() -> None:
    keys = ["model.layers.0.mlp.down_proj.comfy_quant", "model.layers.0.mlp.down_proj.weight"]
    assert is_comfy_scaled_fp8_encoder(keys)
    assert not is_comfy_scaled_fp8_encoder(["model.layers.0.mlp.down_proj.weight"])

    # fp8 weight + scalar scale -> dequantized to compute dtype; comfy_quant/scale consumed.
    weight = torch.ones(2, 3).to(torch.float8_e4m3fn)
    raw = {
        "model.layers.0.mlp.down_proj.weight": weight,
        "model.layers.0.mlp.down_proj.weight_scale": torch.tensor(2.0),
        "model.layers.0.mlp.down_proj.comfy_quant": torch.zeros(4, dtype=torch.uint8),
        "model.visual.blocks.0.norm1.weight": torch.ones(3, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(3, dtype=torch.bfloat16),
    }
    out, dequant = convert_comfy_scaled_fp8_encoder_state_dict(raw, compute_dtype=torch.bfloat16)
    assert dequant == 1
    # prefix remap: model.layers -> language_model.layers, model.visual -> visual, model.norm -> language_model.norm
    assert "language_model.layers.0.mlp.down_proj.weight" in out
    assert "visual.blocks.0.norm1.weight" in out
    assert "language_model.norm.weight" in out
    # dequantized value = 1.0 (fp8) * 2.0 (scale) = 2.0
    assert torch.allclose(
        out["language_model.layers.0.mlp.down_proj.weight"].float(),
        torch.full((2, 3), 2.0),
    )
    # metadata + scale tensors are not emitted.
    assert not any(k.endswith((".comfy_quant", ".weight_scale")) for k in out)
