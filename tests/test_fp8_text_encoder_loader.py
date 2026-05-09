from __future__ import annotations

import pytest
import torch

from app.core.pipeline_factory.zimage import _convert_scaled_fp8_text_encoder_state_dict


def test_convert_scaled_fp8_text_encoder_state_dict_dequantizes_weights() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch build has no float8_e4m3fn dtype")

    base_weight = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.float32)
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": base_weight.to(dtype=fp8_dtype),
        "model.layers.0.self_attn.q_proj.scale_weight": torch.tensor([0.5], dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.scale_input": torch.tensor([1.0], dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(2, dtype=torch.float32),
        "scaled_fp8": torch.empty(0, dtype=fp8_dtype),
    }

    converted, converted_count = _convert_scaled_fp8_text_encoder_state_dict(
        state_dict,
        dtype=torch.bfloat16,
    )

    assert converted_count == 1
    assert set(converted) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    }
    assert converted["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert converted["model.norm.weight"].dtype == torch.bfloat16
    expected = base_weight.to(dtype=fp8_dtype).to(dtype=torch.bfloat16) * torch.tensor(
        [0.5],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(
        converted["model.layers.0.self_attn.q_proj.weight"],
        expected,
    )
