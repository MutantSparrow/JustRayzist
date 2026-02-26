from __future__ import annotations

import torch

from app.core.upscale import _build_upscaler_network, _detect_upscaler_architecture


def _make_synthetic_plksr_state() -> dict[str, torch.Tensor]:
    dim = 8
    hidden = 16
    split = 2
    kernel = 5
    scale = 2
    final_out = 3 * (scale**2)
    return {
        "feats.0.weight": torch.randn(dim, 3, 3, 3),
        "feats.0.bias": torch.randn(dim),
        "feats.1.channel_mixer.0.weight": torch.randn(hidden, dim, 3, 3),
        "feats.1.channel_mixer.0.bias": torch.randn(hidden),
        "feats.1.channel_mixer.2.weight": torch.randn(dim, hidden, 3, 3),
        "feats.1.channel_mixer.2.bias": torch.randn(dim),
        "feats.1.lk.conv.weight": torch.randn(split, split, kernel, kernel),
        "feats.1.lk.conv.bias": torch.randn(split),
        "feats.1.attn.f.0.weight": torch.randn(dim, dim, 3, 3),
        "feats.1.attn.f.0.bias": torch.randn(dim),
        "feats.1.refine.weight": torch.randn(dim, dim, 1, 1),
        "feats.1.refine.bias": torch.randn(dim),
        "feats.1.norm.weight": torch.randn(dim),
        "feats.1.norm.bias": torch.randn(dim),
        "feats.3.weight": torch.randn(final_out, dim, 3, 3),
        "feats.3.bias": torch.randn(final_out),
    }


def test_detect_plksr_architecture_from_feats_keys() -> None:
    state = _make_synthetic_plksr_state()
    architecture = _detect_upscaler_architecture(state)
    assert architecture == "plksr"


def test_build_plksr_network_loads_state_and_scales_output() -> None:
    state = _make_synthetic_plksr_state()
    model, scale_factor = _build_upscaler_network(torch, state)
    assert scale_factor == 2

    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 9, 7))
    assert tuple(output.shape) == (1, 3, 18, 14)
