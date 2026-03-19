from __future__ import annotations

import torch
import torch.nn.functional as F

from app.core.backends.diffusers_zimage import DiffusersZImageBackend


def _channel_divergence(tensor) -> float:
    latent = tensor.squeeze(0)
    mono = latent.mean(dim=0, keepdim=True)
    return float((latent - mono).abs().mean().item())


def _macro_structure_strength(tensor) -> float:
    latent = tensor.mean(dim=1, keepdim=True)
    pooled = F.avg_pool2d(latent, kernel_size=16, stride=16)
    return float(pooled.std(unbiased=False).item())



def test_procedural_latent_generation_is_deterministic_per_level() -> None:
    first_tensor, first_recipe = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=12345,
        creativity=2,
        torch_module=torch,
    )
    second_tensor, second_recipe = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=12345,
        creativity=2,
        torch_module=torch,
    )

    assert first_recipe == second_recipe
    assert first_recipe.startswith("proc_v4/")
    assert "/lvl2/" in first_recipe
    assert first_tensor.shape == (1, 16, 128, 128)
    assert torch.equal(first_tensor, second_tensor)



def test_procedural_latent_levels_are_distinct_for_same_seed() -> None:
    level_1, recipe_1 = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=777,
        creativity=1,
        torch_module=torch,
    )
    level_2, recipe_2 = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=777,
        creativity=2,
        torch_module=torch,
    )
    level_3, recipe_3 = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=777,
        creativity=3,
        torch_module=torch,
    )

    assert recipe_1 != recipe_2
    assert recipe_2 != recipe_3
    assert recipe_1 != recipe_3
    assert "/grain" in recipe_1
    assert "/shapes0" in recipe_1
    assert "/shapes0" in recipe_2
    assert "/quant0" in recipe_1
    assert "/quant0" in recipe_2
    assert "/overlay_final0" in recipe_1
    assert "/overlay_final0" in recipe_2
    assert "/shapes" in recipe_3 and "/shapes0" not in recipe_3
    assert "/quant6" in recipe_3
    assert "/overlay_final" in recipe_3 and "/overlay_final0" not in recipe_3
    assert not torch.equal(level_1, level_2)
    assert not torch.equal(level_2, level_3)
    assert not torch.equal(level_1, level_3)



def test_procedural_latent_levels_shift_monochrome_and_macro_structure() -> None:
    level_1, _ = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=2468,
        creativity=1,
        torch_module=torch,
    )
    level_2, _ = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=2468,
        creativity=2,
        torch_module=torch,
    )
    level_3, _ = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=128,
        target_width=128,
        seed=2468,
        creativity=3,
        torch_module=torch,
    )

    divergence_1 = _channel_divergence(level_1)
    divergence_2 = _channel_divergence(level_2)
    divergence_3 = _channel_divergence(level_3)
    assert divergence_1 < divergence_2
    assert divergence_1 < divergence_3

    macro_1 = _macro_structure_strength(level_1)
    macro_3 = _macro_structure_strength(level_3)
    assert macro_3 > macro_1
    assert float(level_3.max().item()) > 0.95
    assert float(level_3.min().item()) < -0.95



def test_procedural_latent_mix_is_finite_and_distinct_by_seed() -> None:
    first_tensor, first_recipe = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=96,
        target_width=96,
        seed=101,
        creativity=2,
        torch_module=torch,
    )
    second_tensor, second_recipe = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=96,
        target_width=96,
        seed=202,
        creativity=2,
        torch_module=torch,
    )
    first_mixed, first_alpha, first_preprocess = DiffusersZImageBackend._normalize_and_mix_latent(
        latent_tensor=first_tensor,
        seed=101,
        torch_module=torch,
        noise_mix=DiffusersZImageBackend._PROCEDURAL_LATENT_NOISE_MIX,
        preprocess=DiffusersZImageBackend._PROCEDURAL_LATENT_PREPROCESS,
    )

    assert first_recipe != second_recipe
    assert not torch.equal(first_tensor, second_tensor)
    assert first_mixed.shape == (1, 16, 96, 96)
    assert torch.isfinite(first_mixed).all()
    assert abs(float(first_mixed.mean().item())) < 1e-4
    assert 0.9 < float(first_mixed.std(unbiased=False).item()) < 1.1
    assert first_alpha == DiffusersZImageBackend._PROCEDURAL_LATENT_NOISE_MIX
    assert first_preprocess == DiffusersZImageBackend._PROCEDURAL_LATENT_PREPROCESS



def test_level3_procedural_mix_uses_higher_latent_contribution() -> None:
    level_3_tensor, _ = DiffusersZImageBackend._build_procedural_latent_tensor(
        expected_channels=16,
        target_height=96,
        target_width=96,
        seed=303,
        creativity=3,
        torch_module=torch,
    )
    mixed, alpha, preprocess = DiffusersZImageBackend._normalize_and_mix_latent(
        latent_tensor=level_3_tensor,
        seed=303,
        torch_module=torch,
        noise_mix=DiffusersZImageBackend._PROCEDURAL_LATENT_NOISE_MIX_LEVEL3,
        preprocess=DiffusersZImageBackend._PROCEDURAL_LATENT_PREPROCESS,
    )

    assert mixed.shape == (1, 16, 96, 96)
    assert torch.isfinite(mixed).all()
    assert alpha == DiffusersZImageBackend._PROCEDURAL_LATENT_NOISE_MIX_LEVEL3
    assert preprocess == DiffusersZImageBackend._PROCEDURAL_LATENT_PREPROCESS
