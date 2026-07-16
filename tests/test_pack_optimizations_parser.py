"""Parser tests for the modelpack.yaml ``optimizations`` block.

Weightless: no torch, no CUDA, no diffusers. Just YAML → dataclass validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.model_registry import load_model_pack
from app.core.model_registry.model_pack import ModelPackValidationError
from app.core.pipeline_factory.optimizations import OptimizationsConfig


def _write_pack(tmp_path: Path, opt_yaml: str = "") -> Path:
    pack_dir = tmp_path / "TestPack"
    pack_dir.mkdir()
    weights_dir = pack_dir / "weights"
    weights_dir.mkdir()
    (weights_dir / "transformer.safetensors").write_bytes(b"ok")
    (weights_dir / "vae.safetensors").write_bytes(b"ok")
    (weights_dir / "text_encoder.safetensors").write_bytes(b"ok")

    config_dir = pack_dir / "config"
    config_dir.mkdir()
    (config_dir / "model_index.json").write_text("{}", encoding="utf-8")

    manifest = f"""\
name: TestPack
architecture: krea2_turbo
backend_preference:
  - fp8_krea
pipeline_config_dir: ./config
components:
  transformer:
    path: ./weights/transformer.safetensors
    format: safetensors
  vae:
    path: ./weights/vae.safetensors
    format: safetensors
  text_encoder:
    path: ./weights/text_encoder.safetensors
    format: safetensors
required_configs:
  - ./config/model_index.json
{opt_yaml}
"""
    manifest_path = pack_dir / "modelpack.yaml"
    manifest_path.write_text(manifest, encoding="utf-8")
    return manifest_path


def test_absent_optimizations_defaults_disabled(tmp_path: Path) -> None:
    pack = load_model_pack(_write_pack(tmp_path))
    assert isinstance(pack.optimizations, OptimizationsConfig)
    assert pack.optimizations.torch_compile.enabled is False
    assert pack.optimizations.fp8_quantization.enabled is False
    assert pack.optimizations.sage_attention.enabled is False


def test_full_optimizations_block_parsed(tmp_path: Path) -> None:
    pack = load_model_pack(_write_pack(
        tmp_path,
        opt_yaml=(
            "optimizations:\n"
            "  torch_compile:\n"
            "    enabled: true\n"
            "    mode: reduce-overhead\n"
            "    fullgraph: false\n"
            "  fp8_quantization:\n"
            "    enabled: true\n"
            "    scope: transformer\n"
            "  sage_attention:\n"
            "    enabled: true\n"
        ),
    ))
    opts = pack.optimizations
    assert opts.torch_compile.enabled is True
    assert opts.torch_compile.mode == "reduce-overhead"
    assert opts.fp8_quantization.enabled is True
    assert opts.fp8_quantization.scope == "transformer"
    assert opts.sage_attention.enabled is True


def test_boolean_shortcut_accepted(tmp_path: Path) -> None:
    """``torch_compile: true`` should be equivalent to ``torch_compile: {enabled: true}``."""

    pack = load_model_pack(_write_pack(
        tmp_path,
        opt_yaml=(
            "optimizations:\n"
            "  torch_compile: true\n"
            "  sage_attention: false\n"
        ),
    ))
    assert pack.optimizations.torch_compile.enabled is True
    assert pack.optimizations.torch_compile.mode == "reduce-overhead"  # default preserved
    assert pack.optimizations.sage_attention.enabled is False


def test_unknown_optimization_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelPackValidationError, match="Unknown optimization keys"):
        load_model_pack(_write_pack(
            tmp_path,
            opt_yaml=(
                "optimizations:\n"
                "  fp4_quant: true\n"
            ),
        ))


def test_invalid_compile_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelPackValidationError, match="torch_compile.mode"):
        load_model_pack(_write_pack(
            tmp_path,
            opt_yaml=(
                "optimizations:\n"
                "  torch_compile:\n"
                "    enabled: true\n"
                "    mode: mega\n"
            ),
        ))


def test_invalid_fp8_scope_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelPackValidationError, match="fp8_quantization.scope"):
        load_model_pack(_write_pack(
            tmp_path,
            opt_yaml=(
                "optimizations:\n"
                "  fp8_quantization:\n"
                "    enabled: true\n"
                "    scope: full_pipeline\n"
            ),
        ))


def test_non_boolean_enabled_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelPackValidationError, match="must be a boolean"):
        load_model_pack(_write_pack(
            tmp_path,
            opt_yaml=(
                "optimizations:\n"
                "  torch_compile:\n"
                "    enabled: yeah\n"
            ),
        ))


def test_bundled_krea2_pack_ships_expected_optimizations() -> None:
    """The shipped Krea2 manifest should enable torch.compile + SageAttention.

    fp8_quantization is OFF pending torch>=2.11 + torchao>=0.17 (torch 2.9 lacks ``abs`` for the
    ``Float8_e4m3fn`` dtype so the dynamic quantizer's per-token scale computation raises). We
    still validate the field is present + parseable so re-enabling later is a one-line yaml edit.
    """

    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "models" / "packs" / "Krea2_Turbo" / "modelpack.yaml"
    if not manifest.exists():
        pytest.skip("Krea2 pack manifest not present (opt-in provisioning).")
    pack = load_model_pack(manifest)
    assert pack.optimizations.torch_compile.enabled is True
    assert pack.optimizations.torch_compile.mode == "default"
    assert pack.optimizations.fp8_quantization.enabled is False
    assert pack.optimizations.sage_attention.enabled is True
    assert pack.optimizations.tf32.enabled is True
    assert pack.optimizations.vae_tiling.enabled is True


def test_tf32_and_vae_tiling_parsed(tmp_path: Path) -> None:
    pack = load_model_pack(_write_pack(
        tmp_path,
        opt_yaml=(
            "optimizations:\n"
            "  tf32:\n"
            "    enabled: true\n"
            "  vae_tiling: true\n"
        ),
    ))
    assert pack.optimizations.tf32.enabled is True
    assert pack.optimizations.vae_tiling.enabled is True


def test_tf32_and_vae_tiling_default_disabled(tmp_path: Path) -> None:
    pack = load_model_pack(_write_pack(tmp_path))
    assert pack.optimizations.tf32.enabled is False
    assert pack.optimizations.vae_tiling.enabled is False


def test_bundled_zimage_pack_ships_expected_optimizations() -> None:
    """Z-Image: SageAttention + TF32 + VAE tiling ON, compile/fp8 OFF."""

    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "models" / "packs" / "Rayzist_bf16" / "modelpack.yaml"
    if not manifest.exists():
        pytest.skip("Rayzist_bf16 pack manifest not present.")
    pack = load_model_pack(manifest)
    assert pack.optimizations.torch_compile.enabled is False
    assert pack.optimizations.sage_attention.enabled is True
    assert pack.optimizations.fp8_quantization.enabled is False
    assert pack.optimizations.tf32.enabled is True
    assert pack.optimizations.vae_tiling.enabled is True
