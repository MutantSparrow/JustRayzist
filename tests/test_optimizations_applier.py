"""Applier tests for optimization stack — capability gates, offload skip, soft-fail.

Weightless: uses stub pipelines and monkeypatched capability probes. Never calls torchao,
sageattention, or torch.compile for real — the applier's contract is "opts in when supported,
skips loudly otherwise", so that contract is what these tests lock down.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.pipeline_factory import optimizations as opts_mod
from app.core.pipeline_factory.optimizations import (
    AppliedOptimizations,
    Fp8QuantConfig,
    OptimizationsConfig,
    SageAttentionConfig,
    TF32Config,
    TorchCompileConfig,
    VAETilingConfig,
    apply_optimizations,
)


class _FakeTransformer:
    def __init__(self) -> None:
        self.forward = lambda: None

    def modules(self):
        return iter([self])


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = _FakeTransformer()
        self.text_encoder = None


def _all_disabled() -> OptimizationsConfig:
    return OptimizationsConfig()


def _all_enabled() -> OptimizationsConfig:
    return OptimizationsConfig(
        torch_compile=TorchCompileConfig(enabled=True),
        fp8_quantization=Fp8QuantConfig(enabled=True, scope="transformer"),
        sage_attention=SageAttentionConfig(enabled=True),
        tf32=TF32Config(enabled=True),
        vae_tiling=VAETilingConfig(enabled=True),
    )


# --- No-op when nothing is requested ---


def test_all_disabled_reports_nothing_applied() -> None:
    result = apply_optimizations(_FakePipeline(), _all_disabled(), "cuda")
    assert result == AppliedOptimizations()


def test_env_kill_switch_skips_everything(monkeypatch) -> None:
    monkeypatch.setenv("JUSTRAYZIST_DISABLE_OPTIMIZATIONS", "1")
    result = apply_optimizations(_FakePipeline(), _all_enabled(), "cuda")
    assert result == AppliedOptimizations()


# --- CPU / no CUDA ---


def test_cpu_device_skips_all(monkeypatch) -> None:
    result = apply_optimizations(_FakePipeline(), _all_enabled(), "cpu")
    assert result == AppliedOptimizations()


# --- Capability gates ---


def test_fp8_skipped_on_ampere(monkeypatch) -> None:
    """Ampere (sm_86, RTX 30xx) must not attempt native fp8 matmul."""

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 6) if device == "cuda" else None)
    # Bypass the actual compile / sage calls so we're only measuring the gate.
    monkeypatch.setattr(opts_mod, "_apply_torch_compile", lambda *args, **kw: False)
    monkeypatch.setattr(opts_mod, "_apply_sage_attention", lambda *args, **kw: False)

    cfg = OptimizationsConfig(fp8_quantization=Fp8QuantConfig(enabled=True))
    result = apply_optimizations(_FakePipeline(), cfg, "cuda")
    assert result.fp8_quantization is False


def test_fp8_allowed_on_ada(monkeypatch) -> None:
    """Ada (sm_89, RTX 40xx) passes the fp8 capability gate."""

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9) if device == "cuda" else None)

    def _fake_quantize(*args, **kwargs):
        return None

    with patch.dict("sys.modules", {"torchao.quantization": SimpleNamespace(
        Float8DynamicActivationFloat8WeightConfig=lambda: object(),
        quantize_=_fake_quantize,
    )}):
        cfg = OptimizationsConfig(fp8_quantization=Fp8QuantConfig(enabled=True))
        result = apply_optimizations(_FakePipeline(), cfg, "cuda")
        assert result.fp8_quantization is True


def test_sage_skipped_below_turing(monkeypatch) -> None:
    """Pre-Turing (sm_60/70) cannot use SageAttention kernels."""

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (7, 0) if device == "cuda" else None)
    monkeypatch.setattr(opts_mod, "_apply_torch_compile", lambda *args, **kw: False)
    monkeypatch.setattr(opts_mod, "_apply_fp8_quantization", lambda *args, **kw: False)

    cfg = OptimizationsConfig(sage_attention=SageAttentionConfig(enabled=True))
    result = apply_optimizations(_FakePipeline(), cfg, "cuda")
    assert result.sage_attention is False


def test_sage_allowed_on_turing(monkeypatch) -> None:
    """Turing (sm_75, RTX 20xx) meets SageAttention's minimum."""

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (7, 5) if device == "cuda" else None)
    monkeypatch.setattr(opts_mod, "_apply_torch_compile", lambda *args, **kw: False)
    monkeypatch.setattr(opts_mod, "_apply_fp8_quantization", lambda *args, **kw: False)

    called = {"install": False}

    def _fake_install(transformer, sageattn_fn):
        called["install"] = True
        return 1  # pretend we swapped 1 attention module

    monkeypatch.setattr(opts_mod, "_install_sage_attention_processor", _fake_install)
    with patch.dict("sys.modules", {"sageattention": SimpleNamespace(sageattn=lambda *a, **k: None)}):
        cfg = OptimizationsConfig(sage_attention=SageAttentionConfig(enabled=True))
        result = apply_optimizations(_FakePipeline(), cfg, "cuda")
        assert result.sage_attention is True
        assert called["install"] is True


# --- Offload guard ---


def test_transformer_optimizations_skipped_when_pipeline_has_accelerate_hooks(monkeypatch) -> None:
    """When ``enable_sequential_cpu_offload`` has installed hooks on the transformer, the
    optimizations that mutate the transformer (compile, fp8 quant, sage) must not run.
    """

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9) if device == "cuda" else None)
    hooked_transformer = _FakeTransformer()
    hooked_transformer._hf_hook = object()  # marker that accelerate wraps this module
    pipe = _FakePipeline()
    pipe.transformer = hooked_transformer

    # TF32 disabled for this test so we don't touch real torch state and hit unrelated setup bugs.
    cfg = OptimizationsConfig(
        torch_compile=TorchCompileConfig(enabled=True),
        fp8_quantization=Fp8QuantConfig(enabled=True),
        sage_attention=SageAttentionConfig(enabled=True),
    )
    result = apply_optimizations(pipe, cfg, "cuda")
    assert result.torch_compile is False
    assert result.fp8_quantization is False
    assert result.sage_attention is False


# --- Soft-fail on optimizer exception ---


def test_soft_fail_on_missing_optional_dep(monkeypatch) -> None:
    """When sageattention is unimportable at apply time the sage branch must soft-fail."""

    import sys

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(opts_mod, "_apply_torch_compile", lambda *a, **kw: False)
    monkeypatch.setattr(opts_mod, "_apply_fp8_quantization", lambda *a, **kw: False)
    # Force the sageattention import inside _apply_sage_attention to explode.
    monkeypatch.setitem(sys.modules, "sageattention", None)

    cfg = OptimizationsConfig(sage_attention=SageAttentionConfig(enabled=True))
    result = apply_optimizations(_FakePipeline(), cfg, "cuda")
    assert result.sage_attention is False


def test_pipeline_without_transformer_skips_everything(monkeypatch) -> None:
    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9))
    pipe = SimpleNamespace(text_encoder=None, vae=None)  # no .transformer, no .vae
    result = apply_optimizations(pipe, _all_enabled(), "cuda")
    assert result.torch_compile is False
    assert result.fp8_quantization is False
    assert result.sage_attention is False
    assert result.vae_tiling is False


# --- TF32 ---


def test_tf32_skipped_on_turing(monkeypatch) -> None:
    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (7, 5))
    cfg = OptimizationsConfig(tf32=TF32Config(enabled=True))
    result = apply_optimizations(_FakePipeline(), cfg, "cuda")
    assert result.tf32 is False


def test_tf32_applied_on_ampere(monkeypatch) -> None:
    """Ampere/Ada should flip the TF32 flags; older GPUs skip.

    Uses a fake torch module so the test doesn't touch the real process's TF32 flags.
    """

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 6))

    calls: dict[str, object] = {}

    class _FakeCudaMatmul:
        allow_tf32 = False

    class _FakeCudnn:
        allow_tf32 = False

    fake_torch = SimpleNamespace(
        set_float32_matmul_precision=lambda level: calls.setdefault("precision", level),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=_FakeCudaMatmul()),
            cudnn=_FakeCudnn(),
        ),
    )
    with patch.dict("sys.modules", {"torch": fake_torch}):
        cfg = OptimizationsConfig(tf32=TF32Config(enabled=True))
        result = apply_optimizations(_FakePipeline(), cfg, "cuda")
        assert result.tf32 is True
        assert calls["precision"] == "high"


# --- VAE tiling ---


def test_vae_tiling_calls_enable_methods(monkeypatch) -> None:
    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9))

    calls: list[str] = []

    class _FakeVAE:
        def enable_tiling(self):
            calls.append("tiling")

        def enable_slicing(self):
            calls.append("slicing")

    pipe = _FakePipeline()
    pipe.vae = _FakeVAE()

    cfg = OptimizationsConfig(vae_tiling=VAETilingConfig(enabled=True))
    result = apply_optimizations(pipe, cfg, "cuda")
    assert result.vae_tiling is True
    assert calls == ["tiling", "slicing"]


def test_vae_tiling_noop_when_vae_lacks_methods(monkeypatch) -> None:
    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9))
    pipe = _FakePipeline()
    pipe.vae = SimpleNamespace()  # no enable_tiling, no enable_slicing
    cfg = OptimizationsConfig(vae_tiling=VAETilingConfig(enabled=True))
    result = apply_optimizations(pipe, cfg, "cuda")
    assert result.vae_tiling is False


# --- Inductor cache setup ---


def test_inductor_cache_dir_set(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("JUSTRAYZIST_INDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setenv("JUSTRAYZIST_INDUCTOR_CACHE_DIR", str(tmp_path / "ind"))
    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9))

    apply_optimizations(_FakePipeline(), OptimizationsConfig(), "cuda")

    assert opts_mod.os.environ.get("TORCHINDUCTOR_CACHE_DIR") == str(tmp_path / "ind")
    assert opts_mod.os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE") == "1"


def test_inductor_cache_dir_respects_user_override(monkeypatch) -> None:
    """If the user set ``TORCHINDUCTOR_CACHE_DIR`` we don't stomp it."""

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/user_cache")
    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9))

    apply_optimizations(_FakePipeline(), OptimizationsConfig(), "cuda")

    assert opts_mod.os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/user_cache"
