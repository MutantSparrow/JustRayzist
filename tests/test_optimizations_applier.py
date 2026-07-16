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
    TorchCompileConfig,
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


def test_all_skipped_when_pipeline_has_accelerate_hooks(monkeypatch) -> None:
    """When ``enable_sequential_cpu_offload`` has installed hooks, mutating optimizations must not run."""

    monkeypatch.setattr(opts_mod, "_device_capability", lambda device: (8, 9) if device == "cuda" else None)
    hooked_transformer = _FakeTransformer()
    hooked_transformer._hf_hook = object()  # marker that accelerate wraps this module
    pipe = _FakePipeline()
    pipe.transformer = hooked_transformer

    result = apply_optimizations(pipe, _all_enabled(), "cuda")
    assert result == AppliedOptimizations()


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
    pipe = SimpleNamespace(text_encoder=None)  # no .transformer
    result = apply_optimizations(pipe, _all_enabled(), "cuda")
    assert result == AppliedOptimizations()
