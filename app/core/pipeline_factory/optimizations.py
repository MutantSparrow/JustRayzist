"""Post-load optimization pipeline for image-generation backends.

Applies a set of pack-configured runtime optimizations to a freshly-built ``DiffusionPipeline``
after it has been placed on-device. Each optimization has a hardware capability gate so a pack
manifest that requests everything degrades gracefully on older GPUs rather than crashing.

Currently supported:

* **torch.compile** — Inductor-compiles the transformer's forward. Universal (all CUDA).
* **torchao fp8 dynamic quantization** — swaps ``nn.Linear`` weights for the
  ``Float8DynamicActivationFloat8Weight`` layout so the fp8 storage the pack already ships can be
  matmul'd natively instead of promoted to bf16 per step. Requires Ada/Hopper (sm_89+).
* **SageAttention** — replaces the diffusers-registered attention processor with SageAttention
  kernels. Turing (sm_75) gets the v1 path; Ampere/Ada get the faster kernels.

The applier is *architecture-agnostic*: any pipeline exposing a ``.transformer`` submodule with an
``nn.Module.forward`` can be optimized. Both ``ZImagePipeline`` and ``Krea2Pipeline`` qualify.

Config comes from a validated dataclass built at pack-load time (see
``app/core/model_registry/model_pack.py::_parse_optimizations``). Environment overrides are NOT
consulted here — the pack yaml is the source of truth per the /caveman:caveman product decision
(2026-07-16). If a knob must be forced off in the field, edit the pack manifest.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TorchCompileConfig:
    enabled: bool = False
    mode: str = "reduce-overhead"  # inductor mode: "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False


@dataclass(frozen=True)
class Fp8QuantConfig:
    enabled: bool = False
    scope: str = "transformer"  # transformer | transformer+text_encoder


@dataclass(frozen=True)
class SageAttentionConfig:
    enabled: bool = False


@dataclass(frozen=True)
class OptimizationsConfig:
    torch_compile: TorchCompileConfig = TorchCompileConfig()
    fp8_quantization: Fp8QuantConfig = Fp8QuantConfig()
    sage_attention: SageAttentionConfig = SageAttentionConfig()


# --- Capability detection -----------------------------------------------------


def _device_capability(device: str) -> tuple[int, int] | None:
    """Return the CUDA compute capability of ``device`` or None if not CUDA."""
    if device != "cuda":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:  # pragma: no cover - defensive
        return None


def _supports_fp8_compute(device: str) -> bool:
    """Native fp8 matmul requires Ada (sm_89) or newer."""
    cap = _device_capability(device)
    if cap is None:
        return False
    return cap >= (8, 9)


def _supports_sage_attention(device: str) -> bool:
    """SageAttention 1.x requires Turing or newer (sm_75+)."""
    cap = _device_capability(device)
    if cap is None:
        return False
    return cap >= (7, 5)


# --- Individual optimizers ----------------------------------------------------


def _skip_if_offload_active(pipeline: Any, name: str) -> bool:
    """Skip in-place mutations when the pipeline is under accelerate offload hooks.

    Both ``enable_sequential_cpu_offload`` and ``enable_model_cpu_offload`` wrap submodules with
    accelerate ``pre_forward`` hooks that expect their parameters to live on meta placeholders
    with real storage staged on demand. Compiling the forward or swapping ``nn.Linear`` weights
    with quantized tensors interacts badly with those hooks (compiled graph gets baked with
    device-mismatched ops; swapped params drop the accelerate metadata). Detect and skip.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return False
    for module in transformer.modules():
        if hasattr(module, "_hf_hook"):
            LOGGER.info(
                "Skipping %s: pipeline is under accelerate CPU-offload hooks; the offload path "
                "is incompatible with this optimization.",
                name,
            )
            return True
    return False


def _apply_torch_compile(pipeline: Any, cfg: TorchCompileConfig, device: str) -> bool:
    if not cfg.enabled:
        return False
    if device != "cuda":
        LOGGER.info("Skipping torch.compile: device=%s (CUDA-only).", device)
        return False
    if _skip_if_offload_active(pipeline, "torch.compile"):
        return False
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        LOGGER.info("Skipping torch.compile: pipeline has no .transformer attribute.")
        return False
    try:
        import torch

        # Inductor's cudagraphs (implicit under reduce-overhead) can conflict with pipelines that
        # mutate input tensors in place across denoise steps; the diffusers Krea2 / ZImage forwards
        # are pure, so this is safe.
        transformer.forward = torch.compile(
            transformer.forward,
            mode=cfg.mode,
            fullgraph=cfg.fullgraph,
            dynamic=False,
        )
        LOGGER.info("torch.compile applied to transformer (mode=%s).", cfg.mode)
        return True
    except Exception as exc:  # pragma: no cover - hardware/version-specific
        LOGGER.warning("torch.compile failed, continuing uncompiled: %s", exc)
        return False


def _apply_fp8_quantization(pipeline: Any, cfg: Fp8QuantConfig, device: str) -> bool:
    if not cfg.enabled:
        return False
    if not _supports_fp8_compute(device):
        LOGGER.info(
            "Skipping fp8 quantization: GPU compute capability %s does not support native fp8 "
            "matmul (need sm_89 / Ada or newer).",
            _device_capability(device),
        )
        return False
    if _skip_if_offload_active(pipeline, "fp8 quantization"):
        return False
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        LOGGER.info("Skipping fp8 quantization: pipeline has no .transformer attribute.")
        return False
    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )

        # Filter to Linear modules so the quantizer doesn't try to swap non-linear layers.
        import torch.nn as nn

        def _filter_linear(module: Any, _fqn: str) -> bool:
            return isinstance(module, nn.Linear)

        quantize_(
            transformer,
            Float8DynamicActivationFloat8WeightConfig(),
            filter_fn=_filter_linear,
        )
        LOGGER.info("fp8 dynamic quantization applied to transformer linears.")
        if cfg.scope == "transformer+text_encoder":
            text_encoder = getattr(pipeline, "text_encoder", None)
            if text_encoder is not None:
                quantize_(
                    text_encoder,
                    Float8DynamicActivationFloat8WeightConfig(),
                    filter_fn=_filter_linear,
                )
                LOGGER.info("fp8 dynamic quantization applied to text_encoder linears.")
        return True
    except Exception as exc:  # pragma: no cover - hardware/version-specific
        LOGGER.warning("fp8 quantization failed, continuing without: %s", exc)
        return False


def _apply_sage_attention(pipeline: Any, cfg: SageAttentionConfig, device: str) -> bool:
    if not cfg.enabled:
        return False
    if not _supports_sage_attention(device):
        LOGGER.info(
            "Skipping SageAttention: GPU compute capability %s does not support Sage kernels "
            "(need sm_75 / Turing or newer).",
            _device_capability(device),
        )
        return False
    if _skip_if_offload_active(pipeline, "SageAttention"):
        return False
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        LOGGER.info("Skipping SageAttention: pipeline has no .transformer attribute.")
        return False
    try:
        from sageattention import sageattn

        replaced = _install_sage_attention_processor(transformer, sageattn)
        if replaced == 0:
            LOGGER.info("SageAttention: no attention processors found on transformer; skipped.")
            return False
        LOGGER.info("SageAttention installed on %d attention module(s).", replaced)
        return True
    except Exception as exc:  # pragma: no cover - hardware/version-specific
        LOGGER.warning("SageAttention setup failed, continuing with default attention: %s", exc)
        return False


def _install_sage_attention_processor(transformer: Any, sageattn_fn: Any) -> int:
    """Monkey-patch the diffusers ``F.scaled_dot_product_attention`` calls on every attention
    module of the transformer to route through Sage.

    Both ``ZImageTransformer2DModel`` and ``Krea2Transformer2DModel`` use SDPA under the hood via
    the standard diffusers attention processor. Rather than writing a bespoke ``AttnProcessor``
    subclass for each transformer (which would need to know each's attention math), we swap the
    reference to ``F.scaled_dot_product_attention`` inside the transformer's module namespace with
    a Sage-backed shim. The shim signature matches PyTorch's ``scaled_dot_product_attention`` so
    every call site inside the transformer redirects transparently.
    """
    import torch.nn.functional as F

    def _sage_shim(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # SageAttention's ``sageattn`` accepts (q, k, v, is_causal=?, sm_scale=?).
        # It does not accept an ``attn_mask``; fall back to torch SDPA when one is provided.
        if attn_mask is not None or dropout_p != 0.0:
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        # Sage returns the same layout as SDPA (B, H, N, D).
        return sageattn_fn(query, key, value, is_causal=is_causal, sm_scale=scale)

    # Diffusers attention modules import scaled_dot_product_attention as ``F.scaled_dot_product_attention``
    # and call it from ``diffusers.models.attention_processor``. Patch the module's namespace.
    from diffusers.models import attention_processor as ap

    original = getattr(ap, "_rayzist_original_sdpa", None)
    if original is None:
        original = ap.F.scaled_dot_product_attention
        setattr(ap, "_rayzist_original_sdpa", original)

    # Install a wrapper on the module-level F so all diffusers attention paths route through Sage.
    class _FShim:
        scaled_dot_product_attention = staticmethod(_sage_shim)

        def __getattr__(self, name: str) -> Any:
            return getattr(F, name)

    ap.F = _FShim()  # type: ignore[assignment]

    # Count attention modules (heuristic: any submodule with ``to_q``/``to_k``/``to_v`` or ``qkv``).
    count = 0
    for module in transformer.modules():
        has_qkv = (
            hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v")
        ) or hasattr(module, "qkv")
        if has_qkv:
            count += 1
    return count


# --- Public entry -------------------------------------------------------------


@dataclass(frozen=True)
class AppliedOptimizations:
    """Report of which optimizations landed vs. which were skipped by capability."""

    torch_compile: bool = False
    fp8_quantization: bool = False
    sage_attention: bool = False


def apply_optimizations(
    pipeline: Any,
    cfg: OptimizationsConfig,
    device: str,
) -> AppliedOptimizations:
    """Apply pack-configured optimizations to ``pipeline`` in place.

    ``device`` is the string returned by the pipeline builder ("cuda" or "cpu"). All optimizations
    are CUDA-only and no-op on CPU. Individual optimizations soft-fail (log + skip) rather than
    raise, so a manifest can request aggressive settings without breaking on older GPUs.
    """
    # Skip when running the test harness with a JUSTRAYZIST_DISABLE_OPTIMIZATIONS=1 env — same
    # escape hatch we already use for other hot paths in tests.
    if os.environ.get("JUSTRAYZIST_DISABLE_OPTIMIZATIONS") == "1":
        LOGGER.info("JUSTRAYZIST_DISABLE_OPTIMIZATIONS=1 set; skipping all optimizations.")
        return AppliedOptimizations()

    compiled = _apply_torch_compile(pipeline, cfg.torch_compile, device)
    quantized = _apply_fp8_quantization(pipeline, cfg.fp8_quantization, device)
    sage = _apply_sage_attention(pipeline, cfg.sage_attention, device)
    return AppliedOptimizations(
        torch_compile=compiled,
        fp8_quantization=quantized,
        sage_attention=sage,
    )
