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
class TF32Config:
    """Enable TF32 matmul on Ampere/Ada. Free ~1-5% on fp32 residuals.

    Ignored on pre-Ampere GPUs (Turing/Volta) where TF32 tensor cores don't exist. Ampere/Ada
    default is to opt out of TF32 in PyTorch >=1.12 for numerical determinism; this re-enables it
    at the ``high`` precision level (best of both — allow TF32 without dropping to fp16 activations).
    """

    enabled: bool = False


@dataclass(frozen=True)
class VAETilingConfig:
    """Enable VAE tiling/slicing to trim VAE decode cost at large resolutions and cut peak VRAM.

    Both ``vae.enable_tiling()`` and ``vae.enable_slicing()`` are called when available; each is
    a diffusers noop when unsupported. Universal (CUDA/CPU).
    """

    enabled: bool = False


@dataclass(frozen=True)
class OptimizationsConfig:
    torch_compile: TorchCompileConfig = TorchCompileConfig()
    fp8_quantization: Fp8QuantConfig = Fp8QuantConfig()
    sage_attention: SageAttentionConfig = SageAttentionConfig()
    tf32: TF32Config = TF32Config()
    vae_tiling: VAETilingConfig = VAETilingConfig()


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


def _supports_tf32(device: str) -> bool:
    """TF32 tensor cores are Ampere (sm_80) or newer. No-op on Turing/Volta."""
    cap = _device_capability(device)
    if cap is None:
        return False
    return cap >= (8, 0)


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


def _warn_if_fp8_storage_hook_active(pipeline: Any) -> None:
    """Log a heads-up when a fp8-storage forward hook coexists with torch.compile.

    The fp8-storage fallback (see ``_SelectiveTensorStorageHook`` in ``pipeline_factory/zimage.py``)
    swaps parameter ``.data`` between the fp8 storage dtype and the bf16 compute dtype in a
    ``forward_pre_hook`` — this mutates parameter dtype every forward, which triggers dynamo
    recompiles until it hits ``recompile_limit`` and falls back to eager for the affected sub-graph.
    Other sub-graphs still compile and net-benefit (~1.44× warm on RTX 4090 for Krea2), so we
    still enable compile — but the operator should know this is happening so they can tune
    ``torch._dynamo.config.recompile_limit`` if they see excess recompile churn in logs.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return
    for module in transformer.modules():
        if hasattr(module, "_justrayzist_fp8_storage_hook"):
            LOGGER.info(
                "torch.compile: fp8-storage forward hooks detected on the transformer. Dynamo "
                "will likely hit its recompile_limit on the modules whose parameter dtype is "
                "swapped per forward and fall back to eager there; other sub-graphs still "
                "compile. If you see excessive '/N recompile' logs, bump "
                "``torch._dynamo.config.recompile_limit``."
            )
            return


def _apply_torch_compile(pipeline: Any, cfg: TorchCompileConfig, device: str) -> bool:
    if not cfg.enabled:
        return False
    if device != "cuda":
        LOGGER.info("Skipping torch.compile: device=%s (CUDA-only).", device)
        return False
    if _skip_if_offload_active(pipeline, "torch.compile"):
        return False
    # NB: pipelines with a ``_SelectiveTensorStorageHook`` on their transformer (fp8-storage
    # backends) will trigger dynamo recompiles per gen because the hook mutates parameter dtypes
    # in ``forward_pre_hook``. Torch handles this by hitting its ``recompile_limit`` and falling
    # back to eager for the affected sub-graphs; other sub-graphs (like SageAttention shim'd
    # attention) still get compiled and give net speedup. Log a warning but don't skip — the
    # 2026-07-16 RTX 4090 bench measured a 1.44× warm gain on Krea2 with the hook present.
    _warn_if_fp8_storage_hook_active(pipeline)
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        LOGGER.info("Skipping torch.compile: pipeline has no .transformer attribute.")
        return False
    try:
        import torch

        # Inductor's cudagraphs (implicit under reduce-overhead) can conflict with pipelines that
        # mutate input tensors in place across denoise steps; the diffusers Krea2 / ZImage forwards
        # are pure, so this is safe.
        original_forward = transformer.forward
        compiled_forward = torch.compile(
            original_forward,
            mode=cfg.mode,
            fullgraph=cfg.fullgraph,
            dynamic=False,
        )

        # Wrap the compiled forward with a runtime guard: the Z-Image backend can flip a pipeline
        # from `high` full-CUDA mode to `model_offload` mid-session on VRAM pressure (see
        # ``_apply_high_runtime_fallback_if_needed`` / ``_apply_pipe_execution_mode`` in
        # backends/diffusers_zimage.py) — that installs accelerate ``_hf_hook``s AFTER we baked
        # in the compiled graph, and dynamo tracing then crashes inside
        # ``accelerate.hooks.pre_forward`` with InternalTorchDynamoError. Detect the hook at
        # each call and fall back to the pre-compile eager forward when present.
        def _compile_guard(*args, **kwargs):  # pragma: no cover - runtime guard, GPU-only path
            for module in transformer.modules():
                if hasattr(module, "_hf_hook"):
                    return original_forward(*args, **kwargs)
            return compiled_forward(*args, **kwargs)

        transformer.forward = _compile_guard
        setattr(transformer, "_rayzist_original_forward", original_forward)
        setattr(transformer, "_rayzist_compiled_forward", compiled_forward)
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


def _apply_tf32(cfg: TF32Config, device: str) -> bool:
    """Enable TF32 matmul + cuDNN TF32 on Ampere/Ada.

    Process-global setting: only needs to run once per Python session. Repeat calls are idempotent
    (torch flips the same flag each time). Kept inside the applier so a pack manifest that turns it
    off leaves the flag alone (as opposed to explicitly disabling — the pack contract is
    "enabled=true means turn on", not "enabled=false means turn off").
    """
    if not cfg.enabled:
        return False
    if not _supports_tf32(device):
        LOGGER.info(
            "Skipping TF32: GPU compute capability %s does not have TF32 tensor cores "
            "(need sm_80 / Ampere or newer).",
            _device_capability(device),
        )
        return False
    try:
        import torch

        # "high" precision keeps activation fidelity while allowing TF32 in matmul reductions —
        # the sweet spot for image models. "highest" would disable TF32 entirely.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        LOGGER.info("TF32 matmul enabled (precision=high, cuDNN TF32 on).")
        return True
    except Exception as exc:  # pragma: no cover - platform-specific
        LOGGER.warning("TF32 enable failed, continuing: %s", exc)
        return False


def _apply_vae_tiling(pipeline: Any, cfg: VAETilingConfig) -> bool:
    """Enable VAE tiling + slicing on the pipeline's VAE if the model supports it.

    Tiling splits the decode into overlapping tiles (peak VRAM ↓, small quality drift at seams);
    slicing splits the batch dimension (peak VRAM ↓, no quality drift). Both are safe to enable
    together — tiling helps single-image high-res, slicing helps batched decode.
    """
    if not cfg.enabled:
        return False
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        LOGGER.info("Skipping VAE tiling: pipeline has no .vae attribute.")
        return False
    applied = 0
    for method_name in ("enable_tiling", "enable_slicing"):
        method = getattr(vae, method_name, None)
        if callable(method):
            try:
                method()
                applied += 1
            except Exception as exc:  # pragma: no cover - model-specific
                LOGGER.warning("VAE %s() failed: %s", method_name, exc)
    if applied == 0:
        LOGGER.info("Skipping VAE tiling: this VAE class exposes neither enable_tiling nor enable_slicing.")
        return False
    LOGGER.info("VAE tiling/slicing enabled (%d method(s) applied).", applied)
    return True


def _configure_inductor_cache_dir() -> None:
    """Set a stable on-disk cache directory for Inductor + Dynamo so compile warmup persists.

    Compiled artifacts are keyed by graph hash, so the same pack + same shape reuses the cache
    across processes. Uses ``JUSTRAYZIST_INDUCTOR_CACHE_DIR`` if set, otherwise ``.build/inductor``
    at the repo root. Idempotent: repeated calls just overwrite the same env vars.
    """
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return  # honor an existing user override
    cache_root = os.environ.get("JUSTRAYZIST_INDUCTOR_CACHE_DIR")
    if not cache_root:
        # Cache next to the project's other build artifacts. Users can override with the env var.
        try:
            from app.config.settings import _resolve_root

            cache_root = str(_resolve_root() / ".build" / "inductor")
        except Exception:
            return
    try:
        os.makedirs(cache_root, exist_ok=True)
    except OSError:
        return
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_root
    # FX graph cache — cheap to keep too.
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    LOGGER.info("Inductor persistent cache: %s", cache_root)


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
    tf32: bool = False
    vae_tiling: bool = False


def apply_optimizations(
    pipeline: Any,
    cfg: OptimizationsConfig,
    device: str,
) -> AppliedOptimizations:
    """Apply pack-configured optimizations to ``pipeline`` in place.

    ``device`` is the string returned by the pipeline builder ("cuda" or "cpu"). All optimizations
    are CUDA-only and no-op on CPU (except VAE tiling which is universal). Individual
    optimizations soft-fail (log + skip) rather than raise, so a manifest can request aggressive
    settings without breaking on older GPUs.
    """
    if os.environ.get("JUSTRAYZIST_DISABLE_OPTIMIZATIONS") == "1":
        LOGGER.info("JUSTRAYZIST_DISABLE_OPTIMIZATIONS=1 set; skipping all optimizations.")
        return AppliedOptimizations()

    # Process-global setup that must happen BEFORE torch.compile so the compiler picks up the
    # cache directory and the TF32 flag on its first invocation.
    _configure_inductor_cache_dir()
    tf32 = _apply_tf32(cfg.tf32, device)

    # Per-pipeline optimizations.
    vae_tiling = _apply_vae_tiling(pipeline, cfg.vae_tiling)
    sage = _apply_sage_attention(pipeline, cfg.sage_attention, device)
    quantized = _apply_fp8_quantization(pipeline, cfg.fp8_quantization, device)
    # torch.compile is last: it snapshots the current graph, so any prior module-level mutations
    # (SageAttention shim, fp8 weight swap) need to be in place before compile traces the forward.
    compiled = _apply_torch_compile(pipeline, cfg.torch_compile, device)

    return AppliedOptimizations(
        torch_compile=compiled,
        fp8_quantization=quantized,
        sage_attention=sage,
        tf32=tf32,
        vae_tiling=vae_tiling,
    )
