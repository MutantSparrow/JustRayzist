"""Pipeline builders for the Krea2-Turbo model family.

Mirrors ``pipeline_factory/zimage.py`` but targets Krea2's component classes
(``Krea2Transformer2DModel``, ``AutoencoderKLQwenImage``, ``Qwen3VLModel``) and the
``Krea2Pipeline``. Krea2 and Z-Image are near-siblings (both flow-matching, Qwen-conditioned,
Qwen-VAE DiT turbo models), so the shared loading/storage helpers in ``zimage.py`` are reused
directly rather than duplicated.

Krea2's diffusers classes require diffusers ``>=0.39.0``. The Krea classes are imported lazily
so environments with older diffusers still start; if the installed build lacks the Krea symbols,
a clear error with the setup-repair hint is raised at pack-load time instead of failing at
module import. This keeps the Z-Image path completely undisturbed on older pins.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from app.config.profiles import RuntimeProfile
from app.core.model_registry import ModelPack
from app.core.platform_guidance import setup_repair_hint
from app.core.pipeline_factory.qwen import _load_text_encoder_from_local_config
from app.core.pipeline_factory.krea_comfy_convert import (
    convert_comfy_scaled_fp8_encoder_state_dict,
    convert_native_krea2_transformer_state_dict,
    is_comfy_scaled_fp8_encoder,
    is_native_krea2_transformer,
)
from app.core.pipeline_factory.zimage import (
    _apply_fp8_storage_fallback_materialization,
    _checkpoint_contains_fp8_weights,
    _load_state_dict_preserving_dtypes,
    _normalize_fp8_state_dict_for_runtime,
    _pick_component,
    _resolve_dtype,
    _stage_weight,
)

LOGGER = logging.getLogger(__name__)

KREA_ARCHITECTURE = "krea2_turbo"


def _load_native_krea_transformer(
    *,
    transformer_component: Any,
    config_dir: Any,
    transformer_cls: Any,
    torch_module: Any,
    compute_dtype: Any,
    enable_real_fp8_checkpoint_support: bool,
) -> tuple[Any, dict[str, Any]]:
    """Build a Krea2Transformer2DModel from local config and a native/ComfyUI state dict.

    Returns ``(model, fp8_metadata)``. Mirrors the Z-Image fp8 handling: fp8 storage tensors are
    preserved via a materialization hook when the fp8 backend is active, otherwise promoted to the
    compute dtype.
    """
    from safetensors.torch import load_file

    if transformer_component.file_format != "safetensors":
        raise ValueError(
            f"Unsupported transformer format '{transformer_component.file_format}' for Krea pack."
        )

    raw_state = load_file(str(transformer_component.path))
    if is_native_krea2_transformer(raw_state.keys()):
        LOGGER.debug("Detected native/ComfyUI Krea2 transformer format; applying key conversion.")
        state_dict = convert_native_krea2_transformer_state_dict(raw_state)
    else:
        state_dict = raw_state

    has_fp8 = _checkpoint_contains_fp8_weights(transformer_component.path)

    fp8_meta: dict[str, Any] = {
        "fp8_checkpoint": bool(has_fp8),
        "fp8_fallback_used": False,
        "fp8_fallback_reason": None,
        "fp8_runtime_mode": None,
        "fp8_normalized_tensor_count": 0,
        "fp8_storage_preserved_tensor_count": 0,
        "fp8_promoted_tensor_count": 0,
        "fp8_normalized_tensor_names": (),
    }

    if has_fp8 and enable_real_fp8_checkpoint_support:
        preparation = _normalize_fp8_state_dict_for_runtime(
            state_dict,
            compute_dtype=compute_dtype,
            torch_module=torch_module,
        )
        config = transformer_cls.load_config(str(config_dir))
        model = transformer_cls.from_config(config)
        model = _load_state_dict_preserving_dtypes(model, preparation.state_dict)
        storage_meta = _apply_fp8_storage_fallback_materialization(
            model=model,
            torch_module=torch_module,
            compute_dtype=compute_dtype,
        )
        fp8_meta.update(
            {
                "fp8_fallback_used": True,
                "fp8_fallback_reason": "native_fp8_not_implemented",
                "fp8_runtime_mode": "bf16_with_fp8_storage_fallback",
                "fp8_normalized_tensor_count": preparation.normalized_tensor_count,
                "fp8_storage_preserved_tensor_count": storage_meta[
                    "fp8_storage_preserved_tensor_count"
                ],
                "fp8_promoted_tensor_count": preparation.promoted_tensor_count,
                "fp8_normalized_tensor_names": preparation.promoted_names[:16],
            }
        )
        return model, fp8_meta

    # bf16 path (or fp8 checkpoint loaded without the real-fp8 backend): promote everything to the
    # compute dtype so the model runs in bf16/fp16.
    promoted = {
        name: (tensor.to(dtype=compute_dtype) if tensor.is_floating_point() else tensor)
        for name, tensor in state_dict.items()
    }
    config = transformer_cls.load_config(str(config_dir))
    model = transformer_cls.from_config(config)
    model = _load_state_dict_preserving_dtypes(model, promoted)
    return model, fp8_meta


def _load_native_krea_vae(
    *,
    vae_component: Any,
    config_dir: Any,
    vae_cls: Any,
    dtype: Any,
) -> Any:
    """Build AutoencoderKLQwenImage from config and a (possibly native/ComfyUI) state dict.

    Native Qwen-image VAE checkpoints share the Wan VAE key layout; diffusers'
    ``convert_wan_vae_to_diffusers`` remaps them exactly. Already-diffusers checkpoints load as-is.
    """
    from safetensors.torch import load_file

    if vae_component.file_format != "safetensors":
        raise ValueError(f"Unsupported VAE format '{vae_component.file_format}' for Krea pack.")

    raw = load_file(str(vae_component.path))
    model = vae_cls.from_config(vae_cls.load_config(str(config_dir)))
    expected = set(model.state_dict().keys())
    if set(raw.keys()) != expected:
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

        LOGGER.debug("Converting native/ComfyUI VAE keys via convert_wan_vae_to_diffusers.")
        raw = convert_wan_vae_to_diffusers(raw)
    raw = {k: (v.to(dtype=dtype) if v.is_floating_point() else v) for k, v in raw.items()}
    model.load_state_dict(raw, strict=True)
    return model.to(dtype=dtype)


def _load_krea_text_encoder(
    *,
    text_encoder_component: Any,
    config_dir: Any,
    torch_module: Any,
    compute_dtype: Any,
) -> Any:
    """Load the Qwen3VL text encoder, handling the ComfyUI scaled-fp8 layout.

    ComfyUI fp8 encoders store per-linear ``.weight`` (fp8) + ``.weight_scale`` + ``.comfy_quant``
    metadata; these are dequantized to ``compute_dtype`` and loaded into ``Qwen3VLModel`` built from
    the local config. Non-ComfyUI encoders fall back to the shared Qwen loader.
    """
    from safetensors.torch import load_file

    if text_encoder_component.file_format != "safetensors":
        # gguf / other: reuse the shared loader.
        return _load_text_encoder_from_local_config(
            component_path=text_encoder_component.path,
            config_dir=config_dir,
            dtype=compute_dtype,
            local_files_only=True,
            gguf_file=text_encoder_component.path.name,
        )

    raw = load_file(str(text_encoder_component.path))
    if not is_comfy_scaled_fp8_encoder(raw.keys()):
        # Standard checkpoint: stage next to the config and use the shared loader.
        _stage_weight(text_encoder_component.path, config_dir / text_encoder_component.path.name)
        return _load_text_encoder_from_local_config(
            component_path=text_encoder_component.path,
            config_dir=config_dir,
            dtype=compute_dtype,
            local_files_only=True,
        )

    LOGGER.debug("Dequantizing ComfyUI scaled-fp8 Qwen3VL encoder to %s.", compute_dtype)
    state_dict, dequantized = convert_comfy_scaled_fp8_encoder_state_dict(
        raw, compute_dtype=compute_dtype
    )
    LOGGER.debug("Dequantized %d fp8 encoder tensors.", dequantized)

    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            model = AutoModel.from_config(config)
    except Exception:
        model = AutoModel.from_config(config)
    model.load_state_dict(state_dict, strict=False, assign=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model.to(dtype=compute_dtype)


@dataclass(frozen=True)
class LoadedKreaPipeline:
    """Container mirroring ``LoadedZImagePipeline``.

    Field names intentionally match ``LoadedZImagePipeline`` so the inherited
    ``DiffusersZImageBackend`` code paths (which read ``loaded.pipeline`` / ``loaded.device`` /
    ``loaded.dtype_name``) operate unchanged under the Krea backend (Tier A of the plan).
    """

    pipeline: Any
    device: str
    dtype_name: str
    backend_name: str = "diffusers_krea"
    fp8_checkpoint: bool = False
    fp8_fallback_used: bool = False
    fp8_fallback_reason: str | None = None
    fp8_runtime_mode: str | None = None
    fp8_normalized_tensor_count: int = 0
    fp8_storage_preserved_tensor_count: int = 0
    fp8_promoted_tensor_count: int = 0
    fp8_normalized_tensor_names: tuple[str, ...] = ()


def _import_krea_classes() -> tuple[Any, Any, Any]:
    """Lazily import Krea2 diffusers classes.

    Returns ``(Krea2Pipeline, Krea2Transformer2DModel, AutoencoderKLQwenImage)``.
    Raises a descriptive ``ImportError`` if the installed diffusers build predates Krea2 support.
    """

    try:
        from diffusers import (  # type: ignore[attr-defined]
            AutoencoderKLQwenImage,
            Krea2Pipeline,
            Krea2Transformer2DModel,
        )
    except ImportError as exc:
        raise ImportError(
            "Installed diffusers build is missing Krea2 classes "
            "(Krea2Pipeline / Krea2Transformer2DModel / AutoencoderKLQwenImage). "
            "Krea2-Turbo requires diffusers >=0.39.0. "
            + setup_repair_hint()
        ) from exc
    return Krea2Pipeline, Krea2Transformer2DModel, AutoencoderKLQwenImage


def _prefer_expandable_cuda_segments() -> None:
    """Reduce CUDA fragmentation for the 12B Krea transformer, unless the user configured it.

    The fp8-storage/bf16-compute transformer plus the Qwen3VL encoder are large relative to
    consumer VRAM (e.g. a 16GB RTX 4080), and CUDA allocator fragmentation can trigger spurious
    OOMs mid-denoise. Setting ``expandable_segments:True`` avoids this. Must be set before CUDA
    initializes; skipped if ``PYTORCH_CUDA_ALLOC_CONF`` is already set or CUDA is already up.
    """
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return
    except Exception:
        pass
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _build_krea_pipeline(
    pack: ModelPack,
    profile: RuntimeProfile,
    *,
    backend_name: str,
    enable_real_fp8_checkpoint_support: bool,
) -> LoadedKreaPipeline:
    if pack.architecture != KREA_ARCHITECTURE:
        raise ValueError(
            f"Unsupported architecture '{pack.architecture}' for Krea pipeline builder."
        )
    if pack.pipeline_config_dir is None:
        raise ValueError(
            "Model pack is missing 'pipeline_config_dir'. "
            "A local diffusers config directory is required."
        )

    _prefer_expandable_cuda_segments()

    import torch

    (
        Krea2Pipeline,
        Krea2Transformer2DModel,
        AutoencoderKLQwenImage,
    ) = _import_krea_classes()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(torch, device)
    kwargs: dict[str, Any] = {"local_files_only": True}
    pipeline_metadata: dict[str, Any] = {
        "fp8_checkpoint": False,
        "fp8_fallback_used": False,
        "fp8_fallback_reason": None,
        "fp8_runtime_mode": None,
        "fp8_normalized_tensor_count": 0,
        "fp8_storage_preserved_tensor_count": 0,
        "fp8_promoted_tensor_count": 0,
        "fp8_normalized_tensor_names": (),
    }

    transformer_component = _pick_component(pack, "transformer", "checkpoint")
    vae_component = _pick_component(pack, "vae")
    text_encoder_component = _pick_component(pack, "text_encoder", "encoder")
    if transformer_component is None or vae_component is None or text_encoder_component is None:
        raise ValueError(
            "Krea pack requires transformer, vae, and text_encoder components."
        )

    config_dir = pack.pipeline_config_dir

    # --- Transformer (Krea2Transformer2DModel) ---
    # Krea2Transformer2DModel has no from_single_file, and these native/ComfyUI checkpoints use
    # `blocks.*`-style keys, so the model is built from its local config and loaded from a converted
    # state dict (native -> diffusers key remap). See krea_comfy_convert.py.
    transformer, transformer_fp8_meta = _load_native_krea_transformer(
        transformer_component=transformer_component,
        config_dir=config_dir / "transformer",
        transformer_cls=Krea2Transformer2DModel,
        torch_module=torch,
        compute_dtype=dtype,
        enable_real_fp8_checkpoint_support=enable_real_fp8_checkpoint_support,
    )
    pipeline_metadata.update(transformer_fp8_meta)

    # --- VAE (AutoencoderKLQwenImage) ---
    # AutoencoderKLQwenImage has no from_single_file. The Qwen-image VAE shares the Wan VAE key
    # layout, so diffusers' own convert_wan_vae_to_diffusers maps the native/ComfyUI keys exactly
    # (validated: 194/194 keys, zero mismatch). Build from config and load the converted dict.
    LOGGER.debug("Loading Krea VAE component from %s", vae_component.path)
    vae = _load_native_krea_vae(
        vae_component=vae_component,
        config_dir=config_dir / "vae",
        vae_cls=AutoencoderKLQwenImage,
        dtype=dtype,
    )

    # --- Text encoder (Qwen3VLModel) + tokenizer ---
    text_encoder = _load_krea_text_encoder(
        text_encoder_component=text_encoder_component,
        config_dir=config_dir / "text_encoder",
        torch_module=torch,
        compute_dtype=dtype,
    )

    from transformers import AutoTokenizer

    tokenizer_dir = config_dir / "tokenizer"
    tokenizer_source = tokenizer_dir if tokenizer_dir.exists() else config_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=True)

    # --- Scheduler ---
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config_dir / "scheduler"),
        local_files_only=True,
    )

    # --- Assemble the pipeline from components (no from_pretrained dir) ---
    # The distilled turbo checkpoint uses is_distilled=True (fixed mu=1.15); text_encoder_select_layers
    # defaults to the Krea2 Qwen3-VL taps when None.
    LOGGER.debug("Assembling Krea2Pipeline from local components.")
    pipeline = Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)

    if device == "cuda":
        if profile.enable_sequential_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif profile.enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")

    # Apply pack-configured runtime optimizations (torch.compile / fp8 quant / SageAttention).
    # Each is capability-gated; unsupported combinations soft-fail with a log line. Deferred to
    # here so pipeline device placement (or offload hook wiring) has finished first.
    from app.core.pipeline_factory.optimizations import apply_optimizations

    apply_optimizations(pipeline, pack.optimizations, device)

    return LoadedKreaPipeline(
        pipeline=pipeline,
        device=device,
        dtype_name=str(dtype),
        backend_name=backend_name,
        fp8_checkpoint=bool(pipeline_metadata["fp8_checkpoint"]),
        fp8_fallback_used=bool(pipeline_metadata["fp8_fallback_used"]),
        fp8_fallback_reason=pipeline_metadata["fp8_fallback_reason"],
        fp8_runtime_mode=pipeline_metadata["fp8_runtime_mode"],
        fp8_normalized_tensor_count=int(pipeline_metadata["fp8_normalized_tensor_count"]),
        fp8_storage_preserved_tensor_count=int(
            pipeline_metadata["fp8_storage_preserved_tensor_count"]
        ),
        fp8_promoted_tensor_count=int(pipeline_metadata["fp8_promoted_tensor_count"]),
        fp8_normalized_tensor_names=tuple(pipeline_metadata["fp8_normalized_tensor_names"]),
    )


def build_krea_pipeline(pack: ModelPack, profile: RuntimeProfile) -> LoadedKreaPipeline:
    return _build_krea_pipeline(
        pack,
        profile,
        backend_name="diffusers_krea",
        enable_real_fp8_checkpoint_support=False,
    )


def build_fp8_krea_pipeline(pack: ModelPack, profile: RuntimeProfile) -> LoadedKreaPipeline:
    return _build_krea_pipeline(
        pack,
        profile,
        backend_name="fp8_krea",
        enable_real_fp8_checkpoint_support=True,
    )
