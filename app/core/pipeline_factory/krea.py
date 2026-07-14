"""Pipeline builders for the Krea2-Turbo model family.

Mirrors ``pipeline_factory/zimage.py`` but targets Krea2's component classes
(``Krea2Transformer2DModel``, ``AutoencoderKLQwenImage``, ``Qwen3VLModel``) and the
``Krea2Pipeline``. Krea2 and Z-Image are near-siblings (both flow-matching, Qwen-conditioned,
Qwen-VAE DiT turbo models), so the shared loading/storage helpers in ``zimage.py`` are reused
directly rather than duplicated.

WP-0 gate (see JustRayzist-Krea.md §6.1): Krea2's diffusers classes require
diffusers ``>=0.39.0.dev0`` (from source), while the repo pins ``diffusers>=0.36.0``. The Krea
classes are therefore imported lazily and, if the installed diffusers build lacks them, a clear
error with the setup-repair hint is raised instead of failing at import time. This keeps the
Z-Image path completely undisturbed on the current pin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config.profiles import RuntimeProfile
from app.core.model_registry import ModelPack
from app.core.platform_guidance import setup_repair_hint
from app.core.pipeline_factory.qwen import _load_text_encoder_from_local_config
from app.core.pipeline_factory.zimage import (
    _checkpoint_contains_fp8_weights,
    _configure_component_storage,
    _load_fp8_transformer_for_runtime,
    _load_transformer_from_state_dict,
    _model_contains_dtype,
    _pick_component,
    _resolve_dtype,
    _stage_weight,
)

LOGGER = logging.getLogger(__name__)

KREA_ARCHITECTURE = "krea2_turbo"


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
            "Krea2-Turbo requires diffusers >=0.39.0.dev0 (see JustRayzist-Krea.md WP-0). "
            + setup_repair_hint()
        ) from exc
    return Krea2Pipeline, Krea2Transformer2DModel, AutoencoderKLQwenImage


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

    if text_encoder_component and text_encoder_component.file_format == "safetensors":
        _stage_weight(
            text_encoder_component.path,
            pack.pipeline_config_dir / "text_encoder" / "model.safetensors",
        )

    # --- Transformer (Krea2Transformer2DModel) ---
    if transformer_component and transformer_component.file_format == "safetensors":
        LOGGER.debug("Loading Krea transformer component from %s", transformer_component.path)
        transformer_has_fp8_weights = _checkpoint_contains_fp8_weights(transformer_component.path)
        if transformer_has_fp8_weights and enable_real_fp8_checkpoint_support:
            transformer, fp8_metadata = _load_fp8_transformer_for_runtime(
                transformer_component=transformer_component,
                config_dir=pack.pipeline_config_dir / "transformer",
                zimage_transformer_cls=Krea2Transformer2DModel,
                torch_module=torch,
                compute_dtype=dtype,
            )
            pipeline_metadata.update(fp8_metadata)
        elif transformer_has_fp8_weights:
            from safetensors.torch import load_file

            transformer = _load_transformer_from_state_dict(
                state_dict=load_file(str(transformer_component.path)),
                config_dir=pack.pipeline_config_dir / "transformer",
                zimage_transformer_cls=Krea2Transformer2DModel,
            )
        else:
            transformer = Krea2Transformer2DModel.from_single_file(
                str(transformer_component.path),
                config=str(pack.pipeline_config_dir / "transformer"),
                dtype=dtype,
                local_files_only=True,
            )
        if (
            transformer_has_fp8_weights
            and not enable_real_fp8_checkpoint_support
            and not _model_contains_dtype(transformer, getattr(torch, "float8_e4m3fn"))
        ):
            raise ValueError(
                f"Transformer checkpoint '{transformer_component.path}' contains FP8 tensors on disk "
                "but the loaded model has no FP8 parameters or buffers."
            )
        if not (enable_real_fp8_checkpoint_support and transformer_has_fp8_weights):
            transformer = _configure_component_storage(
                model=transformer,
                component=transformer_component,
                torch_module=torch,
                device=device,
                default_compute_dtype=dtype,
                preserve_existing_storage=transformer_has_fp8_weights,
            )
        kwargs["transformer"] = transformer
    elif transformer_component:
        raise ValueError(
            f"Unsupported transformer format '{transformer_component.file_format}' for Krea pack."
        )

    # --- VAE (AutoencoderKLQwenImage) ---
    if vae_component and vae_component.file_format == "safetensors":
        LOGGER.debug("Loading Krea VAE component from %s", vae_component.path)
        vae = AutoencoderKLQwenImage.from_single_file(
            str(vae_component.path),
            config=str(pack.pipeline_config_dir / "vae"),
            dtype=dtype,
            local_files_only=True,
        )
        kwargs["vae"] = vae
    elif vae_component:
        raise ValueError(
            f"Unsupported VAE format '{vae_component.file_format}' for Krea pack."
        )

    # --- Text encoder (Qwen3VLModel) ---
    # Reuses the shared Qwen loader; Qwen3VL is a vision-language model but loads through the same
    # transformers Auto* path. The image-conditioning capability is exercised at generate/img2img
    # time (WP-5), not at load time.
    if text_encoder_component and text_encoder_component.file_format == "gguf":
        kwargs["text_encoder"] = _load_text_encoder_from_local_config(
            component_path=text_encoder_component.path,
            config_dir=pack.pipeline_config_dir / "text_encoder",
            dtype=dtype,
            local_files_only=True,
            gguf_file=text_encoder_component.path.name,
        )
    elif text_encoder_component and text_encoder_component.file_format == "safetensors":
        kwargs["text_encoder"] = _load_text_encoder_from_local_config(
            component_path=text_encoder_component.path,
            config_dir=pack.pipeline_config_dir / "text_encoder",
            dtype=dtype,
            local_files_only=True,
        )

    LOGGER.debug("Building Krea2 pipeline from local config: %s", pack.pipeline_config_dir)
    pipeline = Krea2Pipeline.from_pretrained(str(pack.pipeline_config_dir), **kwargs)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)

    if device == "cuda":
        if profile.enable_sequential_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif profile.enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")

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
