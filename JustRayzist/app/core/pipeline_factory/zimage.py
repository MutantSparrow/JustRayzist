from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.profiles import RuntimeProfile
from app.core.model_registry import ModelPack

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedZImagePipeline:
    pipeline: Any
    device: str
    dtype_name: str


def _resolve_dtype(torch_module: Any, device: str) -> Any:
    if device == "cuda":
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def _pick_component(pack: ModelPack, *roles: str):
    for role in roles:
        component = pack.components.get(role)
        if component is not None:
            return component
    return None


def _stage_weight(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _load_text_encoder_from_gguf(
    *,
    component_path: Path,
    config_dir: Path,
    dtype: Any,
    local_files_only: bool,
) -> Any:
    from transformers import AutoModel, AutoModelForCausalLM

    staged_name = component_path.name
    staged_path = config_dir / staged_name
    _stage_weight(component_path, staged_path)

    common_kwargs = {
        "local_files_only": local_files_only,
        "torch_dtype": dtype,
        "gguf_file": staged_name,
        "output_loading_info": True,
    }

    loaders = (
        ("AutoModelForCausalLM", AutoModelForCausalLM.from_pretrained),
        ("AutoModel", AutoModel.from_pretrained),
    )
    last_error: Exception | None = None
    for loader_name, loader in loaders:
        model = None
        try:
            LOGGER.info(
                "Loading text encoder GGUF with %s from %s (file=%s)",
                loader_name,
                config_dir,
                staged_name,
            )
            loaded = loader(str(config_dir), **common_kwargs)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                model, loading_info = loaded
                missing = loading_info.get("missing_keys") or []
                total = len(model.state_dict())
                missing_ratio = (len(missing) / total) if total else 0.0
                if missing_ratio > 0.10:
                    raise ValueError(
                        f"{loader_name} rejected due to high missing ratio "
                        f"({len(missing)}/{total}, {missing_ratio:.1%})."
                    )
                LOGGER.info(
                    "Accepted text encoder GGUF loader %s (missing=%d/%d).",
                    loader_name,
                    len(missing),
                    total,
                )
                return model
            return loaded
        except Exception as exc:  # pragma: no cover - runtime fallback branch
            last_error = exc
            if model is not None:
                del model

    raise ValueError(
        "Unable to load GGUF text encoder from local config path "
        f"'{config_dir}' and file '{staged_name}'. Last error: {last_error}"
    )


def _is_prefixed_fused_zimage_transformer(checkpoint_path: Path) -> bool:
    from safetensors import safe_open

    prefix_hits = 0
    qkv_hits = 0
    with safe_open(str(checkpoint_path), framework="pt") as handle:
        for idx, key in enumerate(handle.keys()):
            if key.startswith("model.diffusion_model."):
                prefix_hits += 1
            if "attention.qkv.weight" in key:
                qkv_hits += 1
            if idx > 4096:
                break
    return prefix_hits > 0 and qkv_hits > 0


def _load_prefixed_fused_zimage_transformer(
    checkpoint_path: Path,
    config_dir: Path,
    zimage_transformer_cls: Any,
) -> Any:
    from safetensors.torch import load_file

    raw_state = load_file(str(checkpoint_path))
    prefix = "model.diffusion_model."
    converted: dict[str, Any] = {}

    for key, tensor in raw_state.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        if name.startswith("vae."):
            continue
        if name == "norm_final.weight":
            continue
        if name.startswith("x_embedder."):
            name = name.replace("x_embedder.", "all_x_embedder.2-1.", 1)
        if name.startswith("final_layer."):
            name = name.replace("final_layer.", "all_final_layer.2-1.", 1)
        if ".attention.k_norm.weight" in name:
            name = name.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
        if ".attention.q_norm.weight" in name:
            name = name.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        if name.endswith(".attention.qkv.weight"):
            base = name[: -len(".attention.qkv.weight")]
            if tensor.shape[0] % 3 != 0:
                raise ValueError(f"Unexpected qkv tensor shape for key '{key}': {tuple(tensor.shape)}")
            q_weight, k_weight, v_weight = tensor.chunk(3, dim=0)
            converted[f"{base}.attention.to_q.weight"] = q_weight
            converted[f"{base}.attention.to_k.weight"] = k_weight
            converted[f"{base}.attention.to_v.weight"] = v_weight
            continue
        if name.endswith(".attention.out.weight"):
            mapped = name.replace(".attention.out.weight", ".attention.to_out.0.weight")
            converted[mapped] = tensor
            continue
        converted[name] = tensor

    config = zimage_transformer_cls.load_config(str(config_dir))
    model = zimage_transformer_cls.from_config(config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected transformer keys after conversion: {unexpected[:8]}")
    if missing:
        raise ValueError(f"Missing transformer keys after conversion: {missing[:8]}")
    return model


def build_zimage_pipeline(pack: ModelPack, profile: RuntimeProfile) -> LoadedZImagePipeline:
    if pack.architecture != "z_image_turbo":
        raise ValueError(
            f"Unsupported architecture '{pack.architecture}' for Z-Image pipeline builder."
        )
    if pack.pipeline_config_dir is None:
        raise ValueError(
            "Model pack is missing 'pipeline_config_dir'. "
            "A local diffusers config directory is required."
        )

    import torch
    from diffusers import (
        AutoencoderKL,
        GGUFQuantizationConfig,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(torch, device)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }

    transformer_component = _pick_component(pack, "transformer", "checkpoint")
    vae_component = _pick_component(pack, "vae")
    text_encoder_component = _pick_component(pack, "text_encoder", "encoder")
    if text_encoder_component and text_encoder_component.file_format == "safetensors":
        _stage_weight(
            text_encoder_component.path,
            pack.pipeline_config_dir / "text_encoder" / "model.safetensors",
        )

    if transformer_component and transformer_component.file_format == "gguf":
        LOGGER.info(
            "Loading transformer component from %s (%s)",
            transformer_component.path,
            transformer_component.file_format,
        )
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        transformer = ZImageTransformer2DModel.from_single_file(
            str(transformer_component.path),
            quantization_config=quantization_config,
            config=str(pack.pipeline_config_dir / "transformer"),
            torch_dtype=dtype,
            local_files_only=True,
        )
        kwargs["transformer"] = transformer
    elif transformer_component and transformer_component.file_format == "safetensors":
        LOGGER.info("Loading transformer component from %s", transformer_component.path)
        if _is_prefixed_fused_zimage_transformer(transformer_component.path):
            LOGGER.info("Detected fused/prefixed Z-Image checkpoint format; applying key conversion.")
            transformer = _load_prefixed_fused_zimage_transformer(
                checkpoint_path=transformer_component.path,
                config_dir=pack.pipeline_config_dir / "transformer",
                zimage_transformer_cls=ZImageTransformer2DModel,
            )
        else:
            transformer = ZImageTransformer2DModel.from_single_file(
                str(transformer_component.path),
                config=str(pack.pipeline_config_dir / "transformer"),
                torch_dtype=dtype,
                local_files_only=True,
            )
        transformer = transformer.to(dtype=dtype)
        kwargs["transformer"] = transformer

    if vae_component and vae_component.file_format == "gguf":
        LOGGER.info("Loading VAE component from %s (%s)", vae_component.path, vae_component.file_format)
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        try:
            vae = AutoencoderKL.from_single_file(
                str(vae_component.path),
                quantization_config=quantization_config,
                config=str(pack.pipeline_config_dir / "vae"),
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as exc:
            raise ValueError(f"Failed to load GGUF VAE component: {exc}") from exc
        kwargs["vae"] = vae
    elif vae_component and vae_component.file_format == "safetensors":
        LOGGER.info("Loading VAE component from %s", vae_component.path)
        vae = AutoencoderKL.from_single_file(
            str(vae_component.path),
            config=str(pack.pipeline_config_dir / "vae"),
            torch_dtype=dtype,
            local_files_only=True,
        )
        kwargs["vae"] = vae

    if text_encoder_component and text_encoder_component.file_format == "gguf":
        kwargs["text_encoder"] = _load_text_encoder_from_gguf(
            component_path=text_encoder_component.path,
            config_dir=pack.pipeline_config_dir / "text_encoder",
            dtype=dtype,
            local_files_only=True,
        )

    LOGGER.info("Building Z-Image pipeline from local config: %s", pack.pipeline_config_dir)
    pipeline = ZImagePipeline.from_pretrained(str(pack.pipeline_config_dir), **kwargs)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)

    if device == "cuda":
        if profile.enable_sequential_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif profile.enable_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")

    return LoadedZImagePipeline(pipeline=pipeline, device=device, dtype_name=str(dtype))
