from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.profiles import RuntimeProfile
from app.core.model_registry import ModelPack
from app.core.platform_guidance import setup_repair_hint

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedQwenPipeline:
    tokenizer: Any
    text_encoder: Any
    device: str
    dtype_name: str
    backend_name: str = "diffusers_qwen"


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


def _safetensors_floating_dtype_names(checkpoint_path: Path) -> set[str]:
    from safetensors import safe_open

    floating_dtypes: set[str] = set()
    with safe_open(str(checkpoint_path), framework="pt") as handle:
        for key in handle.keys():
            slice_info = handle.get_slice(key)
            dtype_name = str(slice_info.get_dtype()).strip().upper()
            if dtype_name.startswith(("F", "BF")):
                floating_dtypes.add(dtype_name)
    return floating_dtypes


def _checkpoint_contains_fp8_weights(checkpoint_path: Path) -> bool:
    return any(dtype_name.startswith("F8_") for dtype_name in _safetensors_floating_dtype_names(checkpoint_path))


def _tensor_is_fp8(tensor: Any) -> bool:
    dtype_text = str(getattr(tensor, "dtype", "")).lower()
    return "float8" in dtype_text


def _state_dict_contains_fp8(state_dict: dict[str, Any]) -> bool:
    return any(_tensor_is_fp8(tensor) for tensor in state_dict.values())


def _convert_scaled_fp8_text_encoder_state_dict(
    state_dict: dict[str, Any],
    *,
    dtype: Any,
) -> tuple[dict[str, Any], int]:
    converted: dict[str, Any] = {}
    converted_count = 0

    for key, tensor in state_dict.items():
        if key == "scaled_fp8" or key.endswith((".scale_input", ".scale_weight")):
            continue

        if _tensor_is_fp8(tensor):
            scale = state_dict.get(f"{key.rsplit('.', 1)[0]}.scale_weight")
            tensor = tensor.to(dtype=dtype)
            if scale is not None:
                tensor = tensor * scale.to(dtype=dtype)
            converted_count += 1
        elif hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)

        converted[key] = tensor

    return converted, converted_count


def _load_text_encoder_state_dict_if_fp8(
    component_path: Path,
    *,
    dtype: Any,
) -> dict[str, Any] | None:
    from safetensors.torch import load_file

    if not _checkpoint_contains_fp8_weights(component_path):
        return None

    state_dict = load_file(str(component_path), device="cpu")
    if not _state_dict_contains_fp8(state_dict):
        return None

    converted, converted_count = _convert_scaled_fp8_text_encoder_state_dict(
        state_dict,
        dtype=dtype,
    )
    LOGGER.info(
        "Converted %d FP8 text encoder tensors to %s for runtime load.",
        converted_count,
        dtype,
    )
    return converted


def _instantiate_text_encoder_from_config(model_cls: Any, config: Any) -> Any:
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            return model_cls.from_config(config)
    except Exception:
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return model_cls.from_config(config)


def _load_text_encoder_from_state_dict(
    *,
    config_dir: Path,
    state_dict: dict[str, Any],
    dtype: Any,
    local_files_only: bool,
) -> Any:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=local_files_only)
    if hasattr(config, "torch_dtype"):
        config.torch_dtype = dtype

    loaders = (
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("AutoModel", AutoModel),
    )
    last_error: Exception | None = None
    for loader_name, model_cls in loaders:
        model = None
        try:
            LOGGER.debug("Loading text encoder with %s from converted state dict.", loader_name)
            model = _instantiate_text_encoder_from_config(model_cls, config)
            incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
            if hasattr(model, "tie_weights"):
                model.tie_weights()

            missing = list(getattr(incompatible, "missing_keys", ()) or ())
            unexpected = list(getattr(incompatible, "unexpected_keys", ()) or ())
            total = len(model.state_dict())
            missing_ratio = (len(missing) / total) if total else 0.0
            unexpected_ratio = (len(unexpected) / total) if total else 0.0
            if missing_ratio > 0.10:
                raise ValueError(
                    f"{loader_name} rejected due to high missing ratio "
                    f"({len(missing)}/{total}, {missing_ratio:.1%})."
                )
            if unexpected_ratio > 0.10:
                raise ValueError(
                    f"{loader_name} rejected due to high unexpected ratio "
                    f"({len(unexpected)}/{total}, {unexpected_ratio:.1%})."
                )
            return model.to(dtype=dtype)
        except Exception as exc:
            last_error = exc
            if model is not None:
                del model

    raise ValueError(
        "Unable to load text encoder from converted state dict at "
        f"'{config_dir}'. Last error: {last_error}"
    )


def _load_text_encoder_from_local_config(
    *,
    component_path: Path,
    config_dir: Path,
    dtype: Any,
    local_files_only: bool,
    gguf_file: str | None = None,
) -> Any:
    from transformers import AutoModel, AutoModelForCausalLM

    staged_name = gguf_file or component_path.name
    staged_path = config_dir / staged_name
    _stage_weight(component_path, staged_path)

    common_kwargs = {
        "local_files_only": local_files_only,
        "dtype": dtype,
        "output_loading_info": True,
    }
    if gguf_file:
        common_kwargs["gguf_file"] = staged_name

    if gguf_file is None:
        state_dict = _load_text_encoder_state_dict_if_fp8(component_path, dtype=dtype)
        if state_dict is not None:
            return _load_text_encoder_from_state_dict(
                config_dir=config_dir,
                state_dict=state_dict,
                dtype=dtype,
                local_files_only=local_files_only,
            )

    loaders = (
        ("AutoModelForCausalLM", AutoModelForCausalLM.from_pretrained),
        ("AutoModel", AutoModel.from_pretrained),
    )
    last_error: Exception | None = None
    for loader_name, loader in loaders:
        model = None
        try:
            LOGGER.debug(
                "Loading text encoder with %s from %s (file=%s)",
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
                LOGGER.debug(
                    "Accepted text encoder loader %s (missing=%d/%d).",
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
        "Unable to load text encoder from local config path "
        f"'{config_dir}' and file '{staged_name}'. Last error: {last_error}"
    )



def build_qwen_pipeline(pack: ModelPack, profile: RuntimeProfile) -> LoadedQwenPipeline:
    if pack.architecture != "z_image_turbo":
        raise ValueError(f"Unsupported architecture '{pack.architecture}' for Qwen pipeline builder.")
    if pack.pipeline_config_dir is None:
        raise ValueError("Model pack is missing 'pipeline_config_dir'. A local diffusers config directory is required.")

    import torch
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Installed transformers build is missing AutoTokenizer. " + setup_repair_hint()) from exc

    text_encoder_component = _pick_component(pack, "text_encoder", "encoder")
    if text_encoder_component is None:
        raise ValueError("Model pack is missing a text_encoder component.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(torch, device)
    tokenizer_dir = pack.pipeline_config_dir / "tokenizer"
    tokenizer_source = tokenizer_dir if tokenizer_dir.exists() else pack.pipeline_config_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=True)

    if text_encoder_component.file_format == "gguf":
        text_encoder = _load_text_encoder_from_local_config(
            component_path=text_encoder_component.path,
            config_dir=pack.pipeline_config_dir / "text_encoder",
            dtype=dtype,
            local_files_only=True,
            gguf_file=text_encoder_component.path.name,
        )
    elif text_encoder_component.file_format == "safetensors":
        text_encoder = _load_text_encoder_from_local_config(
            component_path=text_encoder_component.path,
            config_dir=pack.pipeline_config_dir / "text_encoder",
            dtype=dtype,
            local_files_only=True,
        )
    else:
        raise ValueError(f"Unsupported text encoder format: {text_encoder_component.file_format}")

    if device == "cuda" and hasattr(text_encoder, "to"):
        text_encoder = text_encoder.to("cuda")

    return LoadedQwenPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        dtype_name=str(dtype),
    )


__all__ = [
    "LoadedQwenPipeline",
    "build_qwen_pipeline",
    "_convert_scaled_fp8_text_encoder_state_dict",
    "_load_text_encoder_from_local_config",
]
