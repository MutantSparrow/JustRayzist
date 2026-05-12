from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.profiles import RuntimeProfile
from app.core.model_registry import ModelComponent, ModelPack
from app.core.platform_guidance import setup_repair_hint
from app.core.pipeline_factory.qwen import (
    _convert_scaled_fp8_text_encoder_state_dict,
    _load_text_encoder_from_local_config,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedZImagePipeline:
    pipeline: Any
    device: str
    dtype_name: str
    backend_name: str = "diffusers_zimage"
    fp8_checkpoint: bool = False
    fp8_fallback_used: bool = False
    fp8_fallback_reason: str | None = None
    fp8_runtime_mode: str | None = None
    fp8_normalized_tensor_count: int = 0
    fp8_storage_preserved_tensor_count: int = 0
    fp8_promoted_tensor_count: int = 0
    fp8_normalized_tensor_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fp8RuntimePreparation:
    state_dict: dict[str, Any]
    promoted_names: tuple[str, ...]
    normalized_tensor_count: int
    storage_preserved_tensor_count: int
    promoted_tensor_count: int


class _SelectiveTensorStorageHook:
    def __init__(
        self,
        *,
        storage_dtype: Any,
        compute_dtype: Any,
        parameter_names: tuple[str, ...],
        buffer_names: tuple[str, ...],
    ) -> None:
        self._storage_dtype = storage_dtype
        self._compute_dtype = compute_dtype
        self._parameter_names = parameter_names
        self._buffer_names = buffer_names
        self._pre_handle: Any | None = None
        self._post_handle: Any | None = None

    def register(self, module: Any) -> None:
        self._cast(module, self._storage_dtype)
        self._pre_handle = module.register_forward_pre_hook(self._pre_forward, with_kwargs=True)
        self._post_handle = module.register_forward_hook(self._post_forward, with_kwargs=True)

    def _cast(self, module: Any, dtype: Any) -> None:
        for name in self._parameter_names:
            parameter = module._parameters.get(name)
            if parameter is None:
                continue
            parameter.data = parameter.data.to(dtype=dtype)
        for name in self._buffer_names:
            buffer = module._buffers.get(name)
            if buffer is None:
                continue
            module._buffers[name] = buffer.to(dtype=dtype)

    def _cast_inputs(self, value: Any) -> Any:
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            return value.to(dtype=self._compute_dtype)
        if isinstance(value, tuple):
            return tuple(self._cast_inputs(item) for item in value)
        if isinstance(value, list):
            return [self._cast_inputs(item) for item in value]
        if isinstance(value, dict):
            return {key: self._cast_inputs(item) for key, item in value.items()}
        return value

    def _pre_forward(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self._cast(module, self._compute_dtype)
        return self._cast_inputs(args), self._cast_inputs(kwargs)

    def _post_forward(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        self._cast(module, self._storage_dtype)
        return output


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


def _resolve_state_dict_target(root_module: Any, tensor_name: str) -> tuple[Any, str]:
    module = root_module
    parts = tensor_name.split(".")
    for part in parts[:-1]:
        if hasattr(module, "_modules") and part in module._modules:
            module = module._modules[part]
            continue
        module = getattr(module, part)
    return module, parts[-1]


def _load_state_dict_preserving_dtypes(model: Any, state_dict: dict[str, Any]) -> Any:
    import torch

    expected = model.state_dict()
    missing = sorted(set(expected.keys()) - set(state_dict.keys()))
    unexpected = sorted(set(state_dict.keys()) - set(expected.keys()))
    if missing:
        raise ValueError(f"Missing transformer keys after load preparation: {missing[:8]}")
    if unexpected:
        raise ValueError(f"Unexpected transformer keys after load preparation: {unexpected[:8]}")

    for tensor_name, tensor in state_dict.items():
        module, leaf_name = _resolve_state_dict_target(model, tensor_name)
        if leaf_name in getattr(module, "_parameters", {}):
            previous = module._parameters[leaf_name]
            if previous is not None and tuple(previous.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for parameter '{tensor_name}': "
                    f"expected {tuple(previous.shape)}, got {tuple(tensor.shape)}"
                )
            if previous is not None and previous.is_contiguous():
                tensor = tensor.contiguous()
            module._parameters[leaf_name] = torch.nn.Parameter(
                tensor,
                requires_grad=bool(previous.requires_grad) if previous is not None else False,
            )
            continue
        if leaf_name in getattr(module, "_buffers", {}):
            previous = module._buffers[leaf_name]
            if previous is not None and tuple(previous.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for buffer '{tensor_name}': "
                    f"expected {tuple(previous.shape)}, got {tuple(tensor.shape)}"
                )
            if previous is not None and previous.is_contiguous():
                tensor = tensor.contiguous()
            module._buffers[leaf_name] = tensor
            continue
        raise ValueError(f"Unable to resolve state-dict target '{tensor_name}'.")
    return model


def _load_transformer_from_state_dict(
    *,
    state_dict: dict[str, Any],
    config_dir: Path,
    zimage_transformer_cls: Any,
) -> Any:
    config = zimage_transformer_cls.load_config(str(config_dir))
    model = zimage_transformer_cls.from_config(config)
    return _load_state_dict_preserving_dtypes(model, state_dict)


def _convert_prefixed_fused_zimage_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
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
    return converted


def _load_transformer_state_dict(transformer_component: ModelComponent) -> dict[str, Any]:
    from safetensors.torch import load_file

    raw_state = load_file(str(transformer_component.path))
    if _is_prefixed_fused_zimage_transformer(transformer_component.path):
        LOGGER.debug("Detected fused/prefixed Z-Image checkpoint format; applying key conversion.")
        return _convert_prefixed_fused_zimage_state_dict(raw_state)
    return raw_state


def _tensor_requires_compute_dtype(tensor_name: str, tensor: Any) -> bool:
    lowered = tensor_name.lower()
    if getattr(tensor, "ndim", 0) < 2:
        return True
    if lowered.endswith(".bias"):
        return True
    if "norm" in lowered:
        return True
    if "token" in lowered:
        return True
    return False


def _normalize_fp8_state_dict_for_runtime(
    state_dict: dict[str, Any],
    *,
    compute_dtype: Any,
    torch_module: Any,
) -> Fp8RuntimePreparation:
    float8_dtype = getattr(torch_module, "float8_e4m3fn", None)
    normalized: dict[str, Any] = {}
    promoted_names: list[str] = []
    storage_preserved_tensor_count = 0

    for tensor_name, tensor in state_dict.items():
        if not hasattr(tensor, "dtype") or not tensor.is_floating_point():
            normalized[tensor_name] = tensor
            continue
        if _tensor_requires_compute_dtype(tensor_name, tensor):
            if tensor.dtype != compute_dtype:
                normalized[tensor_name] = tensor.to(dtype=compute_dtype)
                promoted_names.append(tensor_name)
                continue
            normalized[tensor_name] = tensor
            continue
        if tensor.dtype == float8_dtype:
            normalized[tensor_name] = tensor
            storage_preserved_tensor_count += 1
            continue
        normalized[tensor_name] = tensor
    return Fp8RuntimePreparation(
        state_dict=normalized,
        promoted_names=tuple(promoted_names),
        normalized_tensor_count=len(promoted_names),
        storage_preserved_tensor_count=storage_preserved_tensor_count,
        promoted_tensor_count=len(promoted_names),
    )


def _model_contains_dtype(model: Any, dtype: Any) -> bool:
    for parameter in model.parameters():
        if getattr(parameter, "dtype", None) == dtype:
            return True
    for buffer in model.buffers():
        if getattr(buffer, "dtype", None) == dtype:
            return True
    return False


def _is_fp8_storage_safe_tensor(tensor_name: str, tensor: Any) -> bool:
    if not hasattr(tensor, "dtype") or not tensor.is_floating_point():
        return False
    if getattr(tensor, "ndim", 0) < 2:
        return False
    return not _tensor_requires_compute_dtype(tensor_name, tensor)


def _apply_fp8_storage_fallback_materialization(
    *,
    model: Any,
    torch_module: Any,
    compute_dtype: Any,
) -> dict[str, Any]:
    storage_dtype = getattr(torch_module, "float8_e4m3fn", None)
    if storage_dtype is None:
        return {
            "fp8_storage_preserved_tensor_count": 0,
            "fp8_storage_preserved_tensor_names": (),
        }

    preserved_tensor_names: list[str] = []
    preserved_tensor_count = 0
    for module_name, module in model.named_modules():
        parameter_names: list[str] = []
        buffer_names: list[str] = []

        for leaf_name, parameter in module._parameters.items():
            if parameter is None:
                continue
            full_name = f"{module_name}.{leaf_name}" if module_name else leaf_name
            if _is_fp8_storage_safe_tensor(full_name, parameter):
                parameter_names.append(leaf_name)
                preserved_tensor_count += 1
                if len(preserved_tensor_names) < 32:
                    preserved_tensor_names.append(full_name)
            elif parameter.is_floating_point() and parameter.dtype != compute_dtype:
                parameter.data = parameter.data.to(dtype=compute_dtype)

        for leaf_name, buffer in module._buffers.items():
            if buffer is None:
                continue
            full_name = f"{module_name}.{leaf_name}" if module_name else leaf_name
            if _is_fp8_storage_safe_tensor(full_name, buffer):
                buffer_names.append(leaf_name)
                preserved_tensor_count += 1
                if len(preserved_tensor_names) < 32:
                    preserved_tensor_names.append(full_name)
            elif buffer.is_floating_point() and buffer.dtype != compute_dtype:
                module._buffers[leaf_name] = buffer.to(dtype=compute_dtype)

        if not parameter_names and not buffer_names:
            continue

        hook = _SelectiveTensorStorageHook(
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            parameter_names=tuple(parameter_names),
            buffer_names=tuple(buffer_names),
        )
        hook.register(module)
        setattr(module, "_justrayzist_fp8_storage_hook", hook)

    return {
        "fp8_storage_preserved_tensor_count": preserved_tensor_count,
        "fp8_storage_preserved_tensor_names": tuple(preserved_tensor_names),
    }


def _resolve_named_dtype(torch_module: Any, raw_name: str | None, fallback: Any) -> Any:
    if raw_name is None:
        return fallback
    attr_name = raw_name.strip().lower()
    if not attr_name:
        return fallback
    if attr_name == "fp8_e4m3fn":
        attr_name = "float8_e4m3fn"
    if not hasattr(torch_module, attr_name):
        raise ValueError(f"Unsupported runtime dtype '{raw_name}'.")
    return getattr(torch_module, attr_name)


def _stage_weight(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _tensor_is_fp8(tensor: Any) -> bool:
    dtype_text = str(getattr(tensor, "dtype", "")).lower()
    return "float8" in dtype_text


def _state_dict_contains_fp8(state_dict: dict[str, Any]) -> bool:
    return any(_tensor_is_fp8(tensor) for tensor in state_dict.values())


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
    converted = _convert_prefixed_fused_zimage_state_dict(raw_state)
    return _load_transformer_from_state_dict(
        state_dict=converted,
        config_dir=config_dir,
        zimage_transformer_cls=zimage_transformer_cls,
    )


def _load_fp8_transformer_for_runtime(
    *,
    transformer_component: ModelComponent,
    config_dir: Path,
    zimage_transformer_cls: Any,
    torch_module: Any,
    compute_dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    state_dict = _load_transformer_state_dict(transformer_component)
    preparation = _normalize_fp8_state_dict_for_runtime(
        state_dict,
        compute_dtype=compute_dtype,
        torch_module=torch_module,
    )
    transformer = _load_transformer_from_state_dict(
        state_dict=preparation.state_dict,
        config_dir=config_dir,
        zimage_transformer_cls=zimage_transformer_cls,
    )

    fp8_fallback_used = True
    fp8_fallback_reason = "native_fp8_not_implemented"
    fp8_runtime_mode = "bf16_with_fp8_storage_fallback"
    storage_metadata = {
        "fp8_storage_preserved_tensor_count": preparation.storage_preserved_tensor_count,
        "fp8_storage_preserved_tensor_names": (),
    }
    storage_metadata = _apply_fp8_storage_fallback_materialization(
        model=transformer,
        torch_module=torch_module,
        compute_dtype=compute_dtype,
    )

    return transformer, {
        "fp8_checkpoint": True,
        "fp8_fallback_used": fp8_fallback_used,
        "fp8_fallback_reason": fp8_fallback_reason,
        "fp8_runtime_mode": fp8_runtime_mode,
        "fp8_normalized_tensor_count": preparation.normalized_tensor_count,
        "fp8_storage_preserved_tensor_count": storage_metadata["fp8_storage_preserved_tensor_count"],
        "fp8_promoted_tensor_count": preparation.promoted_tensor_count,
        "fp8_normalized_tensor_names": preparation.promoted_names[:16],
    }


def _configure_component_storage(
    *,
    model: Any,
    component: ModelComponent | None,
    torch_module: Any,
    device: str,
    default_compute_dtype: Any,
    preserve_existing_storage: bool = False,
) -> Any:
    if component is None:
        return model.to(dtype=default_compute_dtype)

    compute_dtype = _resolve_named_dtype(
        torch_module,
        component.compute_dtype,
        default_compute_dtype,
    )
    if not preserve_existing_storage:
        model = model.to(dtype=compute_dtype)

    if component.storage_mode is None:
        if preserve_existing_storage:
            return model
        return model
    if component.storage_mode != "layerwise":
        raise ValueError(f"Unsupported storage mode '{component.storage_mode}'.")
    if device != "cuda":
        LOGGER.warning(
            "Skipping transformer storage mode '%s' for component '%s' on non-CUDA device '%s'.",
            component.storage_mode,
            component.role,
            device,
        )
        if preserve_existing_storage:
            return model.to(dtype=compute_dtype)
        return model

    storage_dtype = _resolve_named_dtype(
        torch_module,
        component.storage_dtype,
        getattr(torch_module, "float8_e4m3fn"),
    )
    LOGGER.info(
        "Applying layerwise storage mode to component '%s' with storage=%s compute=%s.",
        component.role,
        component.storage_dtype or str(storage_dtype),
        component.compute_dtype or str(compute_dtype),
    )
    if hasattr(model, "enable_layerwise_casting"):
        model.enable_layerwise_casting(
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
        )
        return model

    from diffusers.hooks import apply_layerwise_casting

    apply_layerwise_casting(
        model,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
    )
    return model


def _build_zimage_pipeline(
    pack: ModelPack,
    profile: RuntimeProfile,
    *,
    backend_name: str,
    enable_real_fp8_checkpoint_support: bool,
) -> LoadedZImagePipeline:
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
    try:
        from diffusers import (
            AutoencoderKL,
            GGUFQuantizationConfig,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )
    except ImportError as exc:
        raise ImportError(
            "Installed diffusers build is missing required ZImage classes. "
            + setup_repair_hint()
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(torch, device)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    pipeline_metadata: dict[str, Any] = {
        "backend_name": backend_name,
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

    if transformer_component and transformer_component.file_format == "gguf":
        LOGGER.debug(
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
        LOGGER.debug("Loading transformer component from %s", transformer_component.path)
        transformer_has_fp8_weights = _checkpoint_contains_fp8_weights(transformer_component.path)
        if transformer_has_fp8_weights and enable_real_fp8_checkpoint_support:
            transformer, fp8_metadata = _load_fp8_transformer_for_runtime(
                transformer_component=transformer_component,
                config_dir=pack.pipeline_config_dir / "transformer",
                zimage_transformer_cls=ZImageTransformer2DModel,
                torch_module=torch,
                compute_dtype=dtype,
            )
            pipeline_metadata.update(fp8_metadata)
        elif _is_prefixed_fused_zimage_transformer(transformer_component.path):
            LOGGER.debug("Detected fused/prefixed Z-Image checkpoint format; applying key conversion.")
            transformer = _load_prefixed_fused_zimage_transformer(
                checkpoint_path=transformer_component.path,
                config_dir=pack.pipeline_config_dir / "transformer",
                zimage_transformer_cls=ZImageTransformer2DModel,
            )
        elif transformer_has_fp8_weights:
            from safetensors.torch import load_file

            transformer = _load_transformer_from_state_dict(
                state_dict=load_file(str(transformer_component.path)),
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
        if (
            transformer_has_fp8_weights
            and not enable_real_fp8_checkpoint_support
            and not _model_contains_dtype(
            transformer,
            getattr(torch, "float8_e4m3fn"),
            )
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

    if vae_component and vae_component.file_format == "gguf":
        LOGGER.debug("Loading VAE component from %s (%s)", vae_component.path, vae_component.file_format)
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
        LOGGER.debug("Loading VAE component from %s", vae_component.path)
        vae = AutoencoderKL.from_single_file(
            str(vae_component.path),
            config=str(pack.pipeline_config_dir / "vae"),
            torch_dtype=dtype,
            local_files_only=True,
        )
        kwargs["vae"] = vae

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

    LOGGER.debug("Building Z-Image pipeline from local config: %s", pack.pipeline_config_dir)
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

    return LoadedZImagePipeline(
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


def build_zimage_pipeline(pack: ModelPack, profile: RuntimeProfile) -> LoadedZImagePipeline:
    return _build_zimage_pipeline(
        pack,
        profile,
        backend_name="diffusers_zimage",
        enable_real_fp8_checkpoint_support=False,
    )


def build_fp8_zimage_pipeline(pack: ModelPack, profile: RuntimeProfile) -> LoadedZImagePipeline:
    return _build_zimage_pipeline(
        pack,
        profile,
        backend_name="fp8_zimage",
        enable_real_fp8_checkpoint_support=True,
    )
