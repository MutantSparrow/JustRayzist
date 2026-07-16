from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# NOTE: ``OptimizationsConfig`` and its subtypes live in ``app.core.pipeline_factory.optimizations``
# but are imported lazily inside ``_parse_optimizations`` / at the type-annotation boundary below
# to avoid a circular import cycle (pipeline_factory.__init__ pulls in zimage.py, which in turn
# needs ``ModelPack``).
ALLOWED_ARCHITECTURES = {"z_image_turbo", "krea2_turbo"}
ALLOWED_FORMATS = {"safetensors", "gguf"}
ALLOWED_STORAGE_MODES = {"layerwise"}
ALLOWED_RUNTIME_DTYPES = {"float32", "float16", "bfloat16", "fp8_e4m3fn"}
ALLOWED_TIER_NAMES = {"high", "balanced", "constrained"}
ALLOWED_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune"}
ALLOWED_FP8_SCOPES = {"transformer", "transformer+text_encoder"}


@dataclass(frozen=True)
class ModelComponent:
    role: str
    path: Path
    file_format: str
    storage_dtype: str | None = None
    compute_dtype: str | None = None
    storage_mode: str | None = None


@dataclass(frozen=True)
class ModelPack:
    name: str
    architecture: str
    backend_preference: list[str]
    components: dict[str, ModelComponent]
    pipeline_config_dir: Path | None
    required_configs: list[Path]
    source_file: Path
    user_visible: bool = True
    enabled: bool = True
    base_name: str | None = None
    derived_strategy: str | None = None
    # Optional per-pack minimum free-VRAM (GB) required to select each resource tier. Overrides
    # RUNTIME_PROFILES[tier].min_free_vram_gb when the auto-tier selector runs against this pack.
    # Absent keys fall back to global defaults. Keys must be a subset of ALLOWED_TIER_NAMES.
    resource_tier_thresholds: dict[str, int] | None = None
    # Optional post-load runtime optimizations (torch.compile / fp8 quant / SageAttention). Each is
    # capability-gated at apply time (see app/core/pipeline_factory/optimizations.py) so a pack
    # requesting an option that the local GPU cannot support degrades gracefully with a log line.
    # Typed as ``Any`` here to avoid an import cycle at module load; runtime value is always an
    # ``OptimizationsConfig`` instance produced by ``_parse_optimizations``.
    optimizations: Any = field(default_factory=lambda: _default_optimizations_config())


class ModelPackValidationError(ValueError):
    pass


def _parse_user_visible(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    raise ModelPackValidationError("'user_visible' must be a boolean when provided.")


def _parse_enabled(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    raise ModelPackValidationError("'enabled' must be a boolean when provided.")


def _parse_bool(field: str, raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise ModelPackValidationError(f"'{field}' must be a boolean; got {raw!r}.")


def _default_optimizations_config() -> Any:
    from app.core.pipeline_factory.optimizations import OptimizationsConfig

    return OptimizationsConfig()


def _parse_optimizations(raw: Any) -> Any:
    """Parse the top-level ``optimizations`` block of a modelpack.yaml.

    Shape (all fields optional; missing keys default to disabled):

    ```yaml
    optimizations:
      torch_compile: {enabled: true, mode: reduce-overhead, fullgraph: false}
      fp8_quantization: {enabled: true, scope: transformer}
      sage_attention: {enabled: true}
    ```

    Boolean shortcut also accepted: ``torch_compile: true`` is equivalent to
    ``torch_compile: {enabled: true}`` (defaults for other subfields).
    """
    from app.core.pipeline_factory.optimizations import (
        Fp8QuantConfig,
        OptimizationsConfig,
        SageAttentionConfig,
        TF32Config,
        TorchCompileConfig,
        VAETilingConfig,
    )

    if raw is None:
        return OptimizationsConfig()
    if not isinstance(raw, dict):
        raise ModelPackValidationError(
            "'optimizations' must be an object mapping optimization names to config blocks."
        )

    def _sub(name: str) -> dict[str, Any]:
        value = raw.get(name)
        if value is None:
            return {}
        if isinstance(value, bool):
            return {"enabled": value}
        if isinstance(value, dict):
            return value
        raise ModelPackValidationError(
            f"'optimizations.{name}' must be a boolean or an object; got {value!r}."
        )

    compile_raw = _sub("torch_compile")
    compile_mode = str(compile_raw.get("mode", "reduce-overhead")).strip().lower()
    if compile_mode not in ALLOWED_COMPILE_MODES:
        allowed = ", ".join(sorted(ALLOWED_COMPILE_MODES))
        raise ModelPackValidationError(
            f"'optimizations.torch_compile.mode' must be one of: {allowed}. Got {compile_mode!r}."
        )

    fp8_raw = _sub("fp8_quantization")
    fp8_scope = str(fp8_raw.get("scope", "transformer")).strip().lower()
    if fp8_scope not in ALLOWED_FP8_SCOPES:
        allowed = ", ".join(sorted(ALLOWED_FP8_SCOPES))
        raise ModelPackValidationError(
            f"'optimizations.fp8_quantization.scope' must be one of: {allowed}. Got {fp8_scope!r}."
        )

    sage_raw = _sub("sage_attention")
    tf32_raw = _sub("tf32")
    vae_tiling_raw = _sub("vae_tiling")

    allowed_keys = {"torch_compile", "fp8_quantization", "sage_attention", "tf32", "vae_tiling"}
    unknown_keys = set(raw.keys()) - allowed_keys
    if unknown_keys:
        allowed = ", ".join(sorted(allowed_keys))
        raise ModelPackValidationError(
            f"Unknown optimization keys: {sorted(unknown_keys)}. Allowed: {allowed}."
        )

    return OptimizationsConfig(
        torch_compile=TorchCompileConfig(
            enabled=_parse_bool("optimizations.torch_compile.enabled", compile_raw.get("enabled"), False),
            mode=compile_mode,
            fullgraph=_parse_bool(
                "optimizations.torch_compile.fullgraph", compile_raw.get("fullgraph"), False
            ),
        ),
        fp8_quantization=Fp8QuantConfig(
            enabled=_parse_bool("optimizations.fp8_quantization.enabled", fp8_raw.get("enabled"), False),
            scope=fp8_scope,
        ),
        sage_attention=SageAttentionConfig(
            enabled=_parse_bool("optimizations.sage_attention.enabled", sage_raw.get("enabled"), False),
        ),
        tf32=TF32Config(
            enabled=_parse_bool("optimizations.tf32.enabled", tf32_raw.get("enabled"), False),
        ),
        vae_tiling=VAETilingConfig(
            enabled=_parse_bool("optimizations.vae_tiling.enabled", vae_tiling_raw.get("enabled"), False),
        ),
    )


def _parse_resource_tier_thresholds(raw: Any) -> dict[str, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ModelPackValidationError(
            "'resource_tier_thresholds' must be an object mapping tier names to GB integers."
        )
    parsed: dict[str, int] = {}
    for key, value in raw.items():
        name = str(key).strip().lower()
        if name not in ALLOWED_TIER_NAMES:
            allowed = ", ".join(sorted(ALLOWED_TIER_NAMES))
            raise ModelPackValidationError(
                f"Unknown resource tier '{key}' in 'resource_tier_thresholds'. Allowed: {allowed}"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelPackValidationError(
                f"'resource_tier_thresholds.{name}' must be an integer (GB); got {value!r}."
            )
        if value < 0:
            raise ModelPackValidationError(
                f"'resource_tier_thresholds.{name}' must be >= 0; got {value}."
            )
        parsed[name] = value
    return parsed or None


def _is_remote_path(raw_path: str) -> bool:
    lowered = raw_path.lower()
    return lowered.startswith(("http://", "https://", "hf://", "s3://"))


def _resolve_local_path(base_dir: Path, raw_path: str) -> Path:
    if _is_remote_path(raw_path):
        raise ModelPackValidationError(f"Remote path is not allowed: '{raw_path}'")
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _validate_extension(path: Path, declared_format: str, field_name: str) -> None:
    suffix = path.suffix.lower().lstrip(".")
    if declared_format not in ALLOWED_FORMATS:
        allowed = ", ".join(sorted(ALLOWED_FORMATS))
        raise ModelPackValidationError(
            f"Unsupported format '{declared_format}' for component '{field_name}'. "
            f"Allowed: {allowed}"
        )
    if suffix != declared_format:
        raise ModelPackValidationError(
            f"Format mismatch for component '{field_name}': "
            f"path extension '.{suffix}' does not match declared format '{declared_format}'."
        )


def _require_file(path: Path, field_name: str) -> None:
    if not path.exists():
        raise ModelPackValidationError(f"Missing file for '{field_name}': {path}")
    if not path.is_file():
        raise ModelPackValidationError(f"Expected a file for '{field_name}': {path}")


def _require_existing_path(path: Path, field_name: str) -> None:
    if not path.exists():
        raise ModelPackValidationError(f"Missing path for '{field_name}': {path}")


def _parse_runtime_dtype(role: str, field_name: str, raw: Any) -> str | None:
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if not normalized:
        return None
    if normalized not in ALLOWED_RUNTIME_DTYPES:
        allowed = ", ".join(sorted(ALLOWED_RUNTIME_DTYPES))
        raise ModelPackValidationError(
            f"Unsupported {field_name!r} value '{raw}' for component '{role}'. "
            f"Allowed: {allowed}"
        )
    return normalized


def _parse_storage_mode(role: str, raw: Any) -> str | None:
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if not normalized:
        return None
    if normalized not in ALLOWED_STORAGE_MODES:
        allowed = ", ".join(sorted(ALLOWED_STORAGE_MODES))
        raise ModelPackValidationError(
            f"Unsupported 'storage_mode' value '{raw}' for component '{role}'. "
            f"Allowed: {allowed}"
        )
    return normalized


def _parse_component(base_dir: Path, role: str, raw: dict[str, Any]) -> ModelComponent:
    if not isinstance(raw, dict):
        raise ModelPackValidationError(f"Component '{role}' must be an object.")
    raw_path = raw.get("path")
    file_format = raw.get("format")
    if not raw_path or not file_format:
        raise ModelPackValidationError(
            f"Component '{role}' requires both 'path' and 'format'."
        )
    resolved = _resolve_local_path(base_dir, str(raw_path))
    declared_format = str(file_format).lower()
    _validate_extension(resolved, declared_format, role)
    _require_file(resolved, role)
    storage_dtype = _parse_runtime_dtype(role, "storage_dtype", raw.get("storage_dtype"))
    compute_dtype = _parse_runtime_dtype(role, "compute_dtype", raw.get("compute_dtype"))
    storage_mode = _parse_storage_mode(role, raw.get("storage_mode"))
    if storage_mode and storage_dtype is None:
        raise ModelPackValidationError(
            f"Component '{role}' uses storage_mode '{storage_mode}' but is missing 'storage_dtype'."
        )
    return ModelComponent(
        role=role,
        path=resolved,
        file_format=declared_format,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
        storage_mode=storage_mode,
    )


def load_model_pack(pack_file: Path) -> ModelPack:
    if not pack_file.exists():
        raise ModelPackValidationError(f"Model pack file does not exist: {pack_file}")

    with pack_file.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ModelPackValidationError("Model pack root must be an object.")

    base_dir = pack_file.parent
    name = str(payload.get("name") or pack_file.parent.name)
    architecture = str(payload.get("architecture", "")).strip().lower()
    if architecture not in ALLOWED_ARCHITECTURES:
        allowed = ", ".join(sorted(ALLOWED_ARCHITECTURES))
        raise ModelPackValidationError(
            f"Unsupported architecture '{architecture}'. Allowed: {allowed}"
        )

    backend_raw = payload.get("backend_preference", ["diffusers"])
    if isinstance(backend_raw, str):
        backend_preference = [backend_raw]
    elif isinstance(backend_raw, list):
        backend_preference = [str(item) for item in backend_raw if str(item).strip()]
    else:
        raise ModelPackValidationError("'backend_preference' must be a string or list.")
    if not backend_preference:
        raise ModelPackValidationError("'backend_preference' cannot be empty.")

    raw_components = payload.get("components")
    if not isinstance(raw_components, dict) or not raw_components:
        raise ModelPackValidationError("'components' must be a non-empty object.")

    components: dict[str, ModelComponent] = {}
    for role, component_data in raw_components.items():
        role_name = str(role)
        components[role_name] = _parse_component(base_dir, role_name, component_data)

    raw_configs = payload.get("required_configs", [])
    if not isinstance(raw_configs, list):
        raise ModelPackValidationError("'required_configs' must be a list.")
    required_configs = [_resolve_local_path(base_dir, str(item)) for item in raw_configs]
    for config_path in required_configs:
        _require_existing_path(config_path, "required_configs")

    raw_pipeline_config_dir = payload.get("pipeline_config_dir")
    pipeline_config_dir: Path | None = None
    if raw_pipeline_config_dir:
        pipeline_config_dir = _resolve_local_path(base_dir, str(raw_pipeline_config_dir))
        _require_existing_path(pipeline_config_dir, "pipeline_config_dir")
        if not pipeline_config_dir.is_dir():
            raise ModelPackValidationError(
                f"'pipeline_config_dir' must point to a directory: {pipeline_config_dir}"
            )

    user_visible = _parse_user_visible(payload.get("user_visible"))
    enabled = _parse_enabled(payload.get("enabled"))
    resource_tier_thresholds = _parse_resource_tier_thresholds(
        payload.get("resource_tier_thresholds")
    )
    optimizations = _parse_optimizations(payload.get("optimizations"))

    return ModelPack(
        name=name,
        architecture=architecture,
        backend_preference=backend_preference,
        components=components,
        pipeline_config_dir=pipeline_config_dir,
        required_configs=required_configs,
        source_file=pack_file.resolve(),
        user_visible=user_visible,
        enabled=enabled,
        base_name=name,
        derived_strategy=None,
        resource_tier_thresholds=resource_tier_thresholds,
        optimizations=optimizations,
    )


def discover_model_packs(model_packs_dir: Path) -> list[Path]:
    if not model_packs_dir.exists():
        return []
    found = []
    for candidate in sorted(model_packs_dir.glob("*/modelpack.yaml")):
        if candidate.is_file():
            found.append(candidate.resolve())
    return found


def load_model_pack_by_name(model_packs_dir: Path, pack_name: str) -> ModelPack:
    normalized = pack_name.strip().lower()
    if not normalized:
        raise ModelPackValidationError("Pack name cannot be empty.")

    for pack_file in discover_model_packs(model_packs_dir):
        candidate = load_model_pack(pack_file)
        if candidate.name.lower() == normalized or pack_file.parent.name.lower() == normalized:
            return candidate
    raise ModelPackValidationError(
        f"Model pack '{pack_name}' was not found under {model_packs_dir}."
    )
