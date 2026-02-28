from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_ARCHITECTURES = {"z_image_turbo"}
ALLOWED_FORMATS = {"safetensors", "gguf"}


@dataclass(frozen=True)
class ModelComponent:
    role: str
    path: Path
    file_format: str


@dataclass(frozen=True)
class ModelPack:
    name: str
    architecture: str
    backend_preference: list[str]
    components: dict[str, ModelComponent]
    pipeline_config_dir: Path | None
    required_configs: list[Path]
    source_file: Path


class ModelPackValidationError(ValueError):
    pass


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
    return ModelComponent(role=role, path=resolved, file_format=declared_format)


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

    return ModelPack(
        name=name,
        architecture=architecture,
        backend_preference=backend_preference,
        components=components,
        pipeline_config_dir=pipeline_config_dir,
        required_configs=required_configs,
        source_file=pack_file.resolve(),
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
