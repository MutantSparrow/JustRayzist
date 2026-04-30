from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.config.settings import AppSettings

_GALLERY_EXTRA_KEYS: frozenset[str] = frozenset({
    "prompt_wildcard_resolved", "width", "height", "model_pack",
    "backend", "device", "steps", "guidance_scale", "duration_ms",
    "inference_process", "procedural_creativity", "mode", "source_image",
    "source_filename", "source_width", "source_height", "similarity",
    "wildcards_json", "wildcard_count", "loras_json", "lora_count",
    "upscale_auto_content_mode",
})

_RUNTIME_SUMMARY_KEYS: frozenset[str] = frozenset({
    "owner_id", "prompt_effective", "seed", "scheduler_mode",
    "runtime_profile", "resource_tier", "execution_mode", "effective_pack",
})

# Extra-metadata keys kept per mode when meta_debug=False.
# Keys absent from the set are dropped before writing to PNG.
_FINAL_EXTRA_KEYS: dict[str, frozenset[str]] = {
    "generate": _GALLERY_EXTRA_KEYS | _RUNTIME_SUMMARY_KEYS | frozenset({
        "selected_pack", "derived_strategy",
    }),
    "img2img": _GALLERY_EXTRA_KEYS | _RUNTIME_SUMMARY_KEYS | frozenset({
        "selected_pack", "derived_strategy", "refine_strength",
        "source_original_width", "source_original_height",
    }),
    "upscale": _GALLERY_EXTRA_KEYS | _RUNTIME_SUMMARY_KEYS | frozenset({
        "upscale_engine", "working_width", "working_height",
        "upscale_duration_ms",
    }),
    "clarity": _GALLERY_EXTRA_KEYS | _RUNTIME_SUMMARY_KEYS | frozenset({
        "clarity_engine", "clarity_variant", "working_width",
        "working_height", "clarity_duration_ms", "clarity_fs_method",
        "clarity_fs_type", "clarity_fs_intensity", "clarity_unsharp_stage",
        "clarity_unsharp_radius", "clarity_unsharp_percent",
        "clarity_unsharp_threshold",
    }),
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_output_path(output_dir: Path, prefix: str = "justrayzist") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        suffix = f"{index:03d}"
        candidate = output_dir / f"{prefix}_{timestamp}_{suffix}.png"
        if not candidate.exists():
            return candidate
        index += 1


def save_png_with_metadata(
    image: Image.Image,
    prompt: str,
    settings: AppSettings,
    output_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    meta_mode: str | None = None,
) -> Path:
    path = output_path or build_output_path(settings.paths.outputs_dir)

    filter_active = not settings.meta_debug and meta_mode is not None
    allowed_keys = _FINAL_EXTRA_KEYS.get(meta_mode, frozenset()) if filter_active else None

    metadata = PngInfo()
    metadata.add_text("timestamp", _utc_timestamp())
    metadata.add_text("prompt", prompt)
    metadata.add_text("application_name", settings.app_name)
    metadata.add_text("application_version", settings.app_version)
    metadata.add_text("generated_with", "Just Rayzist!")
    metadata.add_text("model_page", "https://huggingface.co/MutantSparrow/Ray")
    if extra_metadata:
        for key, value in extra_metadata.items():
            if allowed_keys is None or key in allowed_keys:
                metadata.add_text(str(key), str(value))
    image.save(path, format="PNG", pnginfo=metadata)
    return path
