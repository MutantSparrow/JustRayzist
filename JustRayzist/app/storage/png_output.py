from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.config.settings import AppSettings


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
) -> Path:
    path = output_path or build_output_path(settings.paths.outputs_dir)
    metadata = PngInfo()
    metadata.add_text("timestamp", _utc_timestamp())
    metadata.add_text("prompt", prompt)
    metadata.add_text("application_name", settings.app_name)
    metadata.add_text("application_version", settings.app_version)
    metadata.add_text("generated_with", "Just Rayzist!")
    metadata.add_text("model_page", "https://huggingface.co/MutantSparrow/Ray")
    if extra_metadata:
        for key, value in extra_metadata.items():
            metadata.add_text(str(key), str(value))
    image.save(path, format="PNG", pnginfo=metadata)
    return path
