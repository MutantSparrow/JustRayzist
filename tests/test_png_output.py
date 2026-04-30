from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image

from app.storage.png_output import save_png_with_metadata


def _read_info(path: Path) -> dict[str, str]:
    with Image.open(path) as image:
        return dict(image.info)


def test_final_metadata_filter_keeps_gallery_fields(make_app_settings, temp_app_paths) -> None:
    settings = replace(make_app_settings(paths=temp_app_paths), meta_debug=False)
    output_path = settings.paths.outputs_dir / "filtered.png"

    save_png_with_metadata(
        image=Image.new("RGB", (8, 8), color=(10, 20, 30)),
        prompt="source prompt",
        settings=settings,
        output_path=output_path,
        meta_mode="upscale",
        extra_metadata={
            "mode": "api_upscale",
            "prompt_effective": "source prompt",
            "width": 16,
            "height": 16,
            "model_pack": "Rayzist_bf16",
            "backend": "baseline_ai_x2_fs",
            "device": "cuda",
            "steps": 0,
            "guidance_scale": 0.0,
            "duration_ms": 123,
            "seed": 42,
            "scheduler_mode": "euler",
            "source_filename": "input.png",
            "source_width": 8,
            "source_height": 8,
            "upscale_auto_content_mode": "art",
            "fp8_normalized_tensor_count": 99,
        },
    )

    metadata = _read_info(output_path)

    assert metadata["prompt"] == "source prompt"
    assert metadata["mode"] == "api_upscale"
    assert metadata["model_pack"] == "Rayzist_bf16"
    assert metadata["source_filename"] == "input.png"
    assert metadata["source_width"] == "8"
    assert metadata["upscale_auto_content_mode"] == "art"
    assert "fp8_normalized_tensor_count" not in metadata


def test_meta_debug_keeps_unfiltered_metadata(make_app_settings, temp_app_paths) -> None:
    settings = replace(make_app_settings(paths=temp_app_paths), meta_debug=True)
    output_path = settings.paths.outputs_dir / "debug.png"

    save_png_with_metadata(
        image=Image.new("RGB", (8, 8), color=(10, 20, 30)),
        prompt="debug prompt",
        settings=settings,
        output_path=output_path,
        meta_mode="generate",
        extra_metadata={"fp8_normalized_tensor_count": 99},
    )

    metadata = _read_info(output_path)

    assert metadata["prompt"] == "debug prompt"
    assert metadata["fp8_normalized_tensor_count"] == "99"
