from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.api.inference_service import InferenceService
from app.core.clarity import CLARITY_ENGINE_NAME, ClarityResult


def test_inference_clarity_persists_lineage(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings=settings)
    saved_metadata: dict[str, object] = {}
    source_path = temp_app_paths.outputs_dir / "example-client" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), color=(12, 34, 56)).save(source_path)

    monkeypatch.setattr(
        "app.api.inference_service.get_image",
        lambda _settings, _filename, owner_id=None: {
            "filename": "source.png",
            "output_path": str(source_path),
            "prompt": "hello world",
            "seed": 123,
            "width": 320,
            "height": 240,
            "model_pack": "Rayzist_bf16",
        },
    )
    monkeypatch.setattr(
        "app.api.inference_service.run_clarity_pipeline",
        lambda **kwargs: ClarityResult(
            image=Image.new("RGB", (320, 240), color=(80, 100, 140)),
            duration_ms=456,
            source_width=320,
            source_height=240,
            working_width=640,
            working_height=480,
            engine_name=CLARITY_ENGINE_NAME,
            variant_label="current",
            device="cpu",
            step_timings_ms={
                "clarity_resize_ms": 1,
                "clarity_fs_ms": 4,
                "clarity_downscale_ms": 7,
                "clarity_pre_downscale_unsharp_ms": 8,
                "clarity_post_downscale_unsharp_ms": 0,
            },
        ),
    )
    monkeypatch.setattr("app.api.inference_service.append_generation_metric", lambda **kwargs: None)

    def fake_save_png_with_metadata(**kwargs):
        saved_metadata["extra_metadata"] = kwargs["extra_metadata"]
        return kwargs["output_path"]

    monkeypatch.setattr("app.api.inference_service.save_png_with_metadata", fake_save_png_with_metadata)
    monkeypatch.setattr(
        "app.api.inference_service.index_image",
        lambda settings, image_path, owner_id=None: {
            "filename": image_path.name,
            "output_path": str(image_path),
            "prompt": "hello world",
            "mode": "api_clarity",
        },
    )

    result = service.clarity(owner_id="example-client", filename="source.png", seed=999)

    metadata = dict(saved_metadata["extra_metadata"])
    assert metadata["mode"] == "api_clarity"
    assert metadata["source_filename"] == "source.png"
    assert metadata["source_width"] == 320
    assert metadata["source_height"] == 240
    assert metadata["working_width"] == 640
    assert metadata["working_height"] == 480
    assert metadata["clarity_engine"] == CLARITY_ENGINE_NAME
    assert metadata["clarity_variant"] == "current"
    assert metadata["clarity_fs_ms"] == 4
    assert metadata["clarity_unsharp_stage"] == "pre_downscale"
    assert result["mode"] == "api_clarity"
    assert result["source_filename"] == "source.png"
    assert result["working_width"] == 640
    assert result["working_height"] == 480
    assert result["clarity_engine"] == CLARITY_ENGINE_NAME
    assert result["clarity_variant"] == "current"


def test_inference_clarity_bubbles_missing_model_error(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    service = InferenceService(settings=settings)
    source_path = temp_app_paths.outputs_dir / "example-client" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color=(22, 44, 66)).save(source_path)

    monkeypatch.setattr(
        "app.api.inference_service.get_image",
        lambda _settings, _filename, owner_id=None: {
            "filename": "source.png",
            "output_path": str(source_path),
            "prompt": "hello world",
            "seed": 123,
            "width": 128,
            "height": 128,
            "model_pack": "Rayzist_bf16",
        },
    )

    def fail_pipeline(**kwargs):
        raise RuntimeError("unexpected legacy clarity model dependency")

    monkeypatch.setattr("app.api.inference_service.run_clarity_pipeline", fail_pipeline)

    with pytest.raises(RuntimeError, match="unexpected legacy clarity model dependency"):
        service.clarity(owner_id="example-client", filename="source.png")
