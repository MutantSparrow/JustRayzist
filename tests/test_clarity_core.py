from __future__ import annotations

from PIL import Image

from app.core import clarity as clarity_core


def test_resolve_clarity_variant_presets() -> None:
    current = clarity_core.resolve_clarity_variant("current")
    assert current.label == "current"
    assert current.pre_downscale_unsharp is True
    assert current.post_downscale_unsharp is False

    post_downscale = clarity_core.resolve_clarity_variant("fs_downsize_final_unsharp")
    assert post_downscale.label == "fs_downsize_final_unsharp"
    assert post_downscale.pre_downscale_unsharp is False
    assert post_downscale.post_downscale_unsharp is True

    fs_only = clarity_core.resolve_clarity_variant("fs_only")
    assert fs_only.label == "fs_only"
    assert fs_only.pre_downscale_unsharp is False
    assert fs_only.post_downscale_unsharp is False


def test_run_clarity_pipeline_current_uses_pre_downscale_unsharp(monkeypatch) -> None:
    source_image = Image.new("RGB", (32, 24), color=(16, 24, 32))
    unsharp_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(clarity_core, "fs_sharpen", lambda image, **kwargs: image)

    def fake_unsharp(image: Image.Image) -> Image.Image:
        unsharp_calls.append(image.size)
        return image

    monkeypatch.setattr(clarity_core, "final_unsharp", fake_unsharp)

    result = clarity_core.run_clarity_pipeline(
        image=source_image,
        variant="current",
    )

    assert result.variant_label == "current"
    assert result.image.size == source_image.size
    assert result.working_width == 64
    assert result.working_height == 64
    assert result.step_timings_ms["clarity_pre_downscale_unsharp_ms"] >= 0
    assert result.step_timings_ms["clarity_post_downscale_unsharp_ms"] == 0
    assert unsharp_calls == [(64, 64)]


def test_run_clarity_pipeline_post_downscale_variant_uses_final_unsharp_after_resize(monkeypatch) -> None:
    source_image = Image.new("RGB", (20, 12), color=(50, 60, 70))
    unsharp_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(clarity_core, "fs_sharpen", lambda image, **kwargs: image)

    def fake_unsharp(image: Image.Image) -> Image.Image:
        unsharp_calls.append(image.size)
        return image

    monkeypatch.setattr(clarity_core, "final_unsharp", fake_unsharp)

    result = clarity_core.run_clarity_pipeline(
        image=source_image,
        variant="fs_downsize_final_unsharp",
    )

    telemetry = result.telemetry_dict()
    assert result.variant_label == "fs_downsize_final_unsharp"
    assert result.image.size == source_image.size
    assert unsharp_calls == [source_image.size]
    assert result.step_timings_ms["clarity_pre_downscale_unsharp_ms"] == 0
    assert result.step_timings_ms["clarity_post_downscale_unsharp_ms"] >= 0
    assert telemetry["clarity_variant"] == "fs_downsize_final_unsharp"
    assert telemetry["working_width"] == 64
    assert telemetry["working_height"] == 64
