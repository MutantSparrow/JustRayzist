from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.core import deblur as deblur_core


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.last_request = None
        self.last_input_size = None

    def refine_image(self, input_image, request):
        self.calls.append(("img2img", input_image.size))
        self.last_request = request
        self.last_input_size = input_image.size
        return SimpleNamespace(
            image=Image.new("RGB", input_image.size, color=(20, 40, 60)),
            duration_ms=111,
            device="cuda:0",
        )


def test_run_deblur_pipeline_uses_fixed_prompt_and_order(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (180, 120), color=(10, 20, 30))
    session = _FakeSession()
    call_order: list[str] = []

    monkeypatch.setattr(deblur_core, "ensure_deblur_prerequisites", lambda settings: call_order.append("prereq"))

    def fake_seedvr2(**kwargs):
        call_order.append("seed")
        assert kwargs["image"].size == session.last_input_size
        assert kwargs["runtime_profile"] == settings.runtime_profile.name
        assert kwargs["seed"] == 37717
        still_cfg = kwargs["still_image_config"]
        assert still_cfg.input_noise_scale == 0.0
        assert still_cfg.latent_noise_scale == 0.0
        assert still_cfg.color_correction == "lab"
        return SimpleNamespace(
            image=Image.new("RGB", (session.last_input_size[0] * 2, session.last_input_size[1] * 2), color=(70, 80, 90)),
            duration_ms=222,
            device="cuda:0",
        )

    def fake_clarity(image):
        call_order.append("clarity")
        assert image.size == (session.last_input_size[0] * 2, session.last_input_size[1] * 2)
        return image.copy(), {
            "clarity_fs_ms": 9,
            "clarity_pre_downscale_unsharp_ms": 5,
        }

    monkeypatch.setattr(deblur_core, "upscale_with_seedvr2_direct_x2", fake_seedvr2)
    monkeypatch.setattr(deblur_core, "apply_clarity_sharpen_core", fake_clarity)

    result = deblur_core.run_deblur_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        seed=37717,
        similarity=0.85,
        scheduler_mode="euler",
    )

    assert call_order == ["prereq", "seed", "clarity"]
    assert session.calls == [("img2img", session.last_input_size)]
    assert result.image.size == source_image.size
    assert result.source_width == source_image.width
    assert result.source_height == source_image.height
    assert result.working_width == session.last_input_size[0] * 2
    assert result.working_height == session.last_input_size[1] * 2
    assert result.prompt == deblur_core.DEBLUR_PROMPT
    assert result.similarity == 0.85
    assert result.seed == 37717
    assert session.last_request.prompt == deblur_core.DEBLUR_PROMPT
    assert session.last_request.enhance_prompt is False
    assert session.last_request.procedural_creativity == 0
    assert session.last_request.scheduler_mode == "euler"
    assert session.last_request.refine_strength == pytest.approx(0.15)
    assert result.step_timings_ms["deblur_img2img_ms"] == 111
    assert result.step_timings_ms["deblur_seed_ms"] == 222
    assert result.step_timings_ms["clarity_fs_ms"] == 9
    assert result.step_timings_ms["clarity_pre_downscale_unsharp_ms"] == 5


def test_run_deblur_pipeline_uses_generated_seed_when_missing(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (96, 96), color=(10, 20, 30))
    session = _FakeSession()

    monkeypatch.setattr(deblur_core, "ensure_deblur_prerequisites", lambda settings: None)
    monkeypatch.setattr(deblur_core.random, "randint", lambda start, end: 24680)
    monkeypatch.setattr(
        deblur_core,
        "upscale_with_seedvr2_direct_x2",
        lambda **kwargs: SimpleNamespace(
            image=Image.new("RGB", (kwargs["image"].width * 2, kwargs["image"].height * 2), color=(1, 2, 3)),
            total_ms=333,
            device="cuda:0",
        ),
    )
    monkeypatch.setattr(
        deblur_core,
        "apply_clarity_sharpen_core",
        lambda image: (image, {"clarity_fs_ms": 4, "clarity_pre_downscale_unsharp_ms": 2}),
    )

    result = deblur_core.run_deblur_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
    )

    assert session.last_request.seed == 24680
    assert result.seed == 24680
    assert result.step_timings_ms["deblur_seed_ms"] == 333


def test_restore_original_chroma_changes_chroma_only() -> None:
    original = Image.new("RGB", (16, 16), color=(160, 90, 70))
    ai_image = Image.new("RGB", (16, 16), color=(70, 120, 200))

    restored = deblur_core.restore_original_chroma(original, ai_image)

    assert restored.size == ai_image.size
    orig_l, _, _ = deblur_core._lab_float_channels(original)
    ai_l, ai_a, ai_b = deblur_core._lab_float_channels(ai_image)
    restored_l, restored_a, restored_b = deblur_core._lab_float_channels(restored)
    assert float(abs(restored_l.mean() - ai_l.mean())) < 2.0
    assert float(abs(restored_a.mean() - ai_a.mean())) > 1.0
    assert float(abs(restored_b.mean() - ai_b.mean())) > 1.0


def test_restore_original_luma_highpass_changes_luminance_only() -> None:
    base = np.tile(np.linspace(32, 224, 32, dtype=np.uint8), (32, 1))
    original = Image.fromarray(np.stack([base, base, base], axis=-1), mode="RGB")
    ai_image = Image.new("RGB", (32, 32), color=(120, 120, 120))

    restored = deblur_core.restore_original_luma_highpass(original, ai_image)

    assert restored.size == ai_image.size
    ai_l, ai_a, ai_b = deblur_core._lab_float_channels(ai_image)
    restored_l, restored_a, restored_b = deblur_core._lab_float_channels(restored)
    assert float(abs(restored_l.mean() - ai_l.mean())) > 0.3
    assert float(abs(restored_a.mean() - ai_a.mean())) < 1.0
    assert float(abs(restored_b.mean() - ai_b.mean())) < 1.0


def test_combined_texture_variant_applies_highpass_then_chroma() -> None:
    original = Image.new("RGB", (20, 20), color=(150, 100, 80))
    ai_image = Image.new("RGB", (20, 20), color=(80, 130, 180))

    expected = deblur_core.restore_original_chroma(
        original,
        deblur_core.restore_original_luma_highpass(original, ai_image),
    )
    combined = deblur_core.run_deblur_texture_variant(
        original_image=original,
        ai_final_image=ai_image,
        variant="orig_chroma_plus_highpass",
    )

    assert list(expected.getdata()) == list(combined.getdata())


def test_run_fidelity_upscale_core_photo_uses_seed_then_img2img(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()

    monkeypatch.setattr(deblur_core, "ensure_deblur_prerequisites", lambda settings: None)
    monkeypatch.setattr(
        deblur_core,
        "upscale_with_seedvr2_direct_x2",
        lambda **kwargs: SimpleNamespace(
            image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
            duration_ms=222,
            device="cuda:0",
        ),
    )

    result = deblur_core.run_fidelity_upscale_core(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        content_type="photo",
        seed=777,
        scheduler_mode="euler",
    )

    assert result.content_type == "photo"
    assert result.x2_image.size == (80, 48)
    assert result.source_x2_image.size == (80, 48)
    assert result.similarity == pytest.approx(0.90)
    assert session.last_input_size == (80, 48)
    assert session.last_request.refine_strength == pytest.approx(0.10)
    assert session.last_request.scheduler_mode == "euler"
    assert result.step_timings_ms["fidelity_seed_ms"] == 222
    assert result.step_timings_ms["fidelity_img2img_ms"] == 111


def test_run_fidelity_upscale_core_photo_accepts_similarity_override(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()

    monkeypatch.setattr(deblur_core, "ensure_deblur_prerequisites", lambda settings: None)
    monkeypatch.setattr(
        deblur_core,
        "upscale_with_seedvr2_direct_x2",
        lambda **kwargs: SimpleNamespace(
            image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
            duration_ms=222,
            device="cuda:0",
        ),
    )

    result = deblur_core.run_fidelity_upscale_core(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        content_type="photo",
        seed=777,
        scheduler_mode="euler",
        photo_similarity_override=0.85,
    )

    assert result.similarity == pytest.approx(0.85)
    assert session.last_request.refine_strength == pytest.approx(0.15)


def test_run_fidelity_upscale_core_art_skips_img2img(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()

    monkeypatch.setattr(deblur_core, "ensure_deblur_prerequisites", lambda settings: None)
    monkeypatch.setattr(
        deblur_core,
        "upscale_with_seedvr2_direct_x2",
        lambda **kwargs: SimpleNamespace(
            image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
            total_ms=333,
            device="cuda:0",
        ),
    )

    result = deblur_core.run_fidelity_upscale_core(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        content_type="art",
        seed=888,
    )

    assert result.content_type == "art"
    assert result.x2_image.size == (80, 48)
    assert result.similarity is None
    assert session.last_request is None
    assert result.step_timings_ms["fidelity_seed_ms"] == 333
    assert result.step_timings_ms["fidelity_img2img_ms"] == 0


def test_run_default_upscale_pipeline_uses_fidelity_core_then_fs(monkeypatch, temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()
    call_order: list[str] = []

    monkeypatch.setattr(
        deblur_core,
        "run_fidelity_upscale_core",
        lambda **kwargs: (
            call_order.append("core"),
            deblur_core.DeblurFidelityCoreResult(
                x2_image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
                source_x2_image=Image.new("RGB", (80, 48), color=(30, 40, 50)),
                source_width=40,
                source_height=24,
                working_width=80,
                working_height=48,
                seed=777,
                similarity=0.90,
                device="cuda:0",
                content_type="photo",
                step_timings_ms={"fidelity_seed_ms": 12, "fidelity_img2img_ms": 34},
            ),
        )[1],
    )

    def fake_fs(image, **kwargs):
        call_order.append("fs")
        assert kwargs["method"] == deblur_core.CLARITY_FS_METHOD
        assert kwargs["blur_type"] == deblur_core.CLARITY_FS_TYPE
        assert kwargs["intensity"] == deblur_core.DEFAULT_UPSCALE_FS_INTENSITY
        return image.copy()

    monkeypatch.setattr(deblur_core, "fs_sharpen", fake_fs)

    result = deblur_core.run_default_upscale_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        seed=777,
        scheduler_mode="euler",
    )

    assert call_order == ["core", "fs"]
    assert result.image.size == (80, 48)
    assert result.engine_name == deblur_core.DEFAULT_UPSCALE_ENGINE_NAME
    assert result.step_timings_ms["fidelity_seed_ms"] == 12
    assert result.step_timings_ms["fidelity_img2img_ms"] == 34
    assert result.step_timings_ms["upscale_fs_ms"] >= 0


def test_run_default_upscale_pipeline_uses_upscale_specific_similarity_override(
    monkeypatch, temp_app_paths, make_app_settings
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()
    captured: dict[str, object] = {}

    def fake_core(**kwargs):
        captured["photo_similarity_override"] = kwargs.get("photo_similarity_override")
        return deblur_core.DeblurFidelityCoreResult(
            x2_image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
            source_x2_image=Image.new("RGB", (80, 48), color=(30, 40, 50)),
            source_width=40,
            source_height=24,
            working_width=80,
            working_height=48,
            seed=777,
            similarity=0.85,
            device="cuda:0",
            content_type="photo",
            step_timings_ms={"fidelity_seed_ms": 12, "fidelity_img2img_ms": 34},
        )

    monkeypatch.setattr(deblur_core, "run_fidelity_upscale_core", fake_core)
    monkeypatch.setattr(deblur_core, "fs_sharpen", lambda image, **kwargs: image.copy())

    deblur_core.run_default_upscale_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        seed=777,
        scheduler_mode="euler",
    )

    assert captured["photo_similarity_override"] == pytest.approx(deblur_core.DEFAULT_UPSCALE_PHOTO_SIMILARITY)


def test_run_default_clarity_pipeline_uses_mb_chroma_edgeaware_fs_unsharp_then_shrink(
    monkeypatch, temp_app_paths, make_app_settings
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()
    call_order: list[str] = []

    monkeypatch.setattr(
        deblur_core,
        "run_fidelity_upscale_core",
        lambda **kwargs: (
            call_order.append("core"),
            deblur_core.DeblurFidelityCoreResult(
                x2_image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
                source_x2_image=Image.new("RGB", (80, 48), color=(30, 40, 50)),
                source_width=40,
                source_height=24,
                working_width=80,
                working_height=48,
                seed=777,
                similarity=0.90,
                device="cuda:0",
                content_type="photo",
                step_timings_ms={"fidelity_seed_ms": 12, "fidelity_img2img_ms": 34},
            ),
        )[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "transfer_multiband_detail",
        lambda *args, **kwargs: (call_order.append("multiband"), Image.new("RGB", (80, 48), color=(80, 90, 100)))[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "restore_original_chroma_x2",
        lambda *args, **kwargs: (call_order.append("chroma"), Image.new("RGB", (80, 48), color=(100, 110, 120)))[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "apply_edge_aware_sharpen",
        lambda *args, **kwargs: (call_order.append("edgeaware"), Image.new("RGB", (80, 48), color=(120, 130, 140)))[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "fs_sharpen",
        lambda *args, **kwargs: (call_order.append("fs"), Image.new("RGB", (80, 48), color=(140, 150, 160)))[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "final_unsharp",
        lambda image: (call_order.append("unsharp"), image.copy())[1],
    )

    result = deblur_core.run_default_clarity_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        seed=777,
        scheduler_mode="euler",
    )

    assert call_order == ["core", "multiband", "chroma", "edgeaware", "fs", "unsharp"]
    assert result.image.size == source_image.size
    assert result.engine_name == deblur_core.DEFAULT_CLARITY_ENGINE_NAME
    assert result.variant_label == "multiband_chroma_edgeaware_fs_unsharp_shrink"
    assert result.step_timings_ms["fidelity_seed_ms"] == 12
    assert result.step_timings_ms["clarity_multiband_ms"] >= 0
    assert result.step_timings_ms["clarity_chroma_ms"] >= 0
    assert result.step_timings_ms["clarity_edgeaware_ms"] >= 0
    assert result.step_timings_ms["clarity_fs_ms"] >= 0
    assert result.step_timings_ms["clarity_pre_downscale_unsharp_ms"] >= 0
    assert result.step_timings_ms["clarity_downscale_ms"] >= 0


def test_transfer_multiband_detail_preserves_dimensions_and_changes_pixels() -> None:
    noise = np.random.default_rng(1234).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    source = Image.fromarray(noise, mode="RGB")
    candidate = source.resize((32, 32), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=1.2))

    restored = deblur_core.transfer_multiband_detail(source, candidate, mid_amount=0.45, high_amount=0.65)

    assert restored.size == candidate.size
    assert list(restored.getdata()) != list(candidate.getdata())


def test_apply_edge_aware_sharpen_only_hits_edges_substantially() -> None:
    array = np.full((48, 48, 3), 128, dtype=np.uint8)
    array[:, 24:] = 220
    image = Image.fromarray(array, mode="RGB")

    sharpened = deblur_core.apply_edge_aware_sharpen(image, content_type="photo")

    before = np.asarray(image, dtype=np.int16)
    after = np.asarray(sharpened, dtype=np.int16)
    flat_delta = np.abs(after[8, 8] - before[8, 8]).mean()
    edge_delta = np.abs(after[24, 24] - before[24, 24]).mean()
    assert edge_delta >= flat_delta
