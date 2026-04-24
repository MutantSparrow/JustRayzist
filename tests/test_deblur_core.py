from __future__ import annotations

from types import SimpleNamespace

import cv2
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


def _col_jump(image: Image.Image, column: int) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    deltas = np.abs(np.diff(gray, axis=1))
    index = max(0, min(deltas.shape[1] - 1, int(column) - 1))
    return float(deltas[:, index].mean())


def _row_jump(image: Image.Image, row: int) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    deltas = np.abs(np.diff(gray, axis=0))
    index = max(0, min(deltas.shape[0] - 1, int(row) - 1))
    return float(deltas[index, :].mean())


def _smoothed_col_jump(image: Image.Image, column: int, sigma: float) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    deltas = np.abs(np.diff(smoothed, axis=1))
    index = max(0, min(deltas.shape[1] - 1, int(column) - 1))
    return float(deltas[:, index].mean())


def _smoothed_row_jump(image: Image.Image, row: int, sigma: float) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    deltas = np.abs(np.diff(smoothed, axis=0))
    index = max(0, min(deltas.shape[0] - 1, int(row) - 1))
    return float(deltas[index, :].mean())


def _textured_photo_like_image(size: tuple[int, int] = (128, 128)) -> Image.Image:
    width, height = size
    yy, xx = np.indices((height, width), dtype=np.float32)
    rng = np.random.default_rng(12345)
    noise = rng.normal(0.0, 16.0, size=(height, width, 3)).astype(np.float32)
    base = np.stack(
        [
            105.0 + (xx * 0.35) + (yy * 0.08),
            92.0 + (yy * 0.42),
            88.0 + ((xx + yy) * 0.22),
        ],
        axis=-1,
    )
    image = np.clip(base + noise, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def _flat_art_image(size: tuple[int, int] = (128, 128)) -> Image.Image:
    width, height = size
    image = np.full((height, width, 3), 236, dtype=np.uint8)
    image[:, : width // 2, :] = np.array([248, 214, 144], dtype=np.uint8)
    image[:, width // 2 :, :] = np.array([255, 164, 120], dtype=np.uint8)
    image[height // 2 :, :, :] = np.array([255, 222, 168], dtype=np.uint8)
    image[12:116, 36:92, :] = np.array([255, 132, 102], dtype=np.uint8)
    image[16:112, 40:88, :] = np.array([255, 228, 118], dtype=np.uint8)
    image[20:108, 44:84, :] = np.array([255, 160, 118], dtype=np.uint8)
    image[10:118, 34:36, :] = 16
    image[10:118, 90:92, :] = 16
    image[10:12, 34:92, :] = 16
    image[116:118, 34:92, :] = 16
    return Image.fromarray(image, mode="RGB")


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
    captured: dict[str, object] = {}

    monkeypatch.setattr(deblur_core, "ensure_deblur_prerequisites", lambda settings: None)

    def fake_seedvr2(**kwargs):
        captured["runtime_preset"] = kwargs.get("runtime_preset")
        return SimpleNamespace(
            image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
            duration_ms=222,
            device="cuda:0",
        )

    monkeypatch.setattr(deblur_core, "upscale_with_seedvr2_direct_x2", fake_seedvr2)

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
    assert captured["runtime_preset"] == deblur_core.SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO
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


def test_detect_upscale_content_mode_prefers_art_prompt_for_photo_like_image() -> None:
    decision = deblur_core.detect_upscale_content_mode(
        _textured_photo_like_image(),
        prompt_text="anime illustration, cel shading",
    )

    assert decision.mode == "art"
    assert decision.source == "prompt"
    assert decision.prompt_art_hits >= 3
    assert decision.prompt_photo_hits == 0


def test_detect_upscale_content_mode_prefers_photo_prompt_for_flat_art_image() -> None:
    decision = deblur_core.detect_upscale_content_mode(
        _flat_art_image(),
        prompt_text="editorial photograph, photorealistic skin texture",
    )

    assert decision.mode == "photo"
    assert decision.source == "prompt"
    assert decision.prompt_photo_hits >= 2
    assert decision.prompt_art_hits == 0


def test_detect_upscale_content_mode_uses_image_for_flat_palette_art_without_prompt() -> None:
    decision = deblur_core.detect_upscale_content_mode(_flat_art_image(), prompt_text=None)

    assert decision.mode == "art"
    assert decision.source == "image"
    assert decision.image_art_signals >= 2


def test_detect_upscale_content_mode_defaults_to_photo_for_ambiguous_textured_image() -> None:
    decision = deblur_core.detect_upscale_content_mode(_textured_photo_like_image(), prompt_text=None)

    assert decision.mode == "photo"
    assert decision.source == "image"
    assert decision.image_art_signals < 2


def test_detect_upscale_content_mode_prefers_photo_when_prompt_has_single_photo_score() -> None:
    decision = deblur_core.detect_upscale_content_mode(
        _flat_art_image(),
        prompt_text="realistic rendering",
    )

    assert decision.mode == "photo"
    assert decision.source == "prompt"
    assert decision.prompt_photo_hits == 1


def test_detect_upscale_content_mode_defaults_to_photo_when_prompt_scores_tie() -> None:
    decision = deblur_core.detect_upscale_content_mode(
        _flat_art_image(),
        prompt_text="portrait illustration",
    )

    assert decision.mode == "photo"
    assert decision.source == "mixed"
    assert decision.prompt_photo_hits == 1
    assert decision.prompt_art_hits == 1


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
        prompt_text="editorial photograph, photorealistic skin",
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
        prompt_text="editorial photograph, photorealistic skin",
    )

    assert captured["photo_similarity_override"] == pytest.approx(deblur_core.DEFAULT_UPSCALE_PHOTO_SIMILARITY)


def test_run_default_upscale_pipeline_uses_current_baseline_runtime_preset(
    monkeypatch, temp_app_paths, make_app_settings
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()
    captured: dict[str, object] = {}

    def fake_core(**kwargs):
        captured["seed_runtime_preset"] = kwargs.get("seed_runtime_preset")
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
        prompt_text="editorial photograph, photorealistic skin",
    )

    assert captured["seed_runtime_preset"] == deblur_core.SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE


def test_run_default_upscale_pipeline_runs_seam_repair_only_for_tiled_decode(
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
                similarity=0.85,
                device="cuda:0",
                content_type="photo",
                step_timings_ms={"fidelity_seed_ms": 12, "fidelity_img2img_ms": 34},
                seed_decode_tiled=True,
                seed_decode_tile_size=1024,
                seed_decode_tile_overlap=256,
            ),
        )[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "repair_photo_upscale_tile_drift",
        lambda source_x2_image, candidate_image, *, tile_size, tile_overlap: (
            call_order.append("repair"),
            (candidate_image.copy(), 32.0),
        )[1],
    )

    def fake_fs(image, **kwargs):
        call_order.append("fs")
        return image.copy()

    monkeypatch.setattr(deblur_core, "fs_sharpen", fake_fs)

    result = deblur_core.run_default_upscale_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        seed=777,
        scheduler_mode="euler",
        prompt_text="editorial photograph, photorealistic skin",
    )

    assert call_order == ["core", "repair", "fs"]
    telemetry = result.telemetry_dict()
    assert telemetry["upscale_seam_repair_applied"] is True
    assert telemetry["upscale_seam_repair_mode"] == "source_guided_lowfreq"
    assert telemetry["upscale_seam_repair_sigma"] == pytest.approx(32.0)
    assert telemetry["upscale_seam_repair_decode_overlap"] == 256
    assert telemetry["upscale_seam_repair_ms"] >= 0


def test_run_default_upscale_pipeline_skips_seam_repair_for_untiled_decode(
    monkeypatch, temp_app_paths, make_app_settings
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = Image.new("RGB", (40, 24), color=(10, 20, 30))
    session = _FakeSession()

    monkeypatch.setattr(
        deblur_core,
        "run_fidelity_upscale_core",
        lambda **kwargs: deblur_core.DeblurFidelityCoreResult(
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
            seed_decode_tiled=False,
            seed_decode_tile_size=None,
            seed_decode_tile_overlap=None,
        ),
    )
    monkeypatch.setattr(
        deblur_core,
        "repair_photo_upscale_tile_drift",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("repair should not run")),
    )
    monkeypatch.setattr(deblur_core, "fs_sharpen", lambda image, **kwargs: image.copy())

    result = deblur_core.run_default_upscale_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        seed=777,
        scheduler_mode="euler",
        prompt_text="editorial photograph, photorealistic skin",
    )

    telemetry = result.telemetry_dict()
    assert telemetry["upscale_seam_repair_applied"] is False
    assert telemetry["upscale_seam_repair_mode"] == "none"
    assert telemetry["upscale_seam_repair_sigma"] == 0.0
    assert telemetry["upscale_seam_repair_decode_overlap"] == 0


def test_run_default_upscale_pipeline_auto_art_path_uses_art_branch(
    monkeypatch, temp_app_paths, make_app_settings
) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    source_image = _textured_photo_like_image(size=(40, 24))
    session = _FakeSession()
    call_order: list[str] = []
    captured: dict[str, object] = {}

    def fake_core(**kwargs):
        call_order.append("core")
        captured["content_type"] = kwargs.get("content_type")
        return deblur_core.DeblurFidelityCoreResult(
            x2_image=Image.new("RGB", (80, 48), color=(60, 70, 80)),
            source_x2_image=Image.new("RGB", (80, 48), color=(30, 40, 50)),
            source_width=40,
            source_height=24,
            working_width=80,
            working_height=48,
            seed=777,
            similarity=None,
            device="cuda:0",
            content_type="art",
            step_timings_ms={"fidelity_seed_ms": 12, "fidelity_img2img_ms": 0},
            seed_decode_tiled=True,
            seed_decode_tile_size=1024,
            seed_decode_tile_overlap=256,
        )

    monkeypatch.setattr(deblur_core, "run_fidelity_upscale_core", fake_core)
    monkeypatch.setattr(
        deblur_core,
        "repair_art_upscale_tile_drift",
        lambda source_x2_image, candidate_image, *, tile_size, tile_overlap: (
            call_order.append("repair_art"),
            (candidate_image.copy(), 24.0),
        )[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "transfer_multiband_detail",
        lambda *args, **kwargs: (call_order.append("multiband"), Image.new("RGB", (80, 48), color=(90, 100, 110)))[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "restore_original_chroma_x2",
        lambda *args, **kwargs: (call_order.append("chroma"), Image.new("RGB", (80, 48), color=(120, 130, 140)))[1],
    )
    monkeypatch.setattr(
        deblur_core,
        "fs_sharpen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fs should not run for art")),
    )

    result = deblur_core.run_default_upscale_pipeline(
        image=source_image,
        settings=settings,
        session=session,
        profile_name=settings.runtime_profile.name,
        seed=777,
        prompt_text="anime illustration, cel shading",
    )

    assert captured["content_type"] == "art"
    assert call_order == ["core", "repair_art", "multiband", "chroma"]
    telemetry = result.telemetry_dict()
    assert telemetry["upscale_auto_content_mode"] == "art"
    assert telemetry["upscale_auto_content_source"] == "prompt"
    assert telemetry["upscale_pipeline_variant"] == "art_preserve"
    assert telemetry["upscale_seam_repair_applied"] is True
    assert telemetry["upscale_fs_ms"] == 0
    assert telemetry["upscale_art_multiband_ms"] >= 0
    assert telemetry["upscale_art_chroma_ms"] >= 0


def test_repair_photo_upscale_tile_drift_reduces_synthetic_tile_jumps_without_blunting_sharpness() -> None:
    height = 192
    width = 192
    yy, xx = np.indices((height, width), dtype=np.float32)
    base_r = 110.0 + (xx * 0.35) + (yy * 0.05)
    base_g = 105.0 + (yy * 0.30)
    base_b = 100.0 + ((xx + yy) * 0.18)
    source = np.stack([base_r, base_g, base_b], axis=-1)
    source[:, 72:120, :] += np.array([12.0, -4.0, -2.0], dtype=np.float32)
    source[72:120, :, :] += np.array([-6.0, 5.0, 8.0], dtype=np.float32)
    source += ((np.sin(xx / 5.0) + np.cos(yy / 7.0)) * 2.0)[..., None]
    source_image = Image.fromarray(np.clip(source, 0.0, 255.0).round().astype(np.uint8), mode="RGB")

    candidate = source.copy()
    biases = {
        (0, 0): np.array([-18.0, 8.0, 4.0], dtype=np.float32),
        (0, 1): np.array([10.0, -7.0, 6.0], dtype=np.float32),
        (1, 0): np.array([14.0, 6.0, -10.0], dtype=np.float32),
        (1, 1): np.array([-12.0, -5.0, 8.0], dtype=np.float32),
    }
    for row_index, y0 in enumerate((0, 96)):
        for col_index, x0 in enumerate((0, 96)):
            y1 = min(height, y0 + 96)
            x1 = min(width, x0 + 96)
            candidate[y0:y1, x0:x1, :] += biases[(row_index, col_index)]
    candidate_image = Image.fromarray(np.clip(candidate, 0.0, 255.0).round().astype(np.uint8), mode="RGB")

    repaired_image, _sigma = deblur_core.repair_photo_upscale_tile_drift(
        source_image,
        candidate_image,
        tile_size=128,
        tile_overlap=32,
    )

    assert _smoothed_col_jump(repaired_image, 96, sigma=8.0) < (
        _smoothed_col_jump(candidate_image, 96, sigma=8.0) * 0.85
    )
    assert _smoothed_row_jump(repaired_image, 96, sigma=8.0) < (
        _smoothed_row_jump(candidate_image, 96, sigma=8.0) * 0.85
    )
    candidate_luma = np.asarray(candidate_image.convert("L"), dtype=np.float32)
    repaired_luma = np.asarray(repaired_image.convert("L"), dtype=np.float32)
    sharpness_ratio = deblur_core._gradient_energy(repaired_luma) / max(
        deblur_core._gradient_energy(candidate_luma),
        1e-6,
    )
    assert 0.90 <= sharpness_ratio <= 1.10


def test_repair_art_upscale_tile_drift_reduces_synthetic_tile_jumps_without_softening_line_art() -> None:
    source_image = _flat_art_image(size=(192, 192))
    source = np.asarray(source_image, dtype=np.float32)
    candidate = source.copy()
    biases = {
        (0, 0): -18.0,
        (0, 1): 12.0,
        (1, 0): 9.0,
        (1, 1): -15.0,
    }
    for row_index, y0 in enumerate((0, 96)):
        for col_index, x0 in enumerate((0, 96)):
            y1 = min(192, y0 + 96)
            x1 = min(192, x0 + 96)
            candidate[y0:y1, x0:x1, :] = np.clip(candidate[y0:y1, x0:x1, :] + biases[(row_index, col_index)], 0.0, 255.0)
    candidate_image = Image.fromarray(candidate.round().astype(np.uint8), mode="RGB")

    repaired_image, _sigma = deblur_core.repair_art_upscale_tile_drift(
        source_image,
        candidate_image,
        tile_size=128,
        tile_overlap=32,
    )

    assert _smoothed_col_jump(repaired_image, 96, sigma=8.0) < (
        _smoothed_col_jump(candidate_image, 96, sigma=8.0) * 0.85
    )
    assert _smoothed_row_jump(repaired_image, 96, sigma=8.0) <= (
        _smoothed_row_jump(candidate_image, 96, sigma=8.0) * 1.02
    )
    candidate_luma = np.asarray(candidate_image.convert("L"), dtype=np.float32)
    repaired_luma = np.asarray(repaired_image.convert("L"), dtype=np.float32)
    sharpness_ratio = deblur_core._gradient_energy(repaired_luma) / max(
        deblur_core._gradient_energy(candidate_luma),
        1e-6,
    )
    assert 0.92 <= sharpness_ratio <= 1.08


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
