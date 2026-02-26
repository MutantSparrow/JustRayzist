from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.config.profiles import RUNTIME_PROFILES
from app.config.settings import AppPaths, AppSettings
from app.core.backends import DiffusersZImageBackend
from app.core.model_registry import ModelPack
from app.core.worker.types import GenerationRequest


ROOT = Path(__file__).resolve().parents[1]


def _settings_for_profile(profile_name: str) -> AppSettings:
    paths = AppPaths(
        root_dir=ROOT,
        models_dir=ROOT / "models",
        model_packs_dir=ROOT / "models" / "packs",
        outputs_dir=ROOT / "outputs",
        data_dir=ROOT / "data",
        ui_dir=ROOT / "app" / "ui",
    )
    return AppSettings(
        app_name="JustRayzist",
        app_version="0.1.0",
        environment="test",
        offline_mode=True,
        runtime_profile=RUNTIME_PROFILES[profile_name],
        paths=paths,
    )


def _dummy_pack() -> ModelPack:
    return ModelPack(
        name="test_pack",
        architecture="z_image_turbo",
        backend_preference=["diffusers"],
        components={},
        pipeline_config_dir=None,
        required_configs=[],
        source_file=ROOT / "models" / "packs" / "test_pack" / "modelpack.yaml",
    )


def test_refine_tiling_defaults_match_profile() -> None:
    request = GenerationRequest(prompt="p", width=1024, height=1024)
    expected_tiles = {"high": 0, "balanced": 1280, "constrained": 896}

    for profile_name, expected in expected_tiles.items():
        backend = DiffusersZImageBackend(
            settings=_settings_for_profile(profile_name),
            model_pack=_dummy_pack(),
        )
        tile_size, overlap = backend._resolve_refine_tiling(request, width=2048, height=2048)
        assert tile_size == expected
        assert overlap == 64


def test_refine_tiling_override_honors_request() -> None:
    backend = DiffusersZImageBackend(
        settings=_settings_for_profile("balanced"),
        model_pack=_dummy_pack(),
    )
    request = GenerationRequest(
        prompt="p",
        width=1024,
        height=1024,
        refine_tile_size=512,
        refine_tile_overlap=4,
    )
    tile_size, overlap = backend._resolve_refine_tiling(request, width=2048, height=2048)
    assert tile_size == 512
    assert overlap == 8


def test_refine_tiling_disables_tiles_for_small_inputs() -> None:
    backend = DiffusersZImageBackend(
        settings=_settings_for_profile("balanced"),
        model_pack=_dummy_pack(),
    )
    request = GenerationRequest(prompt="p", width=512, height=512)
    tile_size, overlap = backend._resolve_refine_tiling(request, width=512, height=512)
    assert tile_size == 0
    assert overlap == 64


def test_sharpen_defaults_and_override() -> None:
    default_request = GenerationRequest(prompt="p", width=1024, height=1024)
    enabled, sigma, amount, threshold = DiffusersZImageBackend._resolve_sharpen_params(default_request)
    assert enabled is True
    assert sigma == 1.0
    assert amount == 0.35
    assert threshold == 3

    override_request = GenerationRequest(
        prompt="p",
        width=1024,
        height=1024,
        sharpen_enabled=False,
        sharpen_sigma=0.5,
        sharpen_amount=0.1,
        sharpen_threshold=7,
    )
    enabled, sigma, amount, threshold = DiffusersZImageBackend._resolve_sharpen_params(override_request)
    assert enabled is False
    assert sigma == 0.5
    assert amount == 0.1
    assert threshold == 7


def test_luma_unsharp_mask_preserves_size_and_changes_edges() -> None:
    image = Image.new("RGB", (16, 16), (128, 64, 64))
    for y in range(16):
        for x in range(16):
            if x < 8:
                image.putpixel((x, y), (60, 20, 20))
            else:
                image.putpixel((x, y), (220, 120, 120))

    sharpened = DiffusersZImageBackend._apply_luma_unsharp_mask(
        image,
        sigma=1.0,
        amount=0.35,
        threshold=0,
    )
    assert sharpened.mode == "RGB"
    assert sharpened.size == image.size
    assert sharpened.tobytes() != image.tobytes()


def test_scheduler_dpm_falls_back_to_euler_for_img2img_without_scale_noise(monkeypatch) -> None:
    import diffusers

    class DummyDpmScheduler:
        def __init__(self, config: dict):
            self.config = dict(config)

        @classmethod
        def from_config(cls, config: dict, **kwargs):
            merged = dict(config)
            merged.update(kwargs)
            merged["scheduler"] = "dpm"
            return cls(merged)

    class DummyEulerScheduler:
        def __init__(self, config: dict):
            self.config = dict(config)

        @classmethod
        def from_config(cls, config: dict, **kwargs):
            merged = dict(config)
            merged.update(kwargs)
            merged["scheduler"] = "euler"
            return cls(merged)

        def scale_noise(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(diffusers, "DPMSolverMultistepScheduler", DummyDpmScheduler)
    monkeypatch.setattr(diffusers, "FlowMatchEulerDiscreteScheduler", DummyEulerScheduler)

    backend = DiffusersZImageBackend(
        settings=_settings_for_profile("balanced"),
        model_pack=_dummy_pack(),
    )

    Img2ImgPipe = type("SomeImg2ImgPipeline", (), {})
    pipe = Img2ImgPipe()
    pipe.scheduler = type("BaseScheduler", (), {"config": {"shift": 3.0}})()

    applied_mode = backend._apply_scheduler_mode(pipe, "dpm")
    assert applied_mode == "euler"
    assert pipe.scheduler.config["scheduler"] == "euler"
    assert hasattr(pipe.scheduler, "scale_noise")


def test_rewrite_rejection_shorter_than_input() -> None:
    original = (
        "A towering mechanical heron robot with angular white and orange paneled body, long sharp beak, "
        "and thin jointed legs, standing protectively over a robed figure."
    )
    rewritten = "Mechanical heron robot over a robed figure in a blurred spaceport."
    reason = DiffusersZImageBackend._rewrite_rejection_reason(original, rewritten)
    assert reason == "shorter_than_input"


def test_rewrite_quality_rejects_shorter_candidate() -> None:
    original = "Detailed cinematic portrait with environmental context, lensing, and realistic textures."
    rewritten = "Cinematic portrait."
    assert DiffusersZImageBackend._rewrite_quality_ok(original, rewritten) is False
