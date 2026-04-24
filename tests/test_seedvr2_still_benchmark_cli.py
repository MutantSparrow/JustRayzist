from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

import app.core.seedvr2 as seedvr2_core
from app.cli import main as cli_main
from app.cli.main import cli


class _FakeSeedVR2Result:
    def __init__(self, *, image: Image.Image, runtime_preset: str) -> None:
        self.image = image
        self.infer_ms = 111
        self.total_ms = 111
        self.duration_ms = 111
        self._runtime_preset = runtime_preset

    def telemetry_dict(self):
        return {
            "upscale_infer_ms": self.infer_ms,
            "upscale_total_ms": self.total_ms,
            "upscale_vram_peak_mb": 512,
            "upscale_runtime_preset": self._runtime_preset,
            "upscale_vae_tiling_policy": "auto",
            "upscale_vae_encode_tiled": False,
            "upscale_vae_encode_tile_size": 1024,
            "upscale_vae_encode_tile_overlap": 128,
            "upscale_vae_decode_tiled": False,
            "upscale_vae_decode_tile_size": 1024,
            "upscale_vae_decode_tile_overlap": 128,
        }


def _write_seedvr2_model_files(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "seedvr2_ema_3b_fp8_e4m3fn.safetensors").write_bytes(b"ok")
    (model_dir / "ema_vae_fp16.safetensors").write_bytes(b"ok")


def test_seedvr2_still_benchmark_records_runtime_preset_and_forwards_choice(
    monkeypatch,
    temp_app_paths,
    make_app_settings,
) -> None:
    runner = CliRunner()
    input_path = temp_app_paths.outputs_dir / "source.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(input_path)
    _write_seedvr2_model_files(temp_app_paths.models_dir / "seedvr2")

    def make_settings(profile_name=None):
        selected_profile = profile_name or "balanced"
        return make_app_settings(
            paths=temp_app_paths,
            runtime_profile_name=selected_profile,
            resource_tier_name=selected_profile,
            override_profile_name=selected_profile,
            auto_resource_tier=False,
        )

    captured: dict[str, object] = {}

    def fake_upscale(**kwargs):
        captured["runtime_preset"] = kwargs.get("runtime_preset")
        source = kwargs["image"]
        return _FakeSeedVR2Result(
            image=Image.new("RGB", (source.width * 2, source.height * 2), color=(40, 50, 60)),
            runtime_preset=str(kwargs.get("runtime_preset")),
        )

    monkeypatch.setattr(cli_main, "load_settings", make_settings)
    monkeypatch.setattr(seedvr2_core, "clear_seedvr2_runtime_cache", lambda profile_name=None: None)
    monkeypatch.setattr(seedvr2_core, "upscale_with_seedvr2_direct_x2", fake_upscale)

    result = runner.invoke(
        cli,
        [
            "seedvr2-still-benchmark",
            "--inputs",
            str(input_path),
            "--output-dir",
            str(temp_app_paths.outputs_dir),
            "--presets",
            "seed_faithful",
            "--runtime-preset",
            "current_baseline",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["runtime_preset"] == seedvr2_core.SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE

    csv_reports = sorted(temp_app_paths.data_dir.glob("seedvr2_still_benchmark_*.csv"))
    assert len(csv_reports) == 1

    with csv_reports[0].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["runtime_preset"] == seedvr2_core.SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE


def test_seedvr2_still_benchmark_accepts_highres_auto_runtime_preset(
    monkeypatch,
    temp_app_paths,
    make_app_settings,
) -> None:
    runner = CliRunner()
    input_path = temp_app_paths.outputs_dir / "source.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(input_path)
    _write_seedvr2_model_files(temp_app_paths.models_dir / "seedvr2")

    def make_settings(profile_name=None):
        selected_profile = profile_name or "balanced"
        return make_app_settings(
            paths=temp_app_paths,
            runtime_profile_name=selected_profile,
            resource_tier_name=selected_profile,
            override_profile_name=selected_profile,
            auto_resource_tier=False,
        )

    captured: dict[str, object] = {}

    def fake_upscale(**kwargs):
        captured["runtime_preset"] = kwargs.get("runtime_preset")
        source = kwargs["image"]
        return _FakeSeedVR2Result(
            image=Image.new("RGB", (source.width * 2, source.height * 2), color=(40, 50, 60)),
            runtime_preset=str(kwargs.get("runtime_preset")),
        )

    monkeypatch.setattr(cli_main, "load_settings", make_settings)
    monkeypatch.setattr(seedvr2_core, "clear_seedvr2_runtime_cache", lambda profile_name=None: None)
    monkeypatch.setattr(seedvr2_core, "upscale_with_seedvr2_direct_x2", fake_upscale)

    result = runner.invoke(
        cli,
        [
            "seedvr2-still-benchmark",
            "--inputs",
            str(input_path),
            "--output-dir",
            str(temp_app_paths.outputs_dir),
            "--presets",
            "seed_faithful",
            "--runtime-preset",
            "highres_auto",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["runtime_preset"] == seedvr2_core.SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO
