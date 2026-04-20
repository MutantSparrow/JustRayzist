from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from typer.testing import CliRunner

import app.core.worker as worker
from app.cli import main as cli_main
from app.cli.main import cli
from app.core.backends.diffusers_zimage import GenerationResult


class _FakeImg2ImgSession:
    last_request = None
    last_input_size = None

    def __init__(self, settings, model_pack, resource_tier=None):
        self._settings = settings
        self._model_pack = model_pack

    def refine_image(self, input_image, request):
        _FakeImg2ImgSession.last_request = request
        _FakeImg2ImgSession.last_input_size = input_image.size
        return GenerationResult(
            image=Image.new("RGB", input_image.size, color=(30, 60, 90)),
            seed=request.seed,
            steps=6,
            guidance_scale=2.5,
            scheduler_mode="euler",
            backend="diffusers_zimage",
            device="cpu",
            duration_ms=120,
            prompt_original=request.prompt,
            prompt_wildcard_resolved=request.prompt,
            prompt_effective_base=request.prompt,
            prompt_effective=request.prompt,
            prompt_enhanced=False,
            mode="img2img",
            refine_strength=request.refine_strength,
            runtime_profile=self._settings.runtime_profile.name,
            resource_tier=self._settings.resource_tier_controller.current().name,
            execution_mode="model_offload",
        )


def test_img2img_command_writes_output_with_lineage(monkeypatch, temp_app_paths, make_app_settings) -> None:
    runner = CliRunner()
    settings = make_app_settings(paths=temp_app_paths)
    input_path = temp_app_paths.root_dir / "reference.png"
    Image.new("RGB", (2200, 1400), color=(12, 34, 56)).save(input_path, format="PNG")
    output_path = temp_app_paths.outputs_dir / "cli_img2img.png"

    monkeypatch.setattr(cli_main, "load_settings", lambda profile_name=None: settings)
    monkeypatch.setattr(
        cli_main,
        "_load_pack_or_exit",
        lambda _settings, pack_name: SimpleNamespace(name=pack_name, backend_preference=["diffusers"]),
    )
    monkeypatch.setattr(worker, "GenerationSession", _FakeImg2ImgSession, raising=False)

    result = runner.invoke(
        cli,
        [
            "img2img",
            "--input",
            str(input_path),
            "--prompt",
            "hello world",
            "--pack",
            "Rayzist_bf16",
            "--similarity",
            "0.8",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output_path.exists()
    assert _FakeImg2ImgSession.last_request is not None
    assert _FakeImg2ImgSession.last_request.refine_strength == pytest.approx(0.2)
    assert _FakeImg2ImgSession.last_input_size[0] * _FakeImg2ImgSession.last_input_size[1] <= 1_500_000

    with Image.open(output_path) as output_image:
        metadata = output_image.info
    assert metadata["mode"] == "img2img"
    assert metadata["source_filename"] == input_path.name
    assert metadata["similarity"] == "0.8"
