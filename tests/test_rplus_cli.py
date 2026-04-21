from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from typer.testing import CliRunner

import app.core.worker as worker
from app.cli import main as cli_main
from app.cli.main import cli
from app.core.backends.diffusers_zimage import GenerationResult
from app.core.memory import CudaMemorySnapshot


class _FakeRplusSession:
    requests: list[object] = []

    def __init__(self, settings, model_pack, resource_tier=None):
        self._settings = settings
        self._model_pack = model_pack

    def generate(self, request):
        type(self).requests.append(request)
        inference_process = getattr(request, "inference_process", "standard")
        is_rplus = inference_process == "rplus"
        guidance_scale = (
            1.0
            if is_rplus
            else (
                request.guidance_scale
                if request.guidance_scale is not None
                else self._settings.runtime_profile.guidance_scale_default
            )
        )
        color = (72, 48, 24) if not is_rplus else (36, 84, 120)
        reserved_bytes = 8_100_000_000 if not is_rplus else 7_300_000_000
        return GenerationResult(
            image=Image.new("RGB", (request.width, request.height), color=color),
            seed=request.seed,
            steps=request.steps or self._settings.runtime_profile.steps_default,
            guidance_scale=guidance_scale,
            scheduler_mode="euler",
            backend="diffusers_zimage",
            device="cpu",
            duration_ms=1440 if not is_rplus else 1180,
            prompt_original=request.prompt,
            prompt_effective=request.prompt,
            prompt_enhanced=False,
            runtime_profile=self._settings.runtime_profile.name,
            resource_tier=self._settings.resource_tier_controller.current().name,
            execution_mode="full_cuda",
            inference_process=inference_process,
            rplus_vibrance=request.rplus_vibrance if is_rplus else None,
            rplus_initial_bias_level=request.rplus_initial_bias_level if is_rplus else None,
            rplus_initial_sample_size=request.rplus_initial_sample_size if is_rplus else None,
            rplus_effective_initial_noise_bias_level=2.5 if is_rplus else None,
            rplus_stage3_seed=37_717 if is_rplus else None,
            rplus_stage_count=3 if is_rplus else None,
            rplus_stage1_steps=1 if is_rplus else None,
            rplus_stage2_steps=5 if is_rplus else None,
            rplus_stage3_steps=3 if is_rplus else None,
            rplus_stage1_ran=True if is_rplus else None,
            rplus_stage2_ran=True if is_rplus else None,
            rplus_stage3_ran=True if is_rplus else None,
            cuda_memory_after=CudaMemorySnapshot(
                allocated_bytes=reserved_bytes - 600_000_000,
                reserved_bytes=reserved_bytes,
                max_allocated_bytes=reserved_bytes - 200_000_000,
                max_reserved_bytes=reserved_bytes + 250_000_000,
            ),
        )


def test_generate_command_wires_rplus_request_and_metadata(
    monkeypatch,
    temp_app_paths,
    make_app_settings,
) -> None:
    runner = CliRunner()
    settings = make_app_settings(paths=temp_app_paths)
    output_path = temp_app_paths.outputs_dir / "rplus_generate.png"
    _FakeRplusSession.requests = []

    monkeypatch.setattr(cli_main, "load_settings", lambda profile_name=None: settings)
    monkeypatch.setattr(
        cli_main,
        "_load_pack_or_exit",
        lambda _settings, pack_name: SimpleNamespace(name=pack_name, backend_preference=["diffusers"]),
    )
    monkeypatch.setattr(worker, "GenerationSession", _FakeRplusSession, raising=False)

    result = runner.invoke(
        cli,
        [
            "generate",
            "--prompt",
            "rplus prompt",
            "--pack",
            "Rayzist_bf16",
            "--width",
            "8",
            "--height",
            "8",
            "--guidance-scale",
            "6.5",
            "--inference-process",
            "rplus",
            "--rplus-vibrance",
            "0.25",
            "--rplus-initial-bias-level",
            "0.1",
            "--rplus-initial-sample-size",
            "192",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Rplus overrides guidance_scale to 1.0." in result.stdout
    assert output_path.exists()

    request = _FakeRplusSession.requests[-1]
    assert request.inference_process == "rplus"
    assert request.guidance_scale == 1.0
    assert request.rplus_vibrance == 0.25
    assert request.rplus_initial_bias_level == 0.1
    assert request.rplus_initial_sample_size == "192"

    with Image.open(output_path) as output_image:
        metadata = output_image.info

    assert metadata["inference_process"] == "rplus"
    assert metadata["rplus_stage3_seed"] == "37717"
    assert metadata["rplus_initial_sample_size"] == "192"
    assert metadata["guidance_scale"] == "1.0"


def test_rplus_compare_command_writes_pair_report(
    monkeypatch,
    temp_app_paths,
    make_app_settings,
) -> None:
    runner = CliRunner()
    settings = make_app_settings(
        paths=temp_app_paths,
        runtime_profile_name="balanced",
        resource_tier_name="balanced",
        override_profile_name="balanced",
        auto_resource_tier=False,
    )
    _FakeRplusSession.requests = []

    monkeypatch.setattr(cli_main, "load_settings", lambda profile_name=None: settings)
    monkeypatch.setattr(
        cli_main,
        "_load_runtime_pack_or_exit",
        lambda _settings, pack_name: SimpleNamespace(name=pack_name, backend_preference=["diffusers"]),
    )
    monkeypatch.setattr(worker, "GenerationSession", _FakeRplusSession, raising=False)

    result = runner.invoke(
        cli,
        [
            "rplus-compare",
            "--prompt",
            "benchmark prompt",
            "--pack",
            "Rayzist_bf16",
            "--seed",
            "17",
            "--width",
            "8",
            "--height",
            "8",
            "--output-dir",
            str(temp_app_paths.outputs_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(_FakeRplusSession.requests) == 2

    standard_request, rplus_request = _FakeRplusSession.requests
    assert standard_request.prompt == rplus_request.prompt == "benchmark prompt"
    assert standard_request.seed == rplus_request.seed == 17
    assert standard_request.width == rplus_request.width == 8
    assert standard_request.height == rplus_request.height == 8
    assert standard_request.inference_process == "standard"
    assert rplus_request.inference_process == "rplus"

    csv_reports = sorted(temp_app_paths.data_dir.glob("rplus_compare_*.csv"))
    jsonl_reports = sorted(temp_app_paths.data_dir.glob("rplus_compare_*.jsonl"))
    assert len(csv_reports) == 1
    assert len(jsonl_reports) == 1

    with csv_reports[0].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["standard_label"] == "Standard"
    assert row["rplus_label"] == "R+"
    assert row["standard_inference_process"] == "standard"
    assert row["rplus_inference_process"] == "rplus"
    assert row["mse"] not in {"", None}
    assert Path(row["standard_output_path"]).exists()
    assert Path(row["rplus_output_path"]).exists()

    jsonl_rows = [json.loads(line) for line in jsonl_reports[0].read_text(encoding="utf-8").splitlines()]
    assert len(jsonl_rows) == 1
    assert jsonl_rows[0]["standard_label"] == "Standard"
    assert jsonl_rows[0]["rplus_label"] == "R+"
