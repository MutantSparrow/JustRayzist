from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from PIL import Image
from typer.testing import CliRunner

import app.core.worker as worker
import app.core.upscale as upscale_module
from app.cli import main as cli_main
from app.cli.main import cli
from app.core.backends.diffusers_zimage import GenerationResult
from app.core.memory import CudaMemorySnapshot


class _FakeSession:
    def __init__(self, settings, model_pack, resource_tier=None):
        self._settings = settings
        self._model_pack = model_pack
        self._resource_tier = resource_tier or settings.resource_tier_controller.current()

    def generate(self, request):
        tier_name = self._resource_tier.name
        duration_ms = {
            "high": 900,
            "balanced": 1200,
            "constrained": 1500,
        }[tier_name]
        image = Image.new(
            "RGB",
            (request.width, request.height),
            color=(30 + len(self._model_pack.name), 40 + len(tier_name), 90),
        )
        reserved_bytes = {
            "high": 9_000_000_000,
            "balanced": 7_600_000_000,
            "constrained": 6_800_000_000,
        }[tier_name]
        return GenerationResult(
            image=image,
            seed=request.seed,
            steps=request.steps or self._settings.runtime_profile.steps_default,
            guidance_scale=(
                request.guidance_scale
                if request.guidance_scale is not None
                else self._settings.runtime_profile.guidance_scale_default
            ),
            scheduler_mode="euler",
            backend="diffusers_zimage",
            device="cuda",
            duration_ms=duration_ms,
            prompt_original=request.prompt,
            prompt_effective=request.prompt,
            prompt_enhanced=False,
            cuda_memory_after=CudaMemorySnapshot(
                allocated_bytes=reserved_bytes - 500_000_000,
                reserved_bytes=reserved_bytes,
                max_allocated_bytes=reserved_bytes - 200_000_000,
                max_reserved_bytes=reserved_bytes + 300_000_000,
            ),
            runtime_profile=self._settings.runtime_profile.name,
            resource_tier=tier_name,
            execution_mode="model_offload" if tier_name != "high" else "full_cuda",
            execution_mode_initial="full_cuda",
            execution_mode_before_generate="model_offload" if tier_name != "high" else "full_cuda",
            execution_mode_after_generate="model_offload",
            preflight_fallback_triggered=tier_name == "high",
            cuda_free_before_load_bytes=10_000_000_000,
            cuda_free_after_load_bytes=9_500_000_000,
            cuda_free_before_generate_bytes=9_000_000_000,
            cuda_free_after_generate_bytes=8_500_000_000,
        )

    def recycle(self, reason: str) -> None:
        return None


class _FakeInferenceService:
    def __init__(self, settings):
        self._settings = settings

    def _resolve_runtime_pack(self, pack_name: str):
        tier_name = self._settings.resource_tier_controller.current().name
        selected_pack = SimpleNamespace(name=pack_name, backend_preference=["diffusers"])
        if tier_name == "constrained":
            effective_pack = SimpleNamespace(
                name=f"{pack_name}__auto_fp8_storage",
                backend_preference=["diffusers"],
                derived_strategy="fp8_storage",
            )
        else:
            effective_pack = SimpleNamespace(
                name=pack_name,
                backend_preference=["diffusers"],
                derived_strategy=None,
            )
        return selected_pack, effective_pack, self._settings.resource_tier_controller.current()

    def resolve_runtime_pack(self, pack_name: str, apply_resource_tier_policy=True):
        return self._resolve_runtime_pack(pack_name)


class _FakeUpscaleResult:
    def __init__(self, image):
        self.image = image
        self.scale_factor = 2
        self.device = "cuda"
        self.precision = "fp16"
        self.tile_size = 256
        self.tile_overlap = 24
        self.duration_ms = 420
        self.source_width = image.width
        self.source_height = image.height
        self.output_width = image.width
        self.output_height = image.height
        self.architecture = "rrdb"
        self.norm_kind = None
        self.cuda_memory_before = None
        self.cuda_memory_after = CudaMemorySnapshot(
            allocated_bytes=3_000_000_000,
            reserved_bytes=3_400_000_000,
            max_allocated_bytes=3_100_000_000,
            max_reserved_bytes=3_600_000_000,
        )
        self.process_memory_before = None
        self.process_memory_after = None

    def telemetry_dict(self):
        return {
            "scale_factor": self.scale_factor,
            "device": self.device,
            "precision": self.precision,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "duration_ms": self.duration_ms,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "architecture": self.architecture,
            "norm_kind": self.norm_kind,
            "cuda_memory_before": None,
            "cuda_memory_after": self.cuda_memory_after.to_dict(),
            "process_memory_before": None,
            "process_memory_after": None,
        }


def test_prompt_grid_benchmark_writes_reports_and_visuals(
    monkeypatch,
    temp_app_paths,
    make_app_settings,
) -> None:
    runner = CliRunner()
    checkpoint_path = temp_app_paths.models_dir / "upscaler" / "2x_RealESRGAN_x2plus.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"ok")

    def make_settings(profile_name=None):
        if profile_name is None:
            return make_app_settings(
                paths=temp_app_paths,
                runtime_profile_name="balanced",
                resource_tier_name="high",
                auto_resource_tier=True,
            )
        return make_app_settings(
            paths=temp_app_paths,
            runtime_profile_name=profile_name,
            resource_tier_name=profile_name,
            override_profile_name=profile_name,
            auto_resource_tier=False,
        )

    monkeypatch.setattr(cli_main, "load_settings", make_settings)
    monkeypatch.setattr(cli_main, "InferenceService", _FakeInferenceService)
    monkeypatch.setattr(worker, "GenerationSession", _FakeSession, raising=False)
    monkeypatch.setattr(
        upscale_module,
        "upscale_image",
        lambda image, checkpoint_path, profile_name: _FakeUpscaleResult(image),
    )

    result = runner.invoke(
        cli,
        [
            "prompt-grid-benchmark",
            "--pack",
            "Rayzist_bf16",
            "--prompt",
            "prompt one",
            "--prompt",
            "prompt two",
            "--prompt",
            "prompt three",
            "--width",
            "8",
            "--height",
            "8",
            "--output-dir",
            str(temp_app_paths.outputs_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    csv_reports = sorted(temp_app_paths.data_dir.glob("prompt_grid_benchmark_*.csv"))
    jsonl_reports = sorted(temp_app_paths.data_dir.glob("prompt_grid_benchmark_*.jsonl"))
    assert len(csv_reports) == 1
    assert len(jsonl_reports) == 1

    with csv_reports[0].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36
    assert {row["row_type"] for row in rows} == {"generation", "upscale", "summary"}
    generation_rows = [row for row in rows if row["row_type"] == "generation"]
    summary_rows = [row for row in rows if row["row_type"] == "summary"]
    assert len(generation_rows) == 12
    assert len(summary_rows) == 12
    assert {row["scenario_label"] for row in summary_rows} == {
        "forced_high",
        "forced_balanced",
        "forced_constrained",
        "auto",
    }
    constrained_summary = next(row for row in summary_rows if row["scenario_label"] == "forced_constrained")
    assert constrained_summary["effective_pack"] == "Rayzist_bf16__auto_fp8_storage"
    auto_summary = next(row for row in summary_rows if row["scenario_label"] == "auto")
    assert auto_summary["detected_resource_tier"] == "high"
    assert auto_summary["preflight_fallback_triggered"] in {"True", "true", "1"}

    jsonl_rows = [json.loads(line) for line in jsonl_reports[0].read_text(encoding="utf-8").splitlines()]
    assert len(jsonl_rows) == 36

    generation_sheet = temp_app_paths.outputs_dir / "generation_contact_sheet.png"
    upscale_sheet = temp_app_paths.outputs_dir / "upscale_contact_sheet.png"
    dashboard = temp_app_paths.outputs_dir / "benchmark_dashboard.png"
    manifest = temp_app_paths.outputs_dir / "manifest.json"
    assert generation_sheet.exists()
    assert upscale_sheet.exists()
    assert dashboard.exists()
    assert manifest.exists()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["pack"] == "Rayzist_bf16"
    assert len(manifest_payload["prompts"]) == 3
