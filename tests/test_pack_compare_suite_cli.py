from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from PIL import Image
from typer.testing import CliRunner

import app.core.worker as worker
from app.cli import main as cli_main
from app.cli.main import cli
from app.core.backends.diffusers_zimage import GenerationResult
from app.core.memory import CudaMemorySnapshot


class _FakeSession:
    def __init__(self, settings, model_pack, resource_tier=None):
        self._settings = settings
        self._model_pack = model_pack
        self._call_count = 0

    def generate(self, request):
        self._call_count += 1
        name = self._model_pack.name.lower()
        if "bf16" in name:
            base_color = 20
            reserved_bytes = 8_200_000_000
            duration_ms = 1500 if self._call_count == 1 else 1350
        elif "storage" in name:
            base_color = 36
            reserved_bytes = 7_500_000_000
            duration_ms = 1325 if self._call_count == 1 else 1200
        elif "mixed" in name:
            base_color = 52
            reserved_bytes = 7_100_000_000
            duration_ms = 1260 if self._call_count == 1 else 1120
        else:
            base_color = 68
            reserved_bytes = 6_900_000_000
            duration_ms = 1240 if self._call_count == 1 else 1100
        image = Image.new(
            "RGB",
            (request.width, request.height),
            color=(base_color + int(request.seed or 0), 48 + len(name), 72),
        )
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
                allocated_bytes=reserved_bytes - 900_000_000,
                reserved_bytes=reserved_bytes,
                max_allocated_bytes=reserved_bytes - 400_000_000,
                max_reserved_bytes=reserved_bytes + 250_000_000,
            ),
            runtime_profile=self._settings.runtime_profile.name,
            resource_tier=self._settings.resource_tier_controller.current().name,
            execution_mode="full_cuda",
        )

    def recycle(self, reason: str) -> None:
        self._call_count = 0


def test_pack_compare_suite_writes_reports_and_contact_sheets(
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

    def fake_load_settings(profile_name=None):
        return settings

    def fake_load_runtime_pack(_settings, pack_name: str):
        return SimpleNamespace(name=pack_name, backend_preference=["diffusers"])

    monkeypatch.setattr(cli_main, "load_settings", fake_load_settings)
    monkeypatch.setattr(cli_main, "_load_runtime_pack_or_exit", fake_load_runtime_pack)
    monkeypatch.setattr(worker, "GenerationSession", _FakeSession, raising=False)

    result = runner.invoke(
        cli,
        [
            "pack-compare-suite",
            "--iterations",
            "1",
            "--width",
            "8",
            "--height",
            "8",
            "--output-dir",
            str(temp_app_paths.outputs_dir),
            "--no-warmup",
        ],
    )

    assert result.exit_code == 0, result.stdout

    csv_reports = sorted(temp_app_paths.data_dir.glob("pack_compare_suite_*.csv"))
    jsonl_reports = sorted(temp_app_paths.data_dir.glob("pack_compare_suite_*.jsonl"))
    assert len(csv_reports) == 1
    assert len(jsonl_reports) == 1

    with csv_reports[0].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert rows[0]["prompt_key"] == "portrait"
    assert rows[0]["pack"] == "Rayzist_bf16"
    assert rows[1]["pack"] == "Rayzist_bf16__auto_fp8_storage"
    assert rows[1]["mse"] not in {"", None}

    jsonl_rows = [json.loads(line) for line in jsonl_reports[0].read_text(encoding="utf-8").splitlines()]
    assert len(jsonl_rows) == 12
    assert {row["prompt_key"] for row in jsonl_rows} == {"portrait", "action", "anime"}

    contact_sheets = sorted(temp_app_paths.outputs_dir.rglob("contact_sheet_*.png"))
    assert len(contact_sheets) == 3
    assert all(path.exists() for path in contact_sheets)

