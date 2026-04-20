from __future__ import annotations

import csv
import json

from PIL import Image
from typer.testing import CliRunner

from app.cli import main as cli_main
from app.cli.main import cli
from app.core import clarity as clarity_core


def test_clarity_compare_writes_reports_and_visuals(
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
    source_path = temp_app_paths.outputs_dir / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), color=(12, 34, 56)).save(source_path)

    def fake_load_settings(profile_name=None):
        return settings

    def fake_run_clarity_pipeline(*, image, variant):
        selected = clarity_core.resolve_clarity_variant(variant)
        color = {
            "current": (90, 110, 140),
            "fs_downsize_final_unsharp": (70, 100, 130),
            "fs_only": (120, 140, 170),
        }[selected.label]
        return clarity_core.ClarityResult(
            image=Image.new("RGB", image.size, color=color),
            duration_ms={
                "current": 101,
                "fs_downsize_final_unsharp": 88,
                "fs_only": 133,
            }[selected.label],
            source_width=image.width,
            source_height=image.height,
            working_width=image.width * 2,
            working_height=image.height * 2,
            engine_name=clarity_core.CLARITY_ENGINE_NAME,
            variant_label=selected.label,
            device="cpu",
            step_timings_ms={
                "clarity_resize_ms": 1,
                "clarity_fs_ms": 4,
                "clarity_pre_downscale_unsharp_ms": 8 if selected.pre_downscale_unsharp else 0,
                "clarity_downscale_ms": 7,
                "clarity_post_downscale_unsharp_ms": 8 if selected.post_downscale_unsharp else 0,
            },
        )

    monkeypatch.setattr(cli_main, "load_settings", fake_load_settings)
    monkeypatch.setattr(clarity_core, "run_clarity_pipeline", fake_run_clarity_pipeline)

    output_dir = temp_app_paths.outputs_dir / "clarity_suite"
    result = runner.invoke(
        cli,
        [
            "clarity-compare",
            "--input",
            str(source_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    csv_reports = sorted(temp_app_paths.data_dir.glob("clarity_compare_*.csv"))
    jsonl_reports = sorted(temp_app_paths.data_dir.glob("clarity_compare_*.jsonl"))
    assert len(csv_reports) == 1
    assert len(jsonl_reports) == 1

    with csv_reports[0].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [row["variant"] for row in rows] == ["current", "fs_downsize_final_unsharp", "fs_only"]
    assert {row["status"] for row in rows} == {"success"}

    jsonl_rows = [json.loads(line) for line in jsonl_reports[0].read_text(encoding="utf-8").splitlines()]
    assert len(jsonl_rows) == 3
    assert {row["variant"] for row in jsonl_rows} == {"current", "fs_downsize_final_unsharp", "fs_only"}

    contact_sheet = output_dir / "clarity_compare_contact_sheet.png"
    manifest = output_dir / "manifest.json"
    assert contact_sheet.exists()
    assert manifest.exists()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["profile"] == "balanced"
    assert manifest_payload["variants"] == ["current", "fs_downsize_final_unsharp", "fs_only"]
    assert set(manifest_payload["outputs"]) == {"current", "fs_downsize_final_unsharp", "fs_only"}


def test_clarity_compare_fails_fast_when_input_missing(
    monkeypatch,
    temp_app_paths,
    make_app_settings,
) -> None:
    runner = CliRunner()
    settings = make_app_settings(paths=temp_app_paths)
    monkeypatch.setattr(cli_main, "load_settings", lambda profile_name=None: settings)

    result = runner.invoke(
        cli,
        [
            "clarity-compare",
            "--input",
            str(temp_app_paths.outputs_dir / "missing.png"),
        ],
    )

    assert result.exit_code == 1
    assert "Input image not found:" in result.stdout
