from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli.main import cli


def test_procedural_latent_preview_creates_expected_files(workspace_tmp_path: Path) -> None:
    runner = CliRunner()
    output_dir = workspace_tmp_path / "latent-preview"

    result = runner.invoke(
        cli,
        [
            "procedural-latent-preview",
            "--width",
            "1024",
            "--height",
            "1024",
            "--count",
            "2",
            "--seed-start",
            "11",
            "--creativity",
            "3",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    manifest_path = output_dir / "manifest.json"
    contact_sheet_path = output_dir / "contact_sheet.png"
    first_preview = output_dir / "seed_000011_preview.png"
    second_preview = output_dir / "seed_000012_preview.png"

    assert manifest_path.exists()
    assert contact_sheet_path.exists()
    assert first_preview.exists()
    assert second_preview.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["count"] == 2
    assert manifest["seed_start"] == 11
    assert manifest["creativity"] == 3
    assert manifest["latent_width"] == 128
    assert manifest["latent_height"] == 128
    assert len(manifest["items"]) == 2
    assert manifest["items"][0]["seed"] == 11
    assert manifest["items"][0]["creativity"] == 3
    assert manifest["items"][1]["seed"] == 12

